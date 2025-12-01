# modules/lif_triton.py

import torch
import triton
import triton.language as tl

# ============== Forward Kernel ==============
@triton.jit
def lif_fwd_kernel(
    x_ptr,          # 输入电流 (num_neurons, T)
    v_init_ptr,     # 初始膜电位 (num_neurons,)
    spike_ptr,      # 输出脉冲 (num_neurons, T)
    v_final_ptr,    # 最终膜电位 (num_neurons,)
    v_trace_ptr,    # 膜电位轨迹 (num_neurons, T)，用于backward
    num_neurons,
    T,
    decay: tl.constexpr,
    v_threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个program处理BLOCK_SIZE个神经元
    pid = tl.program_id(0)
    neuron_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = neuron_idx < num_neurons
    
    # 加载初始膜电位
    v = tl.load(v_init_ptr + neuron_idx, mask=mask, other=0.0)
    
    # 时间循环（串行，因为v有依赖）
    for t in range(T):
        # 计算当前时间步的偏移
        offset = neuron_idx * T + t
        
        # 加载输入
        x_t = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        # LIF动力学: v = v * decay + input
        v = v * decay + x_t
        
        # 保存reset前的膜电位（用于surrogate gradient）
        tl.store(v_trace_ptr + offset, v, mask=mask)
        
        # 发放判断: spike = (v >= threshold)
        spike = (v >= v_threshold).to(tl.float32)
        tl.store(spike_ptr + offset, spike, mask=mask)
        
        # Soft reset: v = v - spike * threshold
        v = v - spike * v_threshold
    
    # 保存最终膜电位
    tl.store(v_final_ptr + neuron_idx, v, mask=mask)


# ============== Backward Kernel ==============
@triton.jit
def lif_bwd_kernel(
    grad_spike_ptr,   # dL/d_spike (num_neurons, T)
    v_trace_ptr,      # 前向保存的膜电位 (num_neurons, T)
    grad_x_ptr,       # dL/d_x 输出 (num_neurons, T)
    grad_v_init_ptr,  # dL/d_v_init 输出 (num_neurons,)
    num_neurons,
    T,
    decay: tl.constexpr,
    v_threshold: tl.constexpr,
    surrogate_alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    neuron_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = neuron_idx < num_neurons
    
    # 从最后一个时间步反向传播
    # grad_v 是 dL/dv 在当前时间步的累积
    grad_v = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for t in range(T - 1, -1, -1):
        offset = neuron_idx * T + t
        
        # 加载梯度和膜电位
        grad_spike_t = tl.load(grad_spike_ptr + offset, mask=mask, other=0.0)
        v_t = tl.load(v_trace_ptr + offset, mask=mask, other=0.0)
        
        # Surrogate gradient: sigmoid'(alpha * (v - threshold))
        # sigmoid(x) * (1 - sigmoid(x)) * alpha
        shifted_v = surrogate_alpha * (v_t - v_threshold)
        sig = 1.0 / (1.0 + tl.exp(-shifted_v))
        surrogate_grad = surrogate_alpha * sig * (1.0 - sig)
        
        # dL/dv_t = grad_spike_t * surrogate + grad_v * (1 - spike * 0)
        # 对于soft reset: dv_after/dv_before = 1 - spike (但spike是不可导的，用surrogate)
        # 简化处理：忽略reset对梯度的影响，或用detach_reset
        grad_v = grad_v + grad_spike_t * surrogate_grad
        
        # dL/dx_t = dL/dv_t (因为 v = v*decay + x，所以 dv/dx = 1)
        tl.store(grad_x_ptr + offset, grad_v, mask=mask)
        
        # 传播到上一时间步: dL/dv_{t-1} = dL/dv_t * decay
        grad_v = grad_v * decay
    
    # 保存对初始膜电位的梯度
    tl.store(grad_v_init_ptr + neuron_idx, grad_v, mask=mask)


# ============== Autograd封装 ==============
class LIFTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v_init, decay, v_threshold, surrogate_alpha):
        """
        x: (N, T, ...) 输入电流
        v_init: (N, ...) 初始膜电位
        """
        # 保存原始shape
        original_shape = x.shape
        N, T = x.shape[0], x.shape[1]
        spatial_shape = x.shape[2:]
        
        # Flatten: (N, T, C, H, W) -> (N*C*H*W, T)
        x_flat = x.permute(0, *range(2, x.dim()), 1).reshape(-1, T).contiguous()
        v_init_flat = v_init.reshape(-1).contiguous()
        num_neurons = x_flat.shape[0]
        
        # 分配输出
        spike_flat = torch.empty_like(x_flat)
        v_final_flat = torch.empty_like(v_init_flat)
        v_trace_flat = torch.empty_like(x_flat)  # 保存用于backward
        
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(num_neurons, BLOCK_SIZE),)
        
        lif_fwd_kernel[grid](
            x_flat, v_init_flat, spike_flat, v_final_flat, v_trace_flat,
            num_neurons, T,
            decay, v_threshold,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape back: (N*C*H*W, T) -> (N, T, C, H, W)
        spike = spike_flat.view(N, *spatial_shape, T).permute(0, -1, *range(1, len(spatial_shape)+1)).contiguous()
        v_final = v_final_flat.view(N, *spatial_shape)
        
        # 保存用于backward
        ctx.save_for_backward(v_trace_flat)
        ctx.num_neurons = num_neurons
        ctx.T = T
        ctx.original_shape = original_shape
        ctx.spatial_shape = spatial_shape
        ctx.decay = decay
        ctx.v_threshold = v_threshold
        ctx.surrogate_alpha = surrogate_alpha
        
        return spike, v_final
    
    @staticmethod
    def backward(ctx, grad_spike, grad_v_final):
        v_trace_flat, = ctx.saved_tensors
        N = ctx.original_shape[0]
        T = ctx.T
        
        # Flatten grad_spike
        grad_spike_flat = grad_spike.permute(0, *range(2, grad_spike.dim()), 1).reshape(-1, T).contiguous()
        
        # 分配梯度输出
        grad_x_flat = torch.empty_like(grad_spike_flat)
        grad_v_init_flat = torch.empty(ctx.num_neurons, device=grad_spike.device, dtype=grad_spike.dtype)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(ctx.num_neurons, BLOCK_SIZE),)
        
        lif_bwd_kernel[grid](
            grad_spike_flat, v_trace_flat, grad_x_flat, grad_v_init_flat,
            ctx.num_neurons, T,
            ctx.decay, ctx.v_threshold, ctx.surrogate_alpha,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape back
        grad_x = grad_x_flat.view(N, *ctx.spatial_shape, T).permute(0, -1, *range(1, len(ctx.spatial_shape)+1)).contiguous()
        grad_v_init = grad_v_init_flat.view(N, *ctx.spatial_shape)
        
        return grad_x, grad_v_init, None, None, None


# ============== 易用接口 ==============
class LIFNodeTriton(torch.nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0, surrogate_alpha=4.0):
        super().__init__()
        self.decay = 1.0 - 1.0 / tau
        self.v_threshold = v_threshold
        self.surrogate_alpha = surrogate_alpha
        self.v = None
    
    def reset(self):
        self.v = None
    
    def forward(self, x):
        # x: (N, T, C, H, W)
        if self.v is None or self.v.shape != x[:, 0].shape:
            self.v = torch.zeros_like(x[:, 0])
        
        spike, self.v = LIFTritonFunction.apply(
            x, self.v, self.decay, self.v_threshold, self.surrogate_alpha
        )
        return spike