import torch
import triton
import triton.language as tl

@triton.jit
def lif_fwd_kernel(
    x_ptr,          # (num_neurons, T)
    v_init_ptr,     # (num_neurons,)
    spike_ptr,      # (num_neurons, T)
    v_final_ptr,    # (num_neurons,)
    v_pre_ptr,      # (num_neurons, T)
    num_neurons,
    T: tl.constexpr,
    decay: tl.constexpr,
    v_threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    neuron_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = neuron_idx < num_neurons
    
    # Load initial membrane potential and immediately promote to fp32
    v_init = tl.load(v_init_ptr + neuron_idx, mask=mask, other=0.0)
    v = v_init.to(tl.float32)
    
    for t in range(T):
        offset = neuron_idx * T + t
        x_t = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        
        # LIF dynamics (all in fp32)
        v = v * decay + x_t
        
        # Store v_pre for backward (cast to ptr dtype)
        tl.store(v_pre_ptr + offset, v.to(v_pre_ptr.dtype.element_ty), mask=mask)
        
        spike = (v >= v_threshold)
        tl.store(spike_ptr + offset, spike, mask=mask)
        
        # Soft reset
        v = v - spike.to(tl.float32) * v_threshold
    
    # Store final v (cast to ptr dtype)
    tl.store(v_final_ptr + neuron_idx, v.to(v_final_ptr.dtype.element_ty), mask=mask)


@triton.jit  
def lif_bwd_kernel(
    grad_spike_ptr,   # (num_neurons, T)
    v_pre_ptr,        # (num_neurons, T)
    grad_x_ptr,       # (num_neurons, T) 
    grad_v_init_ptr,  # (num_neurons,) 
    num_neurons,
    T: tl.constexpr,
    decay: tl.constexpr,
    v_threshold: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    neuron_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = neuron_idx < num_neurons
    
    grad_v = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for t in range(T - 1, -1, -1):
        offset = neuron_idx * T + t
        
        # Load in fp32
        grad_spike_t = tl.load(grad_spike_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        v_pre = tl.load(v_pre_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        
        # Surrogate (Sigmoid derivative)
        x_val = alpha * (v_pre - v_threshold)
        sig = tl.sigmoid(x_val)
        surrogate = alpha * sig * (1.0 - sig)
        
        grad_v = grad_v + grad_spike_t * surrogate
        
        # Store grad_x in original dtype
        tl.store(grad_x_ptr + offset, grad_v.to(grad_x_ptr.dtype.element_ty), mask=mask)
        
        grad_v = grad_v * decay
    
    tl.store(grad_v_init_ptr + neuron_idx, grad_v.to(grad_v_init_ptr.dtype.element_ty), mask=mask)


class LIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, v_init, T, decay, v_threshold, alpha):
        num_neurons = x_flat.shape[0]
        
        # Safety: Contiguous required for pointer math
        x_flat = x_flat.contiguous()
        v_init = v_init.contiguous()
        
        spike_flat = torch.empty_like(x_flat)
        v_final = torch.empty_like(v_init)
        v_pre = torch.empty_like(x_flat) 
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(num_neurons, BLOCK_SIZE),)
        
        lif_fwd_kernel[grid](
            x_flat, v_init, spike_flat, v_final, v_pre,
            num_neurons, T, decay, v_threshold,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        ctx.save_for_backward(v_pre)
        ctx.num_neurons = num_neurons
        ctx.T = T
        ctx.decay = decay
        ctx.v_threshold = v_threshold
        ctx.alpha = alpha
        
        return spike_flat, v_final
    
    @staticmethod
    def backward(ctx, grad_spike, grad_v_final):
        v_pre, = ctx.saved_tensors
        grad_spike = grad_spike.contiguous()
        v_pre = v_pre.contiguous()
        
        grad_x = torch.empty_like(grad_spike)
        grad_v_init = torch.empty(ctx.num_neurons, device=grad_spike.device, dtype=grad_spike.dtype)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(ctx.num_neurons, BLOCK_SIZE),)
        
        lif_bwd_kernel[grid](
            grad_spike, v_pre, grad_x, grad_v_init,
            ctx.num_neurons, ctx.T, ctx.decay, ctx.v_threshold, ctx.alpha,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return grad_x, grad_v_init, None, None, None, None


class LIFNodeTriton(torch.nn.Module):
    """
    Optimized LIF neuron using Triton kernels.
    
    Expected input: (N*T, ...) flattened batch
    Output: (N*T, ...) flattened batch
    """
    def __init__(self, T=4, tau=2.0, v_threshold=1.0, surrogate_alpha=4.0, detach_reset=True, **kwargs):
        super().__init__()
        self.T = T
        self.decay = 1.0 - 1.0 / tau
        self.v_threshold = v_threshold
        self.alpha = surrogate_alpha
        # detach_reset is implicit in our backward (we don't backprop through reset)
        
        self.register_buffer('v', None, persistent=False)
    
    def reset(self):
        self.v = None
    
    def forward(self, x):
        """
        x: (Batch, ...) where Batch = N * T
        Output: (Batch, ...)
        """
        # 1. Check dimensions
        Batch = x.shape[0]
        if Batch % self.T != 0:
            raise ValueError(f"Batch size {Batch} must be divisible by T={self.T}")
        
        N = Batch // self.T
        
        # 2. Prepare for Kernel: We need (Neurons, T) layout where Neurons = N * Spatial
        # Input x is (N*T, C, H, W) or (N*T, C)
        
        if x.dim() > 2:
            # Case: Conv Layer Output (N*T, C, H, W)
            # We want to group T together.
            # View as (N, T, C, H, W)
            # Permute to (N, C, H, W, T) to get T last
            # Flatten to (N*C*H*W, T)
            x_reshaped = x.view(N, self.T, *x.shape[1:])
            x_flat = x_reshaped.permute(0, *range(2, x_reshaped.dim()), 1).flatten(0, -2)
        else:
            # Case: FC Layer Output (N*T, C)
            # View as (N, T, C) -> Permute (N, C, T) -> Flatten (N*C, T)
            x_reshaped = x.view(N, self.T, -1)
            x_flat = x_reshaped.permute(0, 2, 1).flatten(0, 1)

        num_neurons = x_flat.shape[0]

        # 3. Initialize State
        if self.v is None or self.v.shape[0] != num_neurons:
            self.v = torch.zeros(num_neurons, device=x.device, dtype=x.dtype)
        
        # 4. Run Triton Kernel
        # x_flat is contiguous by definition of flatten (usually), but safeguard:
        x_flat = x_flat.contiguous()
        
        spike_flat, self.v = LIFFunction.apply(
            x_flat, self.v, self.T, self.decay, self.v_threshold, self.alpha
        )
        
        # 5. Reshape back to (N*T, ...)
        # spike_flat is (Neurons, T) -> (N*C*H*W, T)
        if x.dim() > 2:
            # (N*C*H*W, T) -> (N, C, H, W, T) -> (N, T, C, H, W) -> (N*T, C, H, W)
            spatial_dims = x.shape[1:] # (C, H, W)
            # We need to carefully reverse the flatten/permute
            # View: (N, C, H, W, T)
            spike = spike_flat.view(N, *spatial_dims, self.T)
            # Permute: (N, T, C, H, W)
            spike = spike.permute(0, -1, *range(1, spike.dim()-1))
            # Flatten Batch: (N*T, C, H, W)
            output = spike.flatten(0, 1)
        else:
            # (N*C, T) -> (N, C, T) -> (N, T, C) -> (N*T, C)
            features = x.shape[1]
            spike = spike_flat.view(N, features, self.T)
            spike = spike.permute(0, 2, 1)
            output = spike.flatten(0, 1)
            
        return output