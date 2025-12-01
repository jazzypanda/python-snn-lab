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
    
    # Load initial membrane potential using pointer's dtype
    v = tl.load(v_init_ptr + neuron_idx, mask=mask, other=0.0)
    
    for t in range(T):
        offset = neuron_idx * T + t
        x_t = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        # Compute in float32 for precision
        v_f32 = v.to(tl.float32)
        x_t_f32 = x_t.to(tl.float32)
        
        v_f32 = v_f32 * decay + x_t_f32
        
        # Store v_pre for backward (in original dtype)
        tl.store(v_pre_ptr + offset, v_f32.to(v_pre_ptr.dtype.element_ty), mask=mask)
        
        spike = (v_f32 >= v_threshold)
        tl.store(spike_ptr + offset, spike, mask=mask)
        
        # Soft reset
        v_f32 = v_f32 - spike.to(tl.float32) * v_threshold
        v = v_f32.to(v.dtype) # Cast back to register dtype
    
    tl.store(v_final_ptr + neuron_idx, v, mask=mask)


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
    def __init__(self, tau=2.0, v_threshold=1.0, surrogate_alpha=4.0, detach_reset=True):
        super().__init__()
        self.decay = 1.0 - 1.0 / tau
        self.v_threshold = v_threshold
        self.alpha = surrogate_alpha
        self.register_buffer('v', None, persistent=False)
    
    def reset(self):
        self.v = None
    
    def forward(self, x):
        # x: (N, T, ...)
        N, T = x.shape[:2]
        # Flatten: (N, T, ...) -> (N * ... , T)
        x_flat = x.flatten(0, 1) if x.dim() == 2 else x.transpose(1, -1).flatten(0, -2) 
        # Logic check: 
        # If (N, T, C, H, W) -> want (N*C*H*W, T)
        # transpose(1, -1) -> (N, W, C, H, T) ... messy.
        
        # Cleaner flatten logic:
        if x.dim() > 2:
            # (N, T, C, H, W) -> permute T to end -> (N, C, H, W, T) -> flatten -> (N*C*H*W, T)
            x_flat = x.permute(0, *range(2, x.dim()), 1).flatten(0, -2)
        else:
            x_flat = x
            
        num_neurons = x_flat.shape[0]

        if self.v is None or self.v.shape[0] != num_neurons:
            self.v = torch.zeros(num_neurons, device=x.device, dtype=x.dtype)
        
        # Execute
        spike_flat, self.v = LIFFunction.apply(
            x_flat, self.v, T, self.decay, self.v_threshold, self.alpha
        )
        
        # Reshape back
        if x.dim() > 2:
            # (N*C*H*W, T) -> (N, C, H, W, T) -> (N, T, C, H, W)
            input_spatial = x.shape[2:]
            spike = spike_flat.view(N, *input_spatial, T).permute(0, -1, *range(1, x.dim()-1))
        else:
            spike = spike_flat
            
        return spike