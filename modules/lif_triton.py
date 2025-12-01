"""
Optimized Triton LIF Neuron Implementation

Key optimizations:
1. Native float16 computation (no fp32 conversion)
2. Fused operations in single kernel
3. Optimized memory layout: (N*C*H*W, T) - time as last dim for coalesced access
4. Minimal tensor reshaping
"""

import torch
import triton
import triton.language as tl


@triton.jit
def lif_fwd_kernel(
    x_ptr,          # (num_neurons, T)
    v_init_ptr,     # (num_neurons,)
    spike_ptr,      # (num_neurons, T)
    v_final_ptr,    # (num_neurons,)
    v_pre_ptr,      # (num_neurons, T) - membrane potential before reset, for backward
    num_neurons,
    T: tl.constexpr,
    decay: tl.constexpr,
    v_threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    neuron_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = neuron_idx < num_neurons
    
    # Load initial membrane potential (same dtype as input)
    v = tl.load(v_init_ptr + neuron_idx, mask=mask, other=0.0)
    
    # Time loop - T is constexpr so compiler can unroll
    for t in range(T):
        offset = neuron_idx * T + t
        
        x_t = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        # LIF dynamics
        v = v * decay + x_t
        
        # Store pre-reset potential for surrogate gradient
        tl.store(v_pre_ptr + offset, v, mask=mask)
        
        # Spike generation (hard threshold)
        spike = (v >= v_threshold)
        tl.store(spike_ptr + offset, spike, mask=mask)
        
        # Soft reset
        v = v - spike.to(v.dtype) * v_threshold
    
    tl.store(v_final_ptr + neuron_idx, v, mask=mask)


@triton.jit  
def lif_bwd_kernel(
    grad_spike_ptr,   # (num_neurons, T)
    v_pre_ptr,        # (num_neurons, T)
    grad_x_ptr,       # (num_neurons, T) output
    grad_v_init_ptr,  # (num_neurons,) output
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
    
    # Accumulator for gradient w.r.t. membrane potential
    grad_v = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # Use fp32 for gradient accumulation only
    
    # Backward through time
    for t in range(T - 1, -1, -1):
        offset = neuron_idx * T + t
        
        grad_spike_t = tl.load(grad_spike_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        v_pre = tl.load(v_pre_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        
        # Surrogate gradient: sigmoid derivative
        x = alpha * (v_pre - v_threshold)
        sig = tl.sigmoid(x)
        surrogate = alpha * sig * (1.0 - sig)
        
        # dL/dv = dL/ds * ds/dv + dL/dv_next * dv_next/dv
        # With soft reset and detached reset: dv_next/dv ≈ decay
        grad_v = grad_v + grad_spike_t * surrogate
        
        # dL/dx = dL/dv (since v = decay*v + x)
        # Store as the original dtype of the pointer
        tl.store(grad_x_ptr + offset, grad_v.to(grad_x_ptr.dtype.element_ty), mask=mask)
        
        grad_v = grad_v * decay
    
    tl.store(grad_v_init_ptr + neuron_idx, grad_v.to(grad_v_init_ptr.dtype.element_ty), mask=mask)


class LIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, v_init, T, decay, v_threshold, alpha):
        """
        x_flat: (num_neurons, T) - already flattened
        v_init: (num_neurons,)
        """
        num_neurons = x_flat.shape[0]
        
        # Ensure contiguous inputs for kernel safety
        x_flat = x_flat.contiguous()
        v_init = v_init.contiguous()
        
        # Allocate outputs
        spike_flat = torch.empty_like(x_flat)
        v_final = torch.empty_like(v_init)
        v_pre = torch.empty_like(x_flat)  # For backward
        
        # Kernel launch
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
        
        # Critical: Ensure contiguous memory for Triton pointer math
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
    
    Expected input: (N, T, C, H, W) or (N, T, C)
    Output: same shape as input
    """
    def __init__(self, tau=2.0, v_threshold=1.0, surrogate_alpha=4.0, detach_reset=True, **kwargs):
        super().__init__()
        self.decay = 1.0 - 1.0 / tau
        self.v_threshold = v_threshold
        self.alpha = surrogate_alpha
        # detach_reset is implicit in our backward (we don't backprop through reset)
        
        self.register_buffer('v', None, persistent=False)
        self._input_shape = None
    
    def reset(self):
        self.v = None
        self._input_shape = None
    
    def forward(self, x):
        """
        x: (N, T, ...) where ... can be (C, H, W) or (C,)
        """
        N, T = x.shape[:2]
        spatial_shape = x.shape[2:]
        num_neurons = N * int(torch.tensor(spatial_shape).prod().item()) if spatial_shape else N
        
        # Initialize membrane potential
        if self.v is None or self.v.shape[0] != num_neurons:
            self.v = torch.zeros(num_neurons, device=x.device, dtype=x.dtype)
        
        # Flatten to (num_neurons, T) - optimal for our kernel
        # From (N, T, C, H, W) -> (N, C, H, W, T) -> (N*C*H*W, T)
        if len(spatial_shape) > 0:
            # Move T to last, then flatten spatial
            x_flat = x.permute(0, *range(2, x.dim()), 1).reshape(num_neurons, T)
        else:
            x_flat = x.reshape(num_neurons, T)
        
        # Ensure contiguous for kernel
        x_flat = x_flat.contiguous()
        
        # Run kernel
        spike_flat, self.v = LIFFunction.apply(
            x_flat, self.v, T, self.decay, self.v_threshold, self.alpha
        )
        
        # Reshape back to (N, T, C, H, W)
        if len(spatial_shape) > 0:
            spike = spike_flat.view(N, *spatial_shape, T).permute(0, -1, *range(1, len(spatial_shape)+1))
        else:
            spike = spike_flat.view(N, T)
        
        return spike.contiguous()

    def extra_repr(self):
        return f'decay={self.decay:.3f}, threshold={self.v_threshold}, alpha={self.alpha}'