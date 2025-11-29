import torch
import torch.nn as nn
from .surrogate import sigmoid_surrogate

@torch.jit.script
def lif_forward_jit(x: torch.Tensor, v: torch.Tensor, decay: float, v_threshold: float, surrogate_alpha: float, detach_reset: bool):
    spikes = []
    time_steps = x.shape[1]
    
    for t in range(time_steps):
        input_t = x[:, t]
        
        v = v * decay + input_t
        
        # Inline surrogate for JIT compatibility if possible, 
        # but since we need the custom backward, we must call the function.
        # TorchScript supports calling python functions but it might break fusion.
        # For maximum speed, often a custom CUDA kernel is best.
        # Here we stick to a simple loop structure optimization.
        # To allow JIT to compile the loop, we need to be careful about `sigmoid_surrogate`.
        # If `sigmoid_surrogate` is imported, we can try to call it. 
        # BUT `sigmoid_surrogate` is likely a Python object (autograd.Function).
        # Using it inside JIT script might cause "Unknown builtin op" or fallback to python.
        
        # Let's use a trick: The surrogate is usually:
        # Forward: heaviside(v - th)
        # Backward: sigmoid_prime * grad
        
        # For JIT, we can't easily embed the custom autograd class. 
        # So we will NOT use JIT for the whole loop if we rely on that specific python autograd class.
        # Instead, we will just optimize the tensor operations *around* it? No, that's too granular.
        
        # ALTERNATIVE: Use a pure Torch implementation of surrogate that JIT understands?
        # It's hard without set_grad_enabled or custom backward in script.
        
        # Let's try to keep the loop in python but use `torch.addcmul` etc? No.
        
        pass # Placeholder description, real code below
        
    return spikes

# Actually, simplest speedup without JIT-ing the custom autograd is:
# Pre-bind methods.
# But to really answer the user's request for "fine-tuning calculation",
# we can rewrite the loop to avoid python overhead using `torch.unbind` and `torch.stack`.

def lif_step_optimized(x, v, decay, v_threshold, surrogate_alpha, detach_reset):
    # Unbind time dimension to avoid indexing overhead in loop
    inputs = x.unbind(1)
    spikes = []
    
    for input_t in inputs:
        v = v * decay + input_t
        
        spike = sigmoid_surrogate(v - v_threshold, surrogate_alpha)
        
        v_reset_term = spike * v_threshold
        if detach_reset:
            v_reset_term = v_reset_term.detach()
        
        v = v - v_reset_term
        spikes.append(spike)
        
    return torch.stack(spikes, dim=1), v

class LIFNode(nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_alpha=4.0, detach_reset=True):
        """
        LIF Neuron with Soft Reset.
        
        Args:
            tau (float): Membrane time constant. Decay factor will be (1 - 1/tau).
            v_threshold (float): Firing threshold.
            v_reset (float): Resting potential (usually 0).
            surrogate_alpha (float): Alpha parameter for sigmoid surrogate gradient.
            detach_reset (bool): Whether to detach the reset term from the computation graph (often helps stability).
        """
        super().__init__()
        self.tau = tau
        self.decay = 1.0 - (1.0 / tau)
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_alpha = surrogate_alpha
        self.detach_reset = detach_reset
        
        # State
        self.v = 0.0
        
        # Monitor hooks can access this
        self.last_v = None # Tensor to store V after update but before reset (for monitoring distribution)
        self.fire_rate = 0.0 # Moving average or batch average firing rate
        self.monitor_mode = False
        self.v_seq = None # Store (N, T, ...) v if monitor_mode is True
        
    def reset(self):
        self.v = 0.0
        self.last_v = None
        self.v_seq = None
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input current. Shape (N, T, ...) or (N, ...). 
                              If (N, T, ...), we iterate over T.
                              
        Returns:
            spike (torch.Tensor): Output spikes. Same shape as x.
        """
        # Handle time dimension if present
        if x.dim() > 2: 
            # Initialize v if it's scalar or shape mismatch (e.g. last batch size diff)
            if isinstance(self.v, float) or self.v.shape != x[:, 0, ...].shape:
                self.v = torch.zeros_like(x[:, 0, ...])
            
            # Optimization: Use unbind to iterate faster than indexing
            # And skip monitoring logic if not needed
            if not self.monitor_mode:
                out_spikes, self.v = lif_step_optimized(x, self.v, self.decay, self.v_threshold, self.surrogate_alpha, self.detach_reset)
                return out_spikes

            # Slow path with monitoring
            time_steps = x.shape[1]
            spikes = []
            v_list = []
            
            for t in range(time_steps):
                input_t = x[:, t, ...]
                
                self.v = self.v * self.decay + input_t
                
                v_list.append(self.v.detach().cpu())
                
                spike = sigmoid_surrogate(self.v - self.v_threshold, self.surrogate_alpha)
                
                v_reset_term = spike * self.v_threshold
                if self.detach_reset:
                    v_reset_term = v_reset_term.detach()
                
                self.v = self.v - v_reset_term
                spikes.append(spike)
            
            self.v_seq = torch.stack(v_list, dim=1)
            return torch.stack(spikes, dim=1)
        
        else:
            raise ValueError("Input must be (N, T, ...). Got shape: " + str(x.shape))

    def __repr__(self):
        return f"LIFNode(tau={self.tau}, v_th={self.v_threshold}, soft_reset=True)"
