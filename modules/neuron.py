import torch
import torch.nn as nn
from .surrogate import sigmoid_surrogate

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
        # We assume input x is (N, T, C, H, W) or (N, T, Features) if dim > 2 and 2nd dim is Time.
        # Or simply check if we want to process step-by-step or all-at-once.
        # For this architecture, we assume the Layer receives the WHOLE sequence (N, T, ...)
        
        if x.dim() > 2: 
            # Heuristic: We assume the standard input format (N, T, C, H, W)
            # We need to iterate over the time dimension (dim=1)
            time_steps = x.shape[1]
            spikes = []
            v_list = [] if self.monitor_mode else None
            
            # Initialize v if it's scalar/reset
            if isinstance(self.v, float):
                # Create a tensor of zeros matching (N, C, H, W) -> x[:, 0, ...]
                self.v = torch.zeros_like(x[:, 0, ...])
            
            for t in range(time_steps):
                input_t = x[:, t, ...]
                
                # LIF Dynamics
                # V[t] = V[t-1] * decay + Input[t]
                self.v = self.v * self.decay + input_t
                
                if self.monitor_mode:
                    v_list.append(self.v.detach().cpu()) # Save cpu copy to save gpu mem
                
                # Store for monitoring (before reset)
                # We might only store the last one or statistics to save memory, 
                # but hooks can capture self.v here if they register a forward_hook.
                # However, hooks on the module usually capture the RETURN value.
                # We will store 'v' implicitly in the state for the hook to read.
                
                # Spike Generation
                # S[t] = H(V[t] - Vth)
                spike = sigmoid_surrogate(self.v - self.v_threshold, self.surrogate_alpha)
                
                # Soft Reset
                # V[t] = V[t] - S[t] * Vth
                v_reset_term = spike * self.v_threshold
                if self.detach_reset:
                    v_reset_term = v_reset_term.detach()
                
                self.v = self.v - v_reset_term
                
                spikes.append(spike)
            
            if self.monitor_mode:
                self.v_seq = torch.stack(v_list, dim=1)

            # Stack back to (N, T, ...)
            return torch.stack(spikes, dim=1)
        
        else:
            # Fallback for single step or flattened input, but we enforced T dimension in design.
            # Raise error to strictly follow design? Or handle single step?
            # Let's assume (N, T, ...) is strictly required for this project.
            raise ValueError("Input must be (N, T, ...). Got shape: " + str(x.shape))

    def __repr__(self):
        return f"LIFNode(tau={self.tau}, v_th={self.v_threshold}, soft_reset=True)"
