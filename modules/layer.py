import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    """
    Wraps a stateless layer (like Conv2d, BatchNorm) to process (N, T, C, ...) inputs.
    It flattens N and T, applies the layer, and reshapes back.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        # x shape: (N, T, C, H, W) or (N, T, Features)
        if x.dim() < 3:
            raise ValueError(f"TimeDistributed expects at least 3 dims (N, T, ...), got {x.shape}")
            
        N, T = x.shape[0], x.shape[1]
        # Merge N and T
        x_reshaped = x.flatten(0, 1) 
        
        # Apply module
        y_reshaped = self.module(x_reshaped)
        
        # Reshape back: (N*T, ...) -> (N, T, ...)
        # We infer the output shape. 
        # y_reshaped shape is (N*T, C_out, H_out, W_out) or similar
        trailing_dims = y_reshaped.shape[1:]
        y = y_reshaped.view(N, T, *trailing_dims)
        
        return y

class DirectEncoder(nn.Module):
    """
    Repeats the static image T times to simulate constant current input.
    """
    def __init__(self, T):
        super().__init__()
        self.T = T
        
    def forward(self, x):
        # x: (N, C, H, W)
        # Out: (N, T, C, H, W)
        return x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
