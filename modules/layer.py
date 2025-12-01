import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    """
    Wraps a stateless layer (like Conv2d, BatchNorm) to process (N, T, C, ...) inputs.
    It flattens N and T into a single batch dimension, applies the layer, and reshapes back.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        # x shape: (N, T, C, H, W) or (N, T, Features)
        if x.dim() < 3:
            raise ValueError(f"TimeDistributed expects at least 3 dims (N, T, ...), got {x.shape}")
            
        N, T = x.shape[0], x.shape[1]
        
        # Merge N and T -> (N*T, ...)
        x_reshaped = x.flatten(0, 1) 
        
        # Apply module
        y_reshaped = self.module(x_reshaped)
        
        # Reshape back: (N*T, ...) -> (N, T, ...)
        # Infer the output shape from the layer's output
        trailing_dims = y_reshaped.shape[1:]
        y = y_reshaped.view(N, T, *trailing_dims)
        
        return y
