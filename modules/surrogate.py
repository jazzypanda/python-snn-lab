import torch
import torch.nn as nn

class SigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha=4.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        
        sigmoid_x = torch.sigmoid(alpha * input)
        grad_input = grad_output * alpha * sigmoid_x * (1 - sigmoid_x)
        return grad_input, None

def sigmoid_surrogate(x, alpha=4.0):
    return SigmoidSurrogate.apply(x, alpha)
