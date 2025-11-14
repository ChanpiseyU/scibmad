from juliacall import Main as jl
from juliacall import AnyValue
import numpy as np
import torch

jl.seval("""
using DifferentiationInterface
import ForwardDiff

const backend = AutoForwardDiff()
""")

# Implementation
class JuliaFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, julia_func):
        x_np = x.detach().cpu().numpy()
        
        if x_np.size == 1:
            jl_x = float(x_np.item())
            ctx.is_scalar = True
        else:
            jl_x = jl.Vector[jl.Float64](x_np.flatten())
            ctx.is_scalar = False
        
        result = julia_func(jl_x)
        
        if hasattr(result, '__len__'):
            result_np = np.array(result)
            ctx.is_vector_output = True
        else:
            result_np = float(result)
            ctx.is_vector_output = False
        
        ctx.jl_x = jl_x
        ctx.julia_func = julia_func
        ctx.save_for_backward(x)
        
        if ctx.is_vector_output:
            return torch.tensor(result_np, dtype=x.dtype, device=x.device)
        else:
            return torch.tensor(result_np, dtype=x.dtype, device=x.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        if ctx.is_scalar:
            grad = jl.derivative(ctx.julia_func, jl.backend, ctx.jl_x)
            grad = np.array([float(grad)])
        else:
            if ctx.is_vector_output:
                jac = jl.jacobian(ctx.julia_func, jl.backend, ctx.jl_x)
                jac_np = np.array(jac)
                grad_output_np = grad_output.detach().cpu().numpy()
                grad = grad_output_np @ jac_np
            else:
                grad = jl.gradient(ctx.julia_func, jl.backend, ctx.jl_x)
                grad = np.array(grad)
        
        if not ctx.is_vector_output:
            grad_output_np = grad_output.detach().cpu().numpy()
            grad = grad * grad_output_np
        
        return torch.tensor(grad, dtype=x.dtype, device=x.device).reshape(x.shape), None

# Create Monkey Patch
_original_call = AnyValue.__call__

def _julia_torch_call(self, *args, **kwargs):

    has_torch_tensor = any(isinstance(arg, torch.Tensor) for arg in args)
    
    if has_torch_tensor and len(args) == 1 and not kwargs:
        x = args[0]
        if isinstance(x, torch.Tensor):
            return JuliaFunction.apply(x, self)

    return _original_call(self, *args, **kwargs)

AnyValue.__call__ = _julia_torch_call
