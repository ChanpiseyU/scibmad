import sys
import types
import subprocess
import importlib

def _ensure_juliacall():
    # Auto-install juliacall if it's not already available
    try:
        import juliacall 
    except ImportError:
        print("[scibmad] juliacall not found — installing via pip...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "juliacall"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[scibmad] juliacall installed successfully.")

_ensure_juliacall()

import juliacall  
from juliacall import Main as jl  

import numpy as np   
import torch         

class _JuliaFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, julia_func):
        x_np = x.detach().cpu().numpy()

        if x_np.size == 1:
            jl_x = float(x_np.item())
            ctx.is_scalar = True
        else:
            jl_x = jl.collect(x_np.flatten())
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

        return torch.tensor(result_np, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        backend = jl.backend

        if ctx.is_scalar:
            grad = np.array([float(jl.derivative(ctx.julia_func, backend, ctx.jl_x))])
        else:
            if ctx.is_vector_output:
                jac_np = np.array(jl.jacobian(ctx.julia_func, backend, ctx.jl_x))
                grad = jac_np.T @ grad_output.detach().cpu().numpy().flatten()
            else:
                grad = np.array(jl.gradient(ctx.julia_func, backend, ctx.jl_x))

        if not ctx.is_vector_output:
            grad = grad * grad_output.detach().cpu().numpy()

        return torch.tensor(grad, dtype=x.dtype, device=x.device).reshape(x.shape), None


class _JuliaTorchCallable:
    """Wraps a Julia callable; tensor arguments automatically flow through autograd."""

    __slots__ = ('_fn',)

    def __init__(self, fn: juliacall.AnyValue):
        object.__setattr__(self, '_fn', fn)

    def __call__(self, *args, **kwargs):
        fn = object.__getattribute__(self, '_fn')
        if len(args) == 1 and not kwargs and isinstance(args[0], torch.Tensor):
            return _JuliaFunction.apply(args[0], fn)
        return fn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_fn'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_fn'), name, value)

    def __repr__(self):
        return repr(object.__getattribute__(self, '_fn'))


class _JuliaProxy(types.ModuleType):
    """
    Replaces the scibmad module in sys.modules.
    Attribute access delegates to juliacall.Main; any callable
    coming back is wrapped in _JuliaTorchCallable automatically.
    Users never need to import juliacall directly.
    """

    def __init__(self, real_module):
        super().__init__(__name__)
        self.__dict__.update({
            k: v for k, v in real_module.__dict__.items()
            if k.startswith('__') and k.endswith('__')
        })
        self.__dict__['_jl'] = jl
        self.__dict__['_juliacall'] = juliacall

        jl.seval("""
        using DifferentiationInterface
        import ForwardDiff
        const backend = AutoForwardDiff()
        """)

    def __getattr__(self, name):
        _jl = self.__dict__['_jl']
        _juliacall = self.__dict__['_juliacall']
        attr = getattr(_jl, name)
        if isinstance(attr, _juliacall.AnyValue) and callable(attr):
            return _JuliaTorchCallable(attr)
        return attr


newpatch = _JuliaProxy(sys.modules[__name__])
sys.modules[__name__] = newpatch