import numpy as np
import torch
from juliacall import Main as jl


def _to_julia_input(x):
    x_np = x.detach().cpu().numpy()
    if x_np.size == 1:
        return float(x_np.item()), True
    return x_np.flatten(), False


def _to_numpy_output(result):
    result_np = np.array(result)
    if result_np.ndim == 0:
        return float(result_np), False
    return result_np, True


class _JuliaFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, julia_func, tensor_arg_indices, *args):
        tensor_arg_indices = tuple(tensor_arg_indices)
        tensor_args = tuple(args[index] for index in tensor_arg_indices)

        jl_args = list(args)
        tensor_metadata = {}

        for index, x in zip(tensor_arg_indices, tensor_args):
            jl_x, is_scalar_input = _to_julia_input(x)
            jl_args[index] = jl_x
            tensor_metadata[index] = {
                "jl_x": jl_x,
                "is_scalar_input": is_scalar_input,
            }

        result = julia_func(*jl_args)
        result_np, is_vector_output = _to_numpy_output(result)

        ctx.is_vector_output = is_vector_output
        ctx.julia_func = julia_func
        ctx.jl_args = tuple(jl_args)
        ctx.tensor_arg_indices = tensor_arg_indices
        ctx.tensor_metadata = tensor_metadata
        ctx.num_args = len(args)
        ctx.save_for_backward(*tensor_args)

        return torch.tensor(
            result_np,
            dtype=tensor_args[0].dtype,
            device=tensor_args[0].device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        tensor_args = ctx.saved_tensors
        backend = jl.backend
        grad_output_np = grad_output.detach().cpu().numpy()
        grad_output_flat = grad_output_np.flatten()

        def grad_for_tensor_arg(tensor_arg_index, x):
            metadata = ctx.tensor_metadata[tensor_arg_index]
            jl_x = metadata["jl_x"]
            is_scalar_input = metadata["is_scalar_input"]

            if is_scalar_input:
                if ctx.is_vector_output:
                    jac_np = np.array(
                        jl.scibmad_jacobian_scalar_arg(
                            ctx.julia_func,
                            backend,
                            ctx.jl_args,
                            tensor_arg_index + 1,
                            jl_x,
                        )
                    ).reshape(-1)
                    grad = np.array([jac_np @ grad_output_flat])
                else:
                    grad = np.array([
                        float(
                            jl.scibmad_derivative_arg(
                                ctx.julia_func,
                                backend,
                                ctx.jl_args,
                                tensor_arg_index + 1,
                                jl_x,
                            )
                        )
                    ])
            else:
                if ctx.is_vector_output:
                    jac_np = np.array(
                        jl.scibmad_jacobian_arg(
                            ctx.julia_func,
                            backend,
                            ctx.jl_args,
                            tensor_arg_index + 1,
                            jl_x,
                        )
                    )
                    grad = jac_np.T @ grad_output_flat
                else:
                    grad = np.array(
                        jl.scibmad_gradient_arg(
                            ctx.julia_func,
                            backend,
                            ctx.jl_args,
                            tensor_arg_index + 1,
                            jl_x,
                        )
                    )

            if not ctx.is_vector_output:
                grad = grad * grad_output_np

            return torch.tensor(grad, dtype=x.dtype, device=x.device).reshape(x.shape)

        arg_grads = [None] * ctx.num_args

        for tensor_arg_index, x in zip(ctx.tensor_arg_indices, tensor_args):
            arg_grads[tensor_arg_index] = grad_for_tensor_arg(tensor_arg_index, x)

        return None, None, *arg_grads


class _JuliaTorchCallable:
    __slots__ = ("_fn",)

    def __init__(self, fn):
        object.__setattr__(self, "_fn", fn)

    def __call__(self, *args, **kwargs):
        fn = object.__getattribute__(self, "_fn")
        tensor_arg_indices = [
            index for index, arg in enumerate(args)
            if isinstance(arg, torch.Tensor)
        ]

        if not kwargs and tensor_arg_indices:
            return _JuliaFunction.apply(fn, tuple(tensor_arg_indices), *args)

        return fn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_fn"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_fn"), name, value)

    def __repr__(self):
        return repr(object.__getattribute__(self, "_fn"))