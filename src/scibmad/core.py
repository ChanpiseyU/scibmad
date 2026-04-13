import sys
import types
import subprocess


def _ensure_juliacall():
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

def _ensure_scibmad():
    try:
        jl.seval("using SciBmad")
    except Exception:
        print("[scibmad] SciBmad not found — installing Julia package...")
        jl.seval('import Pkg; Pkg.add("SciBmad")')
        jl.seval("using SciBmad")
        print("[scibmad] SciBmad installed successfully.")

_ensure_scibmad()

import numpy as np
import torch


def _to_julia_input(x):
    x_np = x.detach().cpu().numpy()
    if x_np.size == 1:
        return float(x_np.item()), True
    return jl.collect(x_np.flatten()), False


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

    def __init__(self, fn: juliacall.AnyValue):
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


class ElementSpec:
    __slots__ = ("kind", "params")

    def __init__(self, kind, *params):
        self.kind = kind
        self.params = params

    def __repr__(self):
        return f"ElementSpec(kind={self.kind!r}, params={self.params!r})"


class BeamlineSpec:
    __slots__ = ("elements",)

    def __init__(self, elements):
        self.elements = list(elements)

    def track(self, coords):
        for ele in self.elements:
            fn = getattr(_proxy, ele.kind)
            coords = fn(coords, *ele.params)
        return coords

    def __call__(self, coords):
        return self.track(coords)

    def __repr__(self):
        return f"BeamlineSpec(elements={self.elements!r})"


class _JuliaProxy(types.ModuleType):

    def __init__(self, real_module):
        super().__init__(__name__)
        self.__dict__.update({
            k: v for k, v in real_module.__dict__.items()
            if k.startswith("__") and k.endswith("__")
        })
        self.__dict__["_jl"] = jl
        self.__dict__["_juliacall"] = juliacall

        jl.seval("""
        using DifferentiationInterface
        import ForwardDiff
        using DifferentiationInterface: gradient, jacobian, derivative
        const backend = AutoForwardDiff()

        function scibmad_call_with_arg(f, args, arg_index, x)
            return f((i == arg_index ? x : args[i] for i in eachindex(args))...)
        end

        function scibmad_derivative_arg(f, backend, args, arg_index, x)
            return derivative(y -> scibmad_call_with_arg(f, args, arg_index, y), backend, x)
        end

        function scibmad_gradient_arg(f, backend, args, arg_index, x)
            return gradient(y -> scibmad_call_with_arg(f, args, arg_index, y), backend, x)
        end

        function scibmad_jacobian_arg(f, backend, args, arg_index, x)
            return jacobian(y -> scibmad_call_with_arg(f, args, arg_index, y), backend, x)
        end

        function scibmad_jacobian_scalar_arg(f, backend, args, arg_index, x)
            return jacobian(y -> scibmad_call_with_arg(f, args, arg_index, y[1]), backend, [x])
        end
        """)

        jl.seval("""
        using SciBmad

        function _scibmad_track_ele(coords, ele::LineElement;
                                    species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(R_ref))
            coordsT = T.(coords)

            v = Matrix{T}(undef, 1, 6)
            v[1, :] .= coordsT

            bunch = Bunch(v; species=species, R_ref=T(R_ref))
            track!(bunch, ele)
            return vec(bunch.coords.v[1, :])
        end

        function quadrupole(coords, L::Real, Kn1::Real;
                            species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(Kn1), typeof(R_ref))
            ele = Quadrupole(L=T(L), Kn1=T(Kn1))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function drift(coords, L::Real;
                       species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(R_ref))
            ele = Drift(L=T(L))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function sbend(coords, L::Real, angle::Real;
                       species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(angle), typeof(R_ref))
            ele = SBend(L=T(L), angle=T(angle))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function sextupole(coords, L::Real, Kn2::Real;
                           species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(Kn2), typeof(R_ref))
            ele = Sextupole(L=T(L), Kn2=T(Kn2))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function octupole(coords, L::Real, Kn3::Real;
                          species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(Kn3), typeof(R_ref))
            ele = Octupole(L=T(L), Kn3=T(Kn3))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function solenoid(coords, L::Real, Ksol::Real;
                          species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(Ksol), typeof(R_ref))
            ele = Solenoid(L=T(L), Ksol=T(Ksol))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function hkicker(coords, L::Real, Kn0::Real;
                         species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(Kn0), typeof(R_ref))
            ele = HKicker(L=T(L), Kn0=T(Kn0))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function vkicker(coords, L::Real, Ks0::Real;
                         species=Species("electron"), R_ref=1.0)
            T = promote_type(eltype(coords), typeof(L), typeof(Ks0), typeof(R_ref))
            ele = VKicker(L=T(L), Ks0=T(Ks0))
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end

        function rfcavity(coords, L::Real, voltage::Real, frequency::Real, phase::Real;
                          species=Species("electron"), R_ref=1.0)
            T = promote_type(
                eltype(coords),
                typeof(L),
                typeof(voltage),
                typeof(frequency),
                typeof(phase),
                typeof(R_ref),
            )
            ele = RFCavity(
                L=T(L),
                voltage=T(voltage),
                frequency=T(frequency),
                phase=T(phase),
            )
            return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
        end
        """)

    def beamline(self, *elements):
        return BeamlineSpec(elements)

    def quadrupole_ele(self, L, Kn1):
        return ElementSpec("quadrupole", L, Kn1)

    def drift_ele(self, L):
        return ElementSpec("drift", L)

    def sbend_ele(self, L, angle):
        return ElementSpec("sbend", L, angle)

    def sextupole_ele(self, L, Kn2):
        return ElementSpec("sextupole", L, Kn2)

    def octupole_ele(self, L, Kn3):
        return ElementSpec("octupole", L, Kn3)

    def solenoid_ele(self, L, Ksol):
        return ElementSpec("solenoid", L, Ksol)

    def hkicker_ele(self, L, Kn0):
        return ElementSpec("hkicker", L, Kn0)

    def vkicker_ele(self, L, Ks0):
        return ElementSpec("vkicker", L, Ks0)

    def rfcavity_ele(self, L, voltage, frequency, phase):
        return ElementSpec("rfcavity", L, voltage, frequency, phase)

    def seval(self, code: str):
        return jl.seval(code)

    def using(self, *packages: str):
        for package in packages:
            jl.seval(f"using {package}")

    def load(self, path: str):
        jl.seval(f'include("{path}")')

    def define(self, code: str):
        jl.seval(code)

    def __getattr__(self, name):
        _jl = self.__dict__["_jl"]
        _juliacall = self.__dict__["_juliacall"]

        try:
            attr = getattr(_jl, name)
        except AttributeError:
            raise AttributeError(
                f"module 'scibmad' has no attribute '{name}' "
                f"(not found in Julia Main namespace either; "
                f"did you forget scibmad.load() or scibmad.define()?)"
            )

        if isinstance(attr, _juliacall.AnyValue) and callable(attr):
            return _JuliaTorchCallable(attr)

        return attr


_proxy = _JuliaProxy(sys.modules[__name__])
sys.modules[__name__] = _proxy
