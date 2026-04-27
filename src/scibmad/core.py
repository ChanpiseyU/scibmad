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

from .autodiff import AUTODIFF_JULIA_CODE
from .elements import ELEMENT_JULIA_CODE, ElementSpec, BeamlineSpec
from .bridge import _to_julia_input, _to_numpy_output, _JuliaFunction, _JuliaTorchCallable


class _JuliaProxy(types.ModuleType):

    def __init__(self, real_module):
        super().__init__(__name__)
        self.__dict__.update({
            k: v for k, v in real_module.__dict__.items()
            if k.startswith("__") and k.endswith("__")
        })
        self.__dict__["_jl"] = jl
        self.__dict__["_juliacall"] = juliacall

        jl.seval(AUTODIFF_JULIA_CODE)
        jl.seval(ELEMENT_JULIA_CODE)

    def beamline(self, *elements):
        return BeamlineSpec(elements)

    def quadrupole_ele(self, L, Kn1, R_ref=1.0):
        return ElementSpec("quadrupole", L, Kn1, R_ref)

    def drift_ele(self, L, R_ref=1.0):
        return ElementSpec("drift", L, R_ref)

    def sbend_ele(self, L, angle, R_ref=1.0):
        return ElementSpec("sbend", L, angle, R_ref)

    def sextupole_ele(self, L, Kn2, R_ref=1.0):
        return ElementSpec("sextupole", L, Kn2, R_ref)

    def octupole_ele(self, L, Kn3, R_ref=1.0):
        return ElementSpec("octupole", L, Kn3, R_ref)

    def solenoid_ele(self, L, Ksol, R_ref=1.0):
        return ElementSpec("solenoid", L, Ksol, R_ref)

    def hkicker_ele(self, L, Kn0, R_ref=1.0):
        return ElementSpec("hkicker", L, Kn0, R_ref)

    def vkicker_ele(self, L, Ks0, R_ref=1.0):
        return ElementSpec("vkicker", L, Ks0, R_ref)

    def rfcavity_ele(self, L, voltage, frequency, phase, R_ref=1.0):
        return ElementSpec("rfcavity", L, voltage, frequency, phase, R_ref)

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