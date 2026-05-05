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
        try:
            jl.seval('import Pkg; Pkg.add("SciBmad")')
            jl.seval("using SciBmad")
            print("[scibmad] SciBmad installed successfully.")
        except Exception as exc:
            raise RuntimeError(
                "SciBmad could not be loaded or installed automatically."
            ) from exc

_ensure_scibmad()

from .autodiff import AUTODIFF_JULIA_CODE
from .elements import (
    ELEMENT_JULIA_CODE,
    ElementSpec,
    BeamlineSpec,
    ObjectElement,
    ObjectBeamline,
    TrackingResult,
    OBJECT_KIND_TO_LOWER,
)
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

    def Beamline(self, elements, **kwargs):
        return ObjectBeamline(elements, **kwargs)

    def _call_julia(self, name, *args, **kwargs):
        attr = getattr(self.__dict__["_jl"], name)
        if isinstance(attr, self.__dict__["_juliacall"].AnyValue) and callable(attr):
            attr = _JuliaTorchCallable(attr)
        return attr(*args, **kwargs)

    def _build_or_call(self, kind, build_arg_counts, *args, **kwargs):
        if len(args) in build_arg_counts and set(kwargs).issubset({"p_over_q_ref"}):
            if "p_over_q_ref" in kwargs:
                return ElementSpec(kind, *args, kwargs["p_over_q_ref"])
            return ElementSpec(kind, *args)
        return self._call_julia(kind, *args, **kwargs)

    def quadrupole(self, *args, **kwargs):
        return self._build_or_call("quadrupole", (2, 3), *args, **kwargs)

    def Quadrupole(self, **kwargs):
        return ObjectElement("Quadrupole", **kwargs)

    def drift(self, *args, **kwargs):
        return self._build_or_call("drift", (1, 2), *args, **kwargs)

    def Drift(self, **kwargs):
        return ObjectElement("Drift", **kwargs)

    def sbend(self, *args, **kwargs):
        return self._build_or_call("sbend", (2, 3), *args, **kwargs)

    def SBend(self, **kwargs):
        return ObjectElement("SBend", **kwargs)

    def sextupole(self, *args, **kwargs):
        return self._build_or_call("sextupole", (2, 3), *args, **kwargs)

    def Sextupole(self, **kwargs):
        return ObjectElement("Sextupole", **kwargs)

    def octupole(self, *args, **kwargs):
        return self._build_or_call("octupole", (2, 3), *args, **kwargs)

    def Octupole(self, **kwargs):
        return ObjectElement("Octupole", **kwargs)

    def solenoid(self, *args, **kwargs):
        return self._build_or_call("solenoid", (2, 3), *args, **kwargs)

    def Solenoid(self, **kwargs):
        return ObjectElement("Solenoid", **kwargs)

    def hkicker(self, *args, **kwargs):
        return self._build_or_call("hkicker", (2, 3), *args, **kwargs)

    def HKicker(self, **kwargs):
        return ObjectElement("HKicker", **kwargs)

    def vkicker(self, *args, **kwargs):
        return self._build_or_call("vkicker", (2, 3), *args, **kwargs)

    def VKicker(self, **kwargs):
        return ObjectElement("VKicker", **kwargs)

    def rfcavity(self, *args, **kwargs):
        return self._build_or_call("rfcavity", (4, 5), *args, **kwargs)

    def RFCavity(self, **kwargs):
        return ObjectElement("RFCavity", **kwargs)

    def E_to_R(self, species, energy):
        return jl.seval("Beamlines.E_to_R")(species, energy)

    def track(self, beamline, *, v0, n_turns=1, save_every_n_turns=1):
        if not isinstance(beamline, ObjectBeamline):
            return self._call_julia("track", beamline, v0=v0, n_turns=n_turns, save_every_n_turns=save_every_n_turns)

        import torch

        coords = v0
        if not isinstance(coords, torch.Tensor):
            raise TypeError("track(..., v0=...) expects a torch.Tensor for ObjectBeamline")

        if coords.ndim == 2:
            if coords.shape[0] != 1:
                raise ValueError("ObjectBeamline currently supports a single particle only")
            current = coords[0]
        elif coords.ndim == 1:
            current = coords
        else:
            raise ValueError("v0 must be a length-6 vector or shape (1, 6) tensor")

        p_over_q_ref = beamline.resolve_p_over_q_ref(self)
        snapshots = [current]

        for turn in range(1, n_turns + 1):
            for ele in beamline.elements:
                fn = getattr(self, OBJECT_KIND_TO_LOWER[ele.kind])
                current = fn(
                    current,
                    *ele.ordered_params(),
                    p_over_q_ref,
                    beamline.species_ref if beamline.species_ref is not None else self.Species("electron"),
                )
            if turn % save_every_n_turns == 0:
                snapshots.append(current)

        v = torch.stack(snapshots, dim=-1).unsqueeze(0)
        return TrackingResult(v)

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
