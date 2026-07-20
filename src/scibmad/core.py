import sys
import types
import subprocess

def _ensure_juliacall():
    """
    Checks whether the juliacall package is importable, and if not, installs it via pip before continuing
    Runs once, automatically, the first time scibmad is imported, the user never has to run this by hand

    Arguments:
    - (none)

    Returns:
    - (none)
    """
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
    """
    Checks whether the Julia SciBmad package can be loaded via `using SciBmad`, and if that fails for any reason, 
    installs it via Julia's Pkg.add and retries. Runs once, automatically, the first time scibmad is imported

    Arguments:
    - (none)

    Returns:
    - (none)

    Raises:
    - RuntimeError: if SciBmad still can't be loaded after attempting to install it
    """
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


class _JuliaProxy(types.ModuleType): # Obj. that scibmad becomes after import, subclass of types.ModuleType that gets swapped into sys.modules in place of plain module

    def __init__(self, real_module):
        """Construct the proxy at import time: copy dunder attrs, store handles to the Julia session and juliacall, 
        and eval the autodiff/element Julia source so everything is ready immediately

        Arguments:
        - real_module: the original (plain) scibmad module object, used only to copy over dunder attributes like __name__ and __file__

        Returns:
        - (none)
        """
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
        """
        Builds the lazy/functional-style beamline (BeamlineSpec) out of the given elements (No physics computation happens here, just packaging)

        Arguments:
        - *elements: ElementSpec instances (e.g. produced by scibmad.drift(0.5)), in beamline order

        Returns:
        - beamline: a BeamlineSpec wrapping the given elements
        """
        return BeamlineSpec(elements)

    def Beamline(self, elements, **kwargs):
        """
        Builds the structured/object-style beamline (ObjectBeamline) out of the given elements and reference beam parameters (No physics computation happens here, just packaging)

        Arguments:
        - elements: an iterable of ObjectElement instances, in beamline order
        - **kwargs: reference beam parameters — species_ref, p_over_q_ref, and/or E_ref

        Returns:
        - beamline: an ObjectBeamline wrapping the given elements and parameters
        """
        return ObjectBeamline(elements, **kwargs)

    def _call_julia(self, name, *args, **kwargs):
        """
        Low-level dispatcher that every physics call goes through, looks up name on the live Julia session,
        wraps it in _JuliaTorchCallable if it's a callable Julia value (so tensor arguments get autograd support), and calls it

        Arguments:
        - name: the name of the Julia function/value to call, as it exists in Julia's Main namespace
        - *args, **kwargs: arguments to forward to the Julia call

        Returns:
        - result: whatever the Julia call returns, autograd-aware if any argument was a tensor
        """
        attr = getattr(self.__dict__["_jl"], name)
        if isinstance(attr, self.__dict__["_juliacall"].AnyValue) and callable(attr):
            attr = _JuliaTorchCallable(attr)
        return attr(*args, **kwargs)

    def _build_or_call(self, kind, build_arg_counts, *args, **kwargs):
        """
        Overload-resolution trick behind every lowercase element method (quadrupole, drift, etc)

        If the positional arg count matches build_arg_counts and the only kwarg (if any) is p_over_q_ref, builds a lazy ElementSpec
        Otherwise runs the element directly via _call_julia.

        Arguments:
        - kind: the element type name, e.g. "drift"
        - build_arg_counts: the positional argument counts that indicate a "build" call rather than a "run" call, e.g. (1, 2) for drift
        - *args, **kwargs: the arguments the caller passed in

        Returns:
        - result: either an ElementSpec (build case) or the result of the actual Julia tracking call (run case)
        """
        if len(args) in build_arg_counts and set(kwargs).issubset({"p_over_q_ref"}):
            if "p_over_q_ref" in kwargs:
                return ElementSpec(kind, *args, kwargs["p_over_q_ref"])
            return ElementSpec(kind, *args)
        return self._call_julia(kind, *args, **kwargs)

    # THE FOLLOWING DOC STRINGS APPLY FOR REST OF BEAMLINE ELEMENTS

    def quadrupole(self, *args, **kwargs):
        """
        [element](self, *args, **kwargs)   ex. quadrupole, drift, sbend, sextupole, octupole, solenoid, hkicker, vkicker, rfcavity

        Lowercase element methods. Each is a one-line call into _build_or_call with that element's kind name and its valid "build" argument-count tuple 
        (ex drift: (1, 2), quadrupole: (2, 3), rfcavity: (4, 5)), so calling with a small number of plain arguments builds a lazy ElementSpec, while calling 
        with real coordinates runs the element for real

        Arguments:
        - *args, **kwargs: either build-time parameters (length, strengths, optionally p_over_q_ref) or run-time arguments (coordinates plus parameters), depending on call shape

        Returns:
        - result: either an ElementSpec (build case) or tracked coordinates (run case)
        """
        return self._build_or_call("quadrupole", (2, 3), *args, **kwargs)

    def Quadrupole(self, **kwargs):
        """
        [Element](self, **kwargs)   ex. Quadrupole, Drift, SBend, Sextupole, Octupole, Solenoid, HKicker, VKicker, RFCavity

        Capitalized element methods. Each is a one-line call constructing an ObjectElement of that kind with the given named parameters,
        always the structured, validated style, never the lazy build/run overload trick used by the lowercase versions

        Arguments:
        - **kwargs: the element's named parameters, e.g. L=0.2, Kn1=1.3 for Quadrupole

        Returns:
        - element: an ObjectElement of the corresponding kind
        """
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
        """
        Converts a reference energy into a reference momentum-over-charge value, by calling Beamlines.E_to_R in Julia directly

        Arguments:
        - species: the reference particle species
        - energy: the reference energy to convert

        Returns:
        - p_over_q_ref: the corresponding reference momentum-over-charge value
        """
        return jl.seval("Beamlines.E_to_R")(species, energy)

    def track(self, beamline, *, v0, n_turns=1, save_every_n_turns=1):
        """
        Run a tracking simulation over one or more turns. Lazy BeamlineSpec forwards straight to Julia's track function

        Structured ObjectBeamline loops in Python: resolves p_over_q_ref once, then pushes the particle through each element per turn (autograd-aware),
        saving a snapshot everysave_every_n_turns turns

        Arguments:
        - beamline: either a BeamlineSpec or an ObjectBeamline to track through
        - v0: the starting particle coordinates, as a torch.Tensor of shape (6,) or (1, 6)
        - n_turns: the number of turns (laps) to track for
        - save_every_n_turns: how often, in turns, to save a coordinate snapshot

        Returns:
        - result: for a BeamlineSpec, whatever Julia's track returns directly; for an ObjectBeamline, a TrackingResult wrapping the stacked snapshot tensor

        Raises:
        - TypeError: if v0 is not a torch.Tensor (ObjectBeamline path)
        - ValueError: if v0 has an unsupported shape, or if more than one particle is given (ObjectBeamline path)
        """
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
        """
        Evaluates a raw string of Julia source code directly in the live Julia session

        Arguments:
        - code: the Julia source code to evaluate

        Returns:
        - result: whatever the evaluated Julia code returns
        """
        return jl.seval(code)

    def using(self, *packages: str):
        """
        Loads one or more Julia packages into the live session by evaluating `using <package>` for each

        Arguments:
        - *packages: the names of the Julia packages to load

        Returns:
        - (none)
        """
        for package in packages:
            jl.seval(f"using {package}")

    def load(self, path: str):
        """
        Includes a Julia source file into the live session, evaluating `include("<path>")`

        Arguments:
        - path: the filesystem path to the Julia source file to include

        Returns:
        - (none)
        """
        jl.seval(f'include("{path}")')

    def define(self, code: str):
        """
        Evaluates a raw string of Julia source code directly in the live Julia session, 
        functionally identical to seval, kept as a separately named alias for defining new Julia code inline

        Arguments:
        code: the Julia source code to evaluate/define

        Returns:
        - (none)
        """
        jl.seval(code)

    def __getattr__(self, name):
        """
        Fallback for attributes not defined above, looks name up live in Julia's Main namespace
        if it's callable, wraps it in _JuliaTorchCallable for autograd support. Lets any Julia-defined name work from Python with no wrapper needed

        Arguments:
        - name: the attribute name being looked up

        Returns:
        - value: the corresponding Julia value, wrapped in _JuliaTorchCallable if it's callable

        Raises:
        - AttributeError: if name isn't found in Julia's Main namespace either, with a message suggesting scibmad.load() or scibmad.define()
        """
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
sys.modules[__name__] = _proxy # Replace this module with the proxy so all future access routes through _JuliaProxy
