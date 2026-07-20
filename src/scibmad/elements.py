
# APPLIES FOR FOLLOWING ELEMENT_JULIA_CODE CODE

"""
HAS FORM:
[element](coords, L, <element-specific parameters...>, [p_over_q_ref]; species=Species("electron"), [p_over_q_ref=1.0])

Shared shape for all nine element-tracking functions (quadrupole, drift,
sbend, sextupole, octupole, solenoid, hkicker, vkicker, rfcavity):
each builds one instance of its element type and hands off to
_scibmad_track_ele, which does the actual physics (one-element
beamline, single-particle bunch, run, return result).

Each element has three overloads (p_over_q_ref positional, keyword, or
positional + species) — all equivalent, just for calling convenience.

Arguments:
- coords: the starting particle coordinates
- L: the element's length
- <element-specific parameters>: the element's physics parameters, e.g. Kn1 (quadrupole), angle (sbend), Kn2 (sextupole), Kn3 (octupole), Ksol (solenoid), Kn0/Ks0 (h/vkicker), voltage/frequency/phase (rfcavity) — none for drift
- p_over_q_ref: the reference momentum-over-charge (positional or keyword depending on overload, defaults to 1.0)
- species: the reference particle species (default electron)

Returns:
- coords_out: the particle coordinates after passing through the element
"""

ELEMENT_JULIA_CODE = '''
using SciBmad

function _scibmad_track_ele(coords, ele::LineElement;
                            species=Species("electron"), p_over_q_ref=1.0)
    T = eltype(coords)
    coordsT = T.(coords)
    signed_p_over_q_ref = T(sign(chargeof(species)) * abs(p_over_q_ref))

    v = Matrix{T}(undef, 1, 6)
    v[1, :] .= coordsT

    bl = Beamline([ele]; species_ref=species, p_over_q_ref=signed_p_over_q_ref)
    bunch = Bunch(v; species=bl.species_ref, p_over_q_ref=bl.p_over_q_ref, t_ref=zero(T))
    track!(bunch, bl)
    return vec(bunch.coords.v[1, :])
end

function quadrupole(coords, L::Real, Kn1::Real, p_over_q_ref::Real;
                    species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn1))
    ele = Quadrupole(L=T(L), Kn1=T(Kn1))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function quadrupole(coords, L::Real, Kn1::Real;
                    species=Species("electron"), p_over_q_ref=1.0)
    return quadrupole(coords, L, Kn1, p_over_q_ref; species=species)
end

function quadrupole(coords, L::Real, Kn1::Real, p_over_q_ref::Real, species::Species)
    return quadrupole(coords, L, Kn1, p_over_q_ref; species=species)
end

function drift(coords, L::Real, p_over_q_ref::Real;
               species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L))
    ele = Drift(L=T(L))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function drift(coords, L::Real;
               species=Species("electron"), p_over_q_ref=1.0)
    return drift(coords, L, p_over_q_ref; species=species)
end

function drift(coords, L::Real, p_over_q_ref::Real, species::Species)
    return drift(coords, L, p_over_q_ref; species=species)
end

function sbend(coords, L::Real, angle::Real, p_over_q_ref::Real;
               species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(angle))
    ele = SBend(L=T(L), angle=T(angle))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function sbend(coords, L::Real, angle::Real;
               species=Species("electron"), p_over_q_ref=1.0)
    return sbend(coords, L, angle, p_over_q_ref; species=species)
end

function sbend(coords, L::Real, angle::Real, p_over_q_ref::Real, species::Species)
    return sbend(coords, L, angle, p_over_q_ref; species=species)
end

function sextupole(coords, L::Real, Kn2::Real, p_over_q_ref::Real;
                   species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn2))
    ele = Sextupole(L=T(L), Kn2=T(Kn2))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function sextupole(coords, L::Real, Kn2::Real;
                   species=Species("electron"), p_over_q_ref=1.0)
    return sextupole(coords, L, Kn2, p_over_q_ref; species=species)
end

function sextupole(coords, L::Real, Kn2::Real, p_over_q_ref::Real, species::Species)
    return sextupole(coords, L, Kn2, p_over_q_ref; species=species)
end

function octupole(coords, L::Real, Kn3::Real, p_over_q_ref::Real;
                  species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn3))
    ele = Octupole(L=T(L), Kn3=T(Kn3))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function octupole(coords, L::Real, Kn3::Real;
                  species=Species("electron"), p_over_q_ref=1.0)
    return octupole(coords, L, Kn3, p_over_q_ref; species=species)
end

function octupole(coords, L::Real, Kn3::Real, p_over_q_ref::Real, species::Species)
    return octupole(coords, L, Kn3, p_over_q_ref; species=species)
end

function solenoid(coords, L::Real, Ksol::Real, p_over_q_ref::Real;
                  species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Ksol))
    ele = Solenoid(L=T(L), Ksol=T(Ksol))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function solenoid(coords, L::Real, Ksol::Real;
                  species=Species("electron"), p_over_q_ref=1.0)
    return solenoid(coords, L, Ksol, p_over_q_ref; species=species)
end

function solenoid(coords, L::Real, Ksol::Real, p_over_q_ref::Real, species::Species)
    return solenoid(coords, L, Ksol, p_over_q_ref; species=species)
end

function hkicker(coords, L::Real, Kn0::Real, p_over_q_ref::Real;
                 species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn0))
    ele = HKicker(L=T(L), Kn0=T(Kn0))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function hkicker(coords, L::Real, Kn0::Real;
                 species=Species("electron"), p_over_q_ref=1.0)
    return hkicker(coords, L, Kn0, p_over_q_ref; species=species)
end

function hkicker(coords, L::Real, Kn0::Real, p_over_q_ref::Real, species::Species)
    return hkicker(coords, L, Kn0, p_over_q_ref; species=species)
end

function vkicker(coords, L::Real, Ks0::Real, p_over_q_ref::Real;
                 species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Ks0))
    ele = VKicker(L=T(L), Ks0=T(Ks0))
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function vkicker(coords, L::Real, Ks0::Real;
                 species=Species("electron"), p_over_q_ref=1.0)
    return vkicker(coords, L, Ks0, p_over_q_ref; species=species)
end

function vkicker(coords, L::Real, Ks0::Real, p_over_q_ref::Real, species::Species)
    return vkicker(coords, L, Ks0, p_over_q_ref; species=species)
end

function rfcavity(coords, L::Real, voltage::Real, frequency::Real, phase::Real,
                  p_over_q_ref::Real; species=Species("electron"))
    T = promote_type(
        eltype(coords),
        typeof(L),
        typeof(voltage),
        typeof(frequency),
        typeof(phase),
    )
    ele = RFCavity(
        L=T(L),
        voltage=T(voltage),
        frequency=T(frequency),
        phase=T(phase),
    )
    return _scibmad_track_ele(T.(coords), ele; species=species, p_over_q_ref=p_over_q_ref)
end

function rfcavity(coords, L::Real, voltage::Real, frequency::Real, phase::Real;
                  species=Species("electron"), p_over_q_ref=1.0)
    return rfcavity(coords, L, voltage, frequency, phase, p_over_q_ref; species=species)
end

function rfcavity(coords, L::Real, voltage::Real, frequency::Real, phase::Real,
                  p_over_q_ref::Real, species::Species)
    return rfcavity(coords, L, voltage, frequency, phase, p_over_q_ref; species=species)
end
'''


class ElementSpec: # lightweight container that remembers an element's kind and parameters without doing anything with them yet (building)
    __slots__ = ("kind", "params")

    def __init__(self, kind, *params):
        """
        Stores the element's kind and its positional parameters

        Arguments:
        - kind: the element type name, e.g. "drift" or "quadrupole" — matches a lowercase function name on the proxy
        - *params: the positional parameters for that element, in call order (e.g. length, strength)

        Returns:
        - (none)
        """
        self.kind = kind
        self.params = params

    def __repr__(self):
        """
        Returns a debug-friendly string representation showing the element's kind and params

        Arguments:
        - (none)

        Returns:
        - repr_str: a string like "ElementSpec(kind='drift', params=(0.5,))"
        """
        return f"ElementSpec(kind={self.kind!r}, params={self.params!r})"


class BeamlineSpec: # Holds an ordered list of ElementSpecs and can track coordinates through them
    # Building one does not trigger any physics computation, that only happens when it is called or .track() is invoked with real coordinates
    __slots__ = ("elements",)

    def __init__(self, elements):
        """
        Stores the list of elements making up the beamline.

        Arguments:
        - elements: an iterable of ElementSpec instances, in beamline order

        Returns:
         (none)
        """
        self.elements = list(elements)

    def track(self, coords):
        """
        Walks the beamline's elements in order, feeding coords through each one in turn by looking up the matching lowercase function on the proxy (core.py) 
        and calling it with the element's stored parameters (where actual physics computation happens for this beamline style)

        Arguments:
        - coords: the starting particle coordinates to track through the beamline

        Returns:
        - coords: the particle coordinates after passing through every element in the beamline
        """
        from . import core as _proxy
        for ele in self.elements:
            fn = getattr(_proxy, ele.kind)
            coords = fn(coords, *ele.params)
        return coords

    def __call__(self, coords):
        """
        Shorthand for track(coords), so a BeamlineSpec instance can be called directly like a function

        Arguments:
        - coords: the starting particle coordinates to track through the beamline

        Returns:
        - coords: the particle coordinates after passing through every element in the beamline
        """
        return self.track(coords)

    def __repr__(self):
        """
        Returns a debug-friendly string representation showing the beamline's elements

        Arguments:
        - (none)

        Returns:
        - repr_str: a string like "BeamlineSpec(elements=[...])"
        """
        return f"BeamlineSpec(elements={self.elements!r})"


OBJECT_PARAM_NAMES = { # Module-level lookup dict mapping each capitalized element to the named parameters ObjectElement required when constructing it in the order they are checked at construction time
    "Quadrupole": ("Kn1", "L"),
    "Drift": ("L",),
    "SBend": ("angle", "L"),
    "Sextupole": ("Kn2", "L"),
    "Octupole": ("Kn3", "L"),
    "Solenoid": ("Ksol", "L"),
    "HKicker": ("Kn0", "L"),
    "VKicker": ("Ks0", "L"),
    "RFCavity": ("voltage", "frequency", "phase", "L"),
}



OBJECT_CALL_PARAM_NAMES = { # Module-level lookup dict mapping each capitalized element kind to its parameter names in the order the underlying Julia track function expects them as positional args
    # used by ObjectElement.ordered_params() to build the correct call
    "Quadrupole": ("L", "Kn1"),
    "Drift": ("L",),
    "SBend": ("L", "angle"),
    "Sextupole": ("L", "Kn2"),
    "Octupole": ("L", "Kn3"),
    "Solenoid": ("L", "Ksol"),
    "HKicker": ("L", "Kn0"),
    "VKicker": ("L", "Ks0"),
    "RFCavity": ("L", "voltage", "frequency", "phase"),
}

# The order in OBJECT_PARAM_NAMES may differ from the order in OBJECT_CALL_PARAM_NAMES

OBJECT_KIND_TO_LOWER = { # Module-level lookup dict mapping each capitalized element kind to the lowercase Julia function name that actually performs the tracking
    "Quadrupole": "quadrupole",
    "Drift": "drift",
    "SBend": "sbend",
    "Sextupole": "sextupole",
    "Octupole": "octupole",
    "Solenoid": "solenoid",
    "HKicker": "hkicker",
    "VKicker": "vkicker",
    "RFCavity": "rfcavity",
}

class ObjectElement: 
    """
    Named-parameter representation of an accelerator element (e.g. Quadrupole(L=0.2, Kn1=1.3)) used by the structured/object-oriented beamline-building style (paired with ObjectBeamline)
    Required and optional parameters per element kind are defined in the module-level OBJECT_PARAM_NAMES dict;
    the order in which parameters must be passed to the underlying Julia function is defined in OBJECT_CALL_PARAM_NAMES
    """
    __slots__ = ("kind", "_params")

    def __init__(self, kind, **kwargs):
         """
        Validates that every required parameter for this element kind was supplied and that no unrecognized parameters were passed, then stores the parameters

        Arguments:
        - kind: the element type name, e.g. "Quadrupole" — must be a key in OBJECT_PARAM_NAMES
        - **kwargs: the element's named parameters, e.g. L=0.2, Kn1=1.3

        Returns:
        - (none)

        Raises:
        - TypeError: if a required parameter is missing, or if an unrecognized keyword argument is passed
        """
        names = OBJECT_PARAM_NAMES[kind]
        missing = [name for name in names if name not in kwargs]
        if missing:
            namestr = ", ".join(missing)
            raise TypeError(f"Missing required parameters for {kind}: {namestr}")

        unknown = sorted(set(kwargs) - set(names))
        if unknown:
            namestr = ", ".join(unknown)
            raise TypeError(f"Unexpected keyword arguments for {kind}: {namestr}")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "_params", {name: kwargs[name] for name in names})

    def __getattr__(self, name):
        """
        Looks up a parameter value by name, so an element's parameters can be read as attributes, ex quad.Kn1.

        Arguments:
        - name: the parameter name being looked up

        Returns:
        - value: the stored value for that parameter

        Raises:
        - AttributeError: if name is not one of this element's stored parameters
        """
        try:
            return self._params[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        """
        Allows updating an existing parameter's value by attribute assignment, ex quad.Kn1 = 1.5. Does not allow adding new parameters that weren't part of the element's definition.

        Arguments:
        - name: the parameter name to update
        - value: the new value to assign

        Returns:
        - (none)

        Raises:
        - AttributeError: if name is not one of this element's existing parameters
        """
        if name in self._params:
            self._params[name] = value
            return
        raise AttributeError(f"{self.kind} has no parameter {name!r}")

    def ordered_params(self):
        """
        Returns this element's parameter values in the specific order the underlying Julia tracking function expects them (per OBJECT_CALL_PARAM_NAMES) 
        (this order can differ from the order parameters are stored/required in (OBJECT_PARAM_NAMES))

        Arguments:
        - (none)

        Returns:
        - params: a tuple of parameter values in Julia call order
        """
        return tuple(self._params[name] for name in OBJECT_CALL_PARAM_NAMES[self.kind])

    def __repr__(self):
        """
        Returns a debug-friendly string representation showing the element's kind and all its parameter values.

        Arguments:
        - (none)

        Returns:
        - repr_str: a string like "Quadrupole(L=0.2, Kn1=1.3)"
        """
        pieces = ", ".join(f"{key}={value!r}" for key, value in self._params.items())
        return f"{self.kind}({pieces})"

class TrackingResult: # A thin wrapper around the stacked tensor output of a multi-turn tracking run (produced by _JuliaProxy.track)
#used only to give the result a nicer repr

    __slots__ = ("v",)

    def __init__(self, v):
        """
        Stores the tracking output tensor

        Arguments:
        - v: the stacked coordinate tensor produced by a tracking run, shape (1, 6, n_saved_turns+1)

        Returns:
        - (none)
        """
        self.v = v

    def __repr__(self):
        """
        Returns a debug-friendly string representation showing the stored tensor

        Arguments:
        - (none)

        Returns:
        - repr_str: a string like "TrackingResult(v=tensor(...))"
        """
        return f"TrackingResult(v={self.v!r})"


class ObjectBeamline: #  Holds an ordered list of ObjectElements plus reference beam information (species, reference momentum or energy), doesn't trigger computation
    __slots__ = ("elements", "species_ref", "p_over_q_ref", "E_ref")

    def __init__(self, elements, *, species_ref=None, p_over_q_ref=None, E_ref=None):
        """
        Stores the beamline's elements and its reference beam parameters, with no validation at construction time

        Arguments:
        - elements: an iterable of ObjectElement instances, in beamline order
        - species_ref: optional reference particle species (e.g. Species("electron"))
        - p_over_q_ref: optional reference momentum-over-charge value
        - E_ref: optional reference energy, used to derive p_over_q_ref if p_over_q_ref itself isn't given

        Returns:
        - (none)
        """
        self.elements = list(elements)
        self.species_ref = species_ref
        self.p_over_q_ref = p_over_q_ref
        self.E_ref = E_ref

    def resolve_p_over_q_ref(self, proxy):
        """
        Resolves the beamline's reference momentum-over-charge value, either using p_over_q_ref directly if it was given, or deriving it from E_ref via the proxy's E_to_R conversion 
        (defaulting species to electron if none was set).

        Arguments:
        - proxy: the _JuliaProxy instance, used to call E_to_R when deriving from E_ref

        Returns:
        - p_over_q_ref: the resolved reference momentum-over-charge value

        Raises:
        - ValueError: if neither p_over_q_ref nor E_ref was set on the beamline
        """
        if self.p_over_q_ref is not None:
            return self.p_over_q_ref
        if self.E_ref is not None:
            species = self.species_ref if self.species_ref is not None else proxy.Species("electron")
            return proxy.E_to_R(species, self.E_ref)
        raise ValueError("Beamline requires either p_over_q_ref or E_ref")

    def __repr__(self):
        """
        ObjectBeamline.__repr__(self)

        Returns a debug-friendly string representation showing the beamline's elements and reference parameters

        Arguments:
        - (none)

        Returns:
        - repr_str: a string like "Beamline(elements=[...], species_ref=..., p_over_q_ref=..., E_ref=...)"
        """
        return (
            f"Beamline(elements={self.elements!r}, species_ref={self.species_ref!r}, "
            f"p_over_q_ref={self.p_over_q_ref!r}, E_ref={self.E_ref!r})"
        )
