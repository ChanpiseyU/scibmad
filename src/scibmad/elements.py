ELEMENT_JULIA_CODE = """
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
"""


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
        from . import core as _proxy
        for ele in self.elements:
            fn = getattr(_proxy, ele.kind)
            coords = fn(coords, *ele.params)
        return coords

    def __call__(self, coords):
        return self.track(coords)

    def __repr__(self):
        return f"BeamlineSpec(elements={self.elements!r})"


OBJECT_PARAM_NAMES = {
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

OBJECT_CALL_PARAM_NAMES = {
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

OBJECT_KIND_TO_LOWER = {
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
    __slots__ = ("kind", "_params")

    def __init__(self, kind, **kwargs):
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
        try:
            return self._params[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name in self._params:
            self._params[name] = value
            return
        raise AttributeError(f"{self.kind} has no parameter {name!r}")

    def ordered_params(self):
        return tuple(self._params[name] for name in OBJECT_CALL_PARAM_NAMES[self.kind])

    def __repr__(self):
        pieces = ", ".join(f"{key}={value!r}" for key, value in self._params.items())
        return f"{self.kind}({pieces})"


class TrackingResult:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return f"TrackingResult(v={self.v!r})"


class ObjectBeamline:
    __slots__ = ("elements", "species_ref", "p_over_q_ref", "E_ref")

    def __init__(self, elements, *, species_ref=None, p_over_q_ref=None, E_ref=None):
        self.elements = list(elements)
        self.species_ref = species_ref
        self.p_over_q_ref = p_over_q_ref
        self.E_ref = E_ref

    def resolve_p_over_q_ref(self, proxy):
        if self.p_over_q_ref is not None:
            return self.p_over_q_ref
        if self.E_ref is not None:
            species = self.species_ref if self.species_ref is not None else proxy.Species("electron")
            return proxy.E_to_R(species, self.E_ref)
        raise ValueError("Beamline requires either p_over_q_ref or E_ref")

    def __repr__(self):
        return (
            f"Beamline(elements={self.elements!r}, species_ref={self.species_ref!r}, "
            f"p_over_q_ref={self.p_over_q_ref!r}, E_ref={self.E_ref!r})"
        )
