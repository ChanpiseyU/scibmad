ELEMENT_JULIA_CODE = """
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

function quadrupole(coords, L::Real, Kn1::Real, R_ref::Real;
                    species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn1), typeof(R_ref))
    ele = Quadrupole(L=T(L), Kn1=T(Kn1))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function quadrupole(coords, L::Real, Kn1::Real;
                    species=Species("electron"), R_ref=1.0)
    return quadrupole(coords, L, Kn1, R_ref; species=species)
end

function drift(coords, L::Real, R_ref::Real;
               species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(R_ref))
    ele = Drift(L=T(L))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function drift(coords, L::Real;
               species=Species("electron"), R_ref=1.0)
    return drift(coords, L, R_ref; species=species)
end

function sbend(coords, L::Real, angle::Real, R_ref::Real;
               species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(angle), typeof(R_ref))
    ele = SBend(L=T(L), angle=T(angle))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function sbend(coords, L::Real, angle::Real;
               species=Species("electron"), R_ref=1.0)
    return sbend(coords, L, angle, R_ref; species=species)
end

function sextupole(coords, L::Real, Kn2::Real, R_ref::Real;
                   species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn2), typeof(R_ref))
    ele = Sextupole(L=T(L), Kn2=T(Kn2))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function sextupole(coords, L::Real, Kn2::Real;
                   species=Species("electron"), R_ref=1.0)
    return sextupole(coords, L, Kn2, R_ref; species=species)
end

function octupole(coords, L::Real, Kn3::Real, R_ref::Real;
                  species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn3), typeof(R_ref))
    ele = Octupole(L=T(L), Kn3=T(Kn3))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function octupole(coords, L::Real, Kn3::Real;
                  species=Species("electron"), R_ref=1.0)
    return octupole(coords, L, Kn3, R_ref; species=species)
end

function solenoid(coords, L::Real, Ksol::Real, R_ref::Real;
                  species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Ksol), typeof(R_ref))
    ele = Solenoid(L=T(L), Ksol=T(Ksol))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function solenoid(coords, L::Real, Ksol::Real;
                  species=Species("electron"), R_ref=1.0)
    return solenoid(coords, L, Ksol, R_ref; species=species)
end

function hkicker(coords, L::Real, Kn0::Real, R_ref::Real;
                 species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Kn0), typeof(R_ref))
    ele = HKicker(L=T(L), Kn0=T(Kn0))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function hkicker(coords, L::Real, Kn0::Real;
                 species=Species("electron"), R_ref=1.0)
    return hkicker(coords, L, Kn0, R_ref; species=species)
end

function vkicker(coords, L::Real, Ks0::Real, R_ref::Real;
                 species=Species("electron"))
    T = promote_type(eltype(coords), typeof(L), typeof(Ks0), typeof(R_ref))
    ele = VKicker(L=T(L), Ks0=T(Ks0))
    return _scibmad_track_ele(T.(coords), ele; species=species, R_ref=T(R_ref))
end

function vkicker(coords, L::Real, Ks0::Real;
                 species=Species("electron"), R_ref=1.0)
    return vkicker(coords, L, Ks0, R_ref; species=species)
end

function rfcavity(coords, L::Real, voltage::Real, frequency::Real, phase::Real,
                  R_ref::Real; species=Species("electron"))
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

function rfcavity(coords, L::Real, voltage::Real, frequency::Real, phase::Real;
                  species=Species("electron"), R_ref=1.0)
    return rfcavity(coords, L, voltage, frequency, phase, R_ref; species=species)
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