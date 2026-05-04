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
    bunch = Bunch(v; species=bl.species_ref, p_over_q_ref=bl.p_over_q_ref)
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
