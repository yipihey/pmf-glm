# cooling_tables.jl — OPTIONAL log–log lookup for the temperature-dependent cooling
# coefficients that `cooling_rate_total` evaluates EVERY chemistry sub-step.
#
# Cooling is the second-largest slice of the stiff network (~23–27% on the GPU after the
# rate tables).  Like the rate fits, the cooling channels are transcendental functions of
# T re-evaluated each sub-step (T moves), plus six evaluated at the FROZEN CMB temperature
# Tc for the H₂ optically-thick floor.  This tabulates all of them on a log-spaced T grid
# and interpolates linearly in (log₁₀T, log₁₀ coeff): power-law channels exact, exponential
# ones smooth, monotonic (no spurious extrema), branchless, AD-friendly — exactly as the
# ChemistryKernels rate table.  The analytic `cooling_rate_total` stays the default/reference;
# `cooling_rate_total_tab` is the opt-in twin with identical term order.

# Tabulated columns IN ORDER: the four atomic channels, the five GA* H₂ density-limited
# rates, the H₂ LTE rate, and the two HD rates. Exact-zero nodes are encoded as `-Inf`.
const _CT_COLS = (:ceHI, :ciHI, :reHII, :brem,
                  :GAHI, :GAH2, :GAHe, :GAHp, :GAel, :H2LTE,
                  :HDlte, :HDlow)
const _NCT = length(_CT_COLS)

"""
    CoolingTables(logk, x0, invdx, N)

Precomputed log–log cooling table: `(N, $(_NCT))` array of `log₁₀(coeff)` on a uniform
`log₁₀T` grid, with exact zero encoded as `-Inf` (`x0=log₁₀Tmin`,
`invdx=1/Δlog₁₀T`). Device-resident (adapts inside a kernel
via `Adapt`).  Build with [`build_cooling_tables`](@ref); use via
`cooling_rate_total(...; tables=…)` / [`cooling_rate_total_tab`](@ref).
"""
struct CoolingTables{A,R}
    logk::A
    x0::R
    invdx::R
    N::Int
end
Adapt.@adapt_structure CoolingTables

"""
    build_cooling_tables(; Tmin=1.0, Tmax=1.0e9, N=1024, precision=Float64, backend=:cpu)

Evaluate every tabulated cooling coefficient on a uniform `log₁₀T` grid of `N` points over
`[Tmin, Tmax]` K, store `log₁₀(coeff)`, and upload to `backend`.  The default `N=1024`
(≈0.0088 dex, ~0.1 MB) holds the cooling sum to ≲1% across the physical range — well inside
the fits' own uncertainty — and stays L2-resident.
"""
function build_cooling_tables(; Tmin::Real = 1.0, Tmax::Real = 1.0e9, N::Int = 1024,
                              precision::Type = Float64, backend::Symbol = :cpu)
    R  = precision
    x0 = log10(Float64(Tmin)); x1 = log10(Float64(Tmax))
    dx = (x1 - x0) / (N - 1)
    M  = Array{R}(undef, N, _NCT)
    for j in 1:N
        Tj   = R(10.0)^R(x0 + (j - 1) * dx)
        vals = (ceHI(Tj), ciHI(Tj), reHII(Tj), brem(Tj),
                GAHI(Tj), GAH2(Tj), GAHe(Tj), GAHp(Tj), GAel(Tj), H2LTE(Tj),
                HDlte(Tj), HDlow(Tj))
        @inbounds for c in 1:_NCT
            v = vals[c]
            M[j, c] = v > zero(R) ? log10(v) : -R(Inf)
        end
    end
    dev = to_device(EmissionKernels.backend(backend), M, R)
    return CoolingTables(dev, R(x0), R(1.0 / dx), N)
end

# Locate T on the log grid → (lower-node 1-based row i, fraction f), clamped to endpoints.
@inline function _ct_locate(x0, invdx, N::Int, T)
    R = typeof(T)
    s = (log10(T) - R(x0)) * R(invdx)
    smax = R(N - 1)
    s = clamp(s, zero(R), smax - eps(R)*smax)
    b = unsafe_trunc(Int, s)
    return b + 1, s - R(b)
end
# Positive endpoints use log–log interpolation. Zero endpoints remain exact and a
# mixed boundary cell uses a linear ramp, so no interpolation ever forms `Inf - Inf`.
@inline function _ct_read(L, N::Int, i::Int, f, c::Int)
    @inbounds lo = L[i + (c - 1) * N]
    @inbounds hi = L[i + 1 + (c - 1) * N]
    lo_finite = isfinite(lo)
    hi_finite = isfinite(hi)
    lo_finite && hi_finite && return exp10(lo + f * (hi - lo))
    !lo_finite && !hi_finite && return zero(f)
    lo_finite && return (one(f) - f) * exp10(lo)
    return f * exp10(hi)
end

"""
    cooling_rate_total_tab(ct::CoolingTables, nHI,nHII,nHeI,nde,nH2,nHD,T,z; nH, metals)

Tabulated twin of [`cooling_rate_total`](@ref): identical assembly and term order, but the
T-dependent (and the Tc, CMB-floor) cooling coefficients come from `ct` by log–log
interpolation instead of analytic fits.  `comp1_cmb`/`comp2_cmb`/`metal_cooling_rate` stay
analytic (cheap / out of the hot tabulated set).
"""
@inline function cooling_rate_total_tab(ct::CoolingTables, nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                                        ih2optical::Bool = false, nH = nothing, metals = nothing)
    R    = typeof(T)
    one_ = one(R)
    Tc   = comp2_cmb(R(z))
    L    = ct.logk; N = ct.N
    iT, fT = _ct_locate(ct.x0, ct.invdx, N, T)
    iC, fC = _ct_locate(ct.x0, ct.invdx, N, Tc)

    atomic = (_ct_read(L,N,iT,fT,1) + _ct_read(L,N,iT,fT,2)) * nHI * nde +
             _ct_read(L,N,iT,fT,3) * nHII * nde +
             _ct_read(L,N,iT,fT,4) * nHII * nde

    galdl = _ct_read(L,N,iT,fT,5)*nHI + _ct_read(L,N,iT,fT,6)*nH2 + _ct_read(L,N,iT,fT,7)*nHeI +
            _ct_read(L,N,iT,fT,8)*nHII + _ct_read(L,N,iT,fT,9)*nde
    h2lte = _ct_read(L,N,iT,fT,10)
    cool_gas = _two_level_cooling(h2lte, galdl)
    galdl_c = _ct_read(L,N,iC,fC,5)*nHI + _ct_read(L,N,iC,fC,6)*nH2 + _ct_read(L,N,iC,fC,7)*nHeI +
              _ct_read(L,N,iC,fC,8)*nHII + _ct_read(L,N,iC,fC,9)*nde
    h2lte_c  = _ct_read(L,N,iC,fC,10)
    cool_cmb = _two_level_cooling(h2lte_c, galdl_c)
    fudge = one_
    if ih2optical && nH !== nothing
        fudge = min((R(nH) / R(8.0e9))^R(-0.45), one_)
    end
    h2 = fudge * nH2 * (cool_gas - cool_cmb)

    hd = zero(R)
    if T > Tc
        hdlte = _ct_read(L,N,iT,fT,11)
        hdlow = _ct_read(L,N,iT,fT,12)
        hd = nHD * _two_level_cooling(hdlte, nHI * hdlow)
    end

    compton = comp1_cmb(R(z)) * (T - Tc) * nde

    metal = metals === nothing ? zero(R) :
            metal_cooling_rate(T, R(z), nHI, nHII, nde, nH2, R(nH), metals)

    return atomic + h2 + hd + compton + metal
end
