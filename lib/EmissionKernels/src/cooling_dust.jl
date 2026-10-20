# cooling_dust.jl — dust THERMAL channels (PE heating, gas-grain coupling, dust
# emission) as pure, precision-generic scalar functions.
#
# These live in EmissionKernels alongside metal_cooling_rate because they are
# radiative/thermal channels of the gas energy equation, not gas-phase reaction
# rates (those — H₂-on-grains, grain-assisted HII recombination — stay in
# ChemistryKernels/rates_dust.jl). ChemistryKernels imports Gamma_PE / Lambda_gr
# here and calls them from its subcycle.
#
# Pattern follows the other cooling_*.jl: R = typeof(first_arg); every numeric
# literal cast to R; no allocation; GPU-safe.
#
# References:
#   Bakes & Tielens (1994) ApJ 427 822          — photoelectric heating
#   Hollenbach & McKee (1989) ApJ 342 306       — gas-grain cooling
#   Hollenbach & McKee (1979); Krumholz+ (2011) — dust thermal emission

# ── Photoelectric heating rate per H nucleus [erg/s] ─────────────────────────
# Bakes & Tielens (1994), Eq. 2. UV photons eject electrons from PAHs/small
# grains; the heating efficiency ε depends on the grain charging parameter
# ψ = G₀√T / nₑ.  Multiply by n_H to get the volumetric heating rate [erg/cm³/s].
@inline function Gamma_PE(T_gas::Real, G0::Real, Z_rel::Real, n_e::Real)
    R = typeof(T_gas)
    (G0 > zero(R) && Z_rel > zero(R) && T_gas > zero(R) && n_e > zero(R)) || return zero(R)
    ψ   = G0 * sqrt(T_gas) / n_e
    # Two-branch efficiency: collisional de-excitation (first) and recombination (second)
    ε_1 = R(4.87e-2) / (one(R) + R(4e-3) * ψ^R(0.73))
    ε_2 = R(3.65e-2) * (T_gas / R(1e4))^R(0.7) / (one(R) + R(2e-4) * ψ)
    # 1.3e-24 erg/s per H atom per Habing field unit; Z_rel scales the grain abundance
    return R(1.3e-24) * (ε_1 + ε_2) * G0 * Z_rel
end

# ── Gas-grain collisional coupling [erg/cm³/s] ───────────────────────────────
# Hollenbach & McKee (1989), positive when gas is hotter than dust (gas cools).
# The sign flips and the gas heats when T_gas < T_dust (e.g., in warm PDR skins).
@inline function Lambda_gr(T_gas::Real, T_dust::Real, n_H::Real, Z_rel::Real)
    R = typeof(T_gas)
    return R(2e-33) * sqrt(T_gas) * (T_gas - T_dust) * n_H^2 * Z_rel
end

# ── Dust thermal emission [erg/cm³/s] ────────────────────────────────────────
# Modified blackbody with κ ∝ ν² → Λ ∝ n_H · Z_rel · T_dust^6 (Hollenbach &
# McKee 1979, Krumholz et al. 2011).  This is the energy the dust radiates away;
# it does NOT directly enter the GAS energy equation (gas couples to dust only
# through Lambda_gr).  Exported as a diagnostic and for energy-balance checks.
@inline function Lambda_dust(T_dust::Real, Z_rel::Real, n_H::Real)
    R = typeof(T_dust)
    return R(2.0e-27) * n_H * Z_rel * T_dust^R(6)
end
