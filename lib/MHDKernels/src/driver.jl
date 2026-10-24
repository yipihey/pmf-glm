# ── Time integration driver: step! / evolve! ─────────────────────────────────
# GLM cleaning schedule (Dedner+ 2002): by default the hyperbolic cleaning speed
# `ch` is the max MHD signal speed and ψ is damped each step by
# `decay = exp(-cr·ch·dt/dx)`. PMF runs can raise `ch` above the physical MHD
# speed; `evolve!` then includes that speed in the CFL denominator.
export step!, evolve!

const GLM_CR = 0.18

"""
    step!(s, dt; ch, glm_cr=GLM_CR, integrator=:ref)

Advance the state one step of size `dt` with GLM cleaning speed `ch`. `integrator`
selects `:ref` (portable per-cell) or `:cube` (GPU shared-memory; added next).

For a constrained-transport state (`mhd_solver=:ct`) there is no cleaning: `ch` and
`glm_cr` are ignored (`ch` may be omitted), nothing is divided by `ch`, and only
`integrator=:ref` is available.
"""
function step!(s::MHDState{T}, dt::Real; ch::Union{Nothing,Real} = nothing,
               glm_cr::Real = GLM_CR, integrator::Symbol = :ref) where {T}
    if uses_ct(s)
        integrator === :ref || integrator === :cube ||
            error("unknown integrator :$integrator (have :ref, :cube)")
        integrator === :ref || _require_ct_integrator(s, "step!")
        return step_ref!(s, dt; ch = zero(T), decay = one(T))
    end
    ch === nothing && error("step! needs the GLM cleaning speed `ch` (GLM states)")
    decay = exp(-T(glm_cr) * T(ch) * T(dt) / s.dx)
    if integrator === :ref
        step_ref!(s, dt; ch = ch, decay = decay)
    elseif integrator === :cube
        step_cube!(s, dt; ch = ch, decay = decay)
    else
        error("unknown integrator :$integrator (have :ref, :cube)")
    end
end

"""
    evolve!(s, tfinal; cfl=0.4, glm_ch_fac=1, glm_cr=GLM_CR,
            integrator=:ref, callback=nothing) -> (t, nsteps)

Integrate to `tfinal` with CFL-limited steps. `callback(s, t, n)` (if given) runs
after each step (diagnostics/output). Returns the final time and step count.
For CT states `glm_ch_fac`/`glm_cr` are irrelevant and `dt` is never shrunk by the
cleaning speed.
"""
function evolve!(s::MHDState{T}, tfinal::Real; cfl::Real = 0.4,
                 glm_ch_fac::Real = 1, glm_cr::Real = GLM_CR,
                 integrator::Symbol = :ref, callback = nothing) where {T}
    glm_ch_fac > 0 || error("glm_ch_fac must be positive")
    glm_cr >= 0 || error("glm_cr must be non-negative")
    t = zero(Float64); n = 0; tf = Float64(tfinal)
    while t < tf * (1 - 1e-9)
        dt, smax = compute_dt(s; cfl = cfl)
        if uses_ct(s)
            ch = zero(T)
            dtf = Float64(dt)
        else
            ch = T(glm_ch_fac) * T(smax)
            dtf = Float64(dt) * Float64(smax) / Float64(max(T(smax), ch))
        end
        if t + dtf > tf
            dtf = tf - t
        end
        step!(s, dtf; ch = ch, glm_cr = glm_cr, integrator = integrator)
        t += dtf; n += 1
        callback === nothing || callback(s, t, n)
    end
    return t, n
end
