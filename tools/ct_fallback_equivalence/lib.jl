# Shared helpers for the CT positivity-fallback equivalence experiments.
# Usage: include("lib.jl") after `using MHDKernels` (and `import Metal` for the Metal backend).
using MHDKernels, Printf, Logging
using MHDKernels: backend, allocate_state, allocate_radiation_workspace,
    step_radiation_godunov!, RadiationStepTiming, fields_to_host, max_wavespeed!,
    linidx, ct_faces_from_cells!, max_divb_face, to_device, uses_ct

# Collects @warn records (message text + keyword values) without printing them.
struct CollectLogger <: Logging.AbstractLogger
    msgs::Vector{String}
end
Logging.min_enabled_level(::CollectLogger) = Logging.Warn
Logging.shouldlog(::CollectLogger, args...) = true
Logging.catch_exceptions(::CollectLogger) = true
function Logging.handle_message(l::CollectLogger, level, message, args...; kwargs...)
    push!(l.msgs, string(message))
end

# 123-type rarefaction in x with weak oblique B: vx = -u (x<1/2), +u (x>=1/2), uniform rho=1,
# cold gas (pressure p0), Bx(y) = 0.3+0.2 sin(2pi y), By = bamp cos(2pi x), Bz = bamp sin(2pi x+1/2).
function make_state(solver::Symbol, bname::Symbol; N=16, u=10f0, p0=1f-5, bamp=0.1f0, T=Float32)
    gamma = T(5)/3
    s = allocate_state(backend(bname), T, (N,N,N); dx=1/N, gamma=gamma, riemann=:hll, mhd_solver=solver)
    h = ntuple(_ -> zeros(T, N^3), 9)
    for k in 1:N, j in 1:N, i in 1:N
        c = linidx(N,N,i,j,k); x=(i-1)/T(N); y=(j-1)/T(N)
        rho = one(T); vx = x < 0.5 ? -u : u; vy = T(0.05)*u*sin(2T(pi)*x+1)*cos(2T(pi)*y)
        bx = T(0.2)*sin(2T(pi)*y)+T(0.3); by = bamp*cos(2T(pi)*x); bz = bamp*sin(2T(pi)*x+T(0.5))
        h[1][c]=rho; h[2][c]=rho*vx; h[3][c]=rho*vy
        h[6][c]=bx; h[7][c]=by; h[8][c]=bz
        h[5][c]=p0/(gamma-1)+T(0.5)*rho*(vx^2+vy^2)+T(0.5)*(bx^2+by^2+bz^2)
    end
    for v in 1:9; copyto!(s.U[v], to_device(s.be, h[v], T)); end
    solver === :ct && ct_faces_from_cells!(s)
    s
end

const CLOSURE = MHDKernels.RadiationClosure{Float32}(1e-4, 0.001, 0.01, 1, 5, 4/3)  # photon coupling negligible

"""
Advance `nsteps` fixed-dt radiation-coupled steps. Returns a NamedTuple with the final state,
timing counters, collected warnings, thrown error (or nothing) and per-step records.
"""
function advance(s, integrator; nsteps, dt, ch, kw...)
    ws = allocate_radiation_workspace(s)
    tm = RadiationStepTiming()
    log = CollectLogger(String[])
    err = nothing; done = 0; divmax = 0.0
    for n in 1:nsteps
        try
            with_logger(log) do
                step_radiation_godunov!(s, ws, Float32(dt); drag_impulse=0f0, closure=CLOSURE,
                                        ch=Float32(ch), integrator=integrator, timing=tm, kw...)
            end
            done += 1
            uses_ct(s) && (divmax = max(divmax, max_divb_face(s)*Float64(s.dx)))
        catch e
            err = e; break
        end
    end
    (; s, ws, tm, log, err, done, divmax)
end

branch(r) = r.err !== nothing ? "error" :
            r.tm.subdivision_steps > 0 ? "subdivision" :
            r.tm.plm_hll_fallback_steps > 0 ? "robust-PLM/HLL" : "none"

function finite_min(s)
    h = fields_to_host(s)
    all(x->all(isfinite,x),h), minimum(h[1])
end

# relative L1 differences |a-b|_1/|b|_1 of rho, |mom|, E, |B| (cell centred) between two states
function rel_diffs(a, b)
    ha = fields_to_host(a); hb = fields_to_host(b)
    l1(x,y) = sum(abs.(x .- y)) / max(sum(abs.(y)), eps(Float64))
    magn(h, v) = sqrt.(Float64.(h[v[1]]).^2 .+ Float64.(h[v[2]]).^2 .+ Float64.(h[v[3]]).^2)
    (rho = l1(Float64.(ha[1]), Float64.(hb[1])),
     mom = l1(magn(ha,(2,3,4)), magn(hb,(2,3,4))),
     E   = l1(Float64.(ha[5]), Float64.(hb[5])),
     B   = l1(magn(ha,(6,7,8)), magn(hb,(6,7,8))))
end

function report(label, r)
    fin, rmin = finite_min(r.s)
    div = uses_ct(r.s) ? @sprintf(" divB_face*dx(max over steps)=%.3e", r.divmax) : ""
    @printf("%-28s steps=%d finite=%s rhomin=%.4e branch=%s robust_fb=%d subdiv=%d warns=%d%s err=%s\n",
            label, r.done, fin, rmin, branch(r), r.tm.plm_hll_fallback_steps, r.tm.subdivision_steps,
            length(r.log.msgs), div,
            r.err === nothing ? "none" : first(split(sprint(showerror, r.err), '\n')))
end
