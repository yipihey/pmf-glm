# Linear stability scan of the unsplit 3D MUSCL-Hancock/HLLD `:ref` integrator (no corner
# transport upwind terms) versus CFL, for the GLM and CT solvers.  CPU, Float64, periodic box,
# uniform rho=1, p=1, B = bscale*(0.8,0.5,0.4), optional bulk velocity vbulk*(1,1,1), seeded with
# an oblique k=(1,2,3) wave plus 1e-6 white noise in rho, v, p and (div-free, centred-curl) B.
# The run is flagged BLOWUP once max|rho-1| exceeds 0.1.  `nominal cfl` is the number passed to
# `evolve!`: it is the Courant number of the signal speed `smax` for CT, and of the cleaning
# speed `ch = glm_ch_fac*smax` (glm_ch_fac = 2, the production default) for GLM.
#
#   julia --project=. -t 4 tools/cfl_stability/cfl_stability.jl glm plm 16 1500 0.2 0.25 0.3
#   BSCALE=0 VBULK=0 julia ... (environment overrides the field strength / bulk velocity)
using MHDKernels, Random

function cfl_stability_run(solver::Symbol, cfl::Real; recon::Symbol=:plm, N::Int=16,
                           nsteps::Int=1500, bscale::Real=1.0, vbulk::Real=0.0,
                           glm_ch_fac::Real=2, eps0::Real=1e-6, seed::Int=11)
    γ = 5 / 3
    be = MHDKernels.backend(:cpu)
    s = MHDKernels.allocate_state(be, Float64, (N, N, N); dx=1.0 / N, gamma=γ,
                                  recon=recon, mhd_solver=solver)
    rng = MersenneTwister(seed)
    B0 = bscale .* (0.8, 0.5, 0.4)
    Ax = eps0 * randn(rng, N, N, N); Ay = eps0 * randn(rng, N, N, N); Az = eps0 * randn(rng, N, N, N)
    idx(i, j, k) = mod1(i, N) + N * (mod1(j, N) - 1) + N * N * (mod1(k, N) - 1)
    for k in 1:N, j in 1:N, i in 1:N
        c = idx(i, j, k)
        ph = 2π * (i + 2j + 3k) / N
        ρ = 1 + eps0 * (sin(ph) + randn(rng))
        vx = vbulk + eps0 * (0.5cos(ph) + randn(rng))
        vy = vbulk + eps0 * (0.3cos(ph) + randn(rng))
        vz = vbulk + eps0 * (0.2cos(ph) + randn(rng))
        bx = B0[1] + (Az[idx(i,j+1,k)] - Az[idx(i,j-1,k)] - Ay[idx(i,j,k+1)] + Ay[idx(i,j,k-1)]) * N / 2
        by = B0[2] + (Ax[idx(i,j,k+1)] - Ax[idx(i,j,k-1)] - Az[idx(i+1,j,k)] + Az[idx(i-1,j,k)]) * N / 2
        bz = B0[3] + (Ay[idx(i+1,j,k)] - Ay[idx(i-1,j,k)] - Ax[idx(i,j+1,k)] + Ax[idx(i,j-1,k)]) * N / 2
        p = 1 + eps0 * sin(ph)
        s.U[1][c] = ρ; s.U[2][c] = ρ * vx; s.U[3][c] = ρ * vy; s.U[4][c] = ρ * vz
        s.U[6][c] = bx; s.U[7][c] = by; s.U[8][c] = bz; s.U[9][c] = 0
        s.U[5][c] = p / (γ - 1) + 0.5ρ * (vx^2 + vy^2 + vz^2) + 0.5 * (bx^2 + by^2 + bz^2)
    end
    solver === :ct && MHDKernels.ct_faces_from_cells!(s)
    amp(s) = maximum(abs.(s.U[1] .- 1.0))
    a0 = amp(s); amax = a0; steps = 0; blown = false
    try
        MHDKernels.evolve!(s, 1e9; cfl=cfl, glm_ch_fac=glm_ch_fac, callback=(s, t, n) -> begin
            a = amp(s); steps = n; amax = max(amax, a)
            (!isfinite(a) || a > 0.1) && (blown = true; error("blowup"))
            n >= nsteps && error("done")
        end)
    catch e
        occursin("blowup", sprint(showerror, e)) || occursin("done", sprint(showerror, e)) || rethrow()
    end
    return (; solver, recon, N, cfl, steps, amp0=a0, amp_max=amax, growth_max=amax / a0, blown)
end

if abspath(PROGRAM_FILE) == @__FILE__
    solver = Symbol(ARGS[1]); recon = Symbol(ARGS[2]); N = parse(Int, ARGS[3])
    nsteps = parse(Int, ARGS[4]); cfls = parse.(Float64, ARGS[5:end])
    bscale = parse(Float64, get(ENV, "BSCALE", "1")); vbulk = parse(Float64, get(ENV, "VBULK", "0"))
    for cfl in cfls
        r = cfl_stability_run(solver, cfl; recon, N, nsteps, bscale, vbulk)
        println("RESULT solver=$(r.solver) recon=$(r.recon) N=$(r.N) bscale=$bscale vbulk=$vbulk ",
                "cfl=$(r.cfl) steps=$(r.steps) growth_max=$(r.growth_max) ",
                "status=$(r.blown ? "BLOWUP" : "ok")")
    end
end
