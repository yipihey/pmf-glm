# Radiation-step positivity protection (trial admissibility, restore, robust minmod-PLM/HLL retry,
# subdivision, checkpoint fallback) for the constrained-transport solver on the :ref integrator.
# GLM protects only :cube; its :ref path is intentionally unchanged (no fallback). The GLM :cube vs
# CT equivalence runs on the Metal backend: tools/ct_fallback_equivalence/gpu_equiv.jl.
module CTFallbackHarness
include(joinpath(@__DIR__, "..", "..", "tools", "ct_fallback_equivalence", "lib.jl"))
end

@testset "CT radiation positivity fallback (:ref)" begin
    H = CTFallbackHarness
    # Problem A: strong rarefaction (u=10 step, p0=1e-3), single step at cfl 1.5: both the PLM/HLL trial
    # and the minmod-PLM/HLL retry are rejected, so the step is subdivided once.
    # Problem B3: weak B, 8 steps at cfl 0.8: the retry is accepted (robust fallback, no subdivision).
    function setup(u, p0, cfl, bscale)
        s0 = H.make_state(:glm, :cpu; u=u, p0=p0, bscale=bscale)
        sm = Float32(max_wavespeed!(allocate_radiation_workspace(s0).fft_real, s0))
        (dt=cfl*s0.dx/sm, ch=2sm)
    end
    gate = max(1f-6, 1f-4)   # smallr, llf_dmin of the default state

    @testset "workspace memory: 6 face arrays for CT, none for GLM" begin
        c = H.make_state(:ct, :cpu); g = H.make_state(:glm, :cpu)
        @test allocate_radiation_workspace(g).ct_snapshot === nothing
        snap = allocate_radiation_workspace(c).ct_snapshot
        @test length(snap) == 6*ncells(c) && eltype(snap) == Float32
    end

    @testset "problem A: subdivision recovers where GLM :ref has no protection" begin
        (dt, ch) = setup(10f0, 1f-3, 1.5f0, 1f0)
        g = H.advance(H.make_state(:glm, :cpu; u=10f0, p0=1f-3), :ref; nsteps=1, dt=dt, ch=ch)
        @test minimum(fields_to_host(g.s)[1]) < gate          # unprotected: density at the floor
        @test g.tm.fallback_steps == 0 && g.tm.subdivision_steps == 0
        sc = H.make_state(:ct, :cpu; u=10f0, p0=1f-3)
        mass0 = sum(Float64, fields_to_host(sc)[1])
        c = H.advance(sc, :ref; nsteps=1, dt=dt, ch=ch)
        h = fields_to_host(c.s)
        @test c.err === nothing && all(x -> all(isfinite, x), h)
        @test minimum(h[1]) >= gate
        @test c.tm.subdivision_steps == 1 && c.tm.plm_hll_fallback_steps == 0
        @test H.branch(c) == "subdivision"
        brms = sqrt(sum(abs2, Float64.(h[6]))/length(h[6]))
        @test c.divmax/brms < 1e-5                             # face divergence at round-off
        @test abs(sum(Float64, h[1])/mass0 - 1) < 1e-5
        @printf("  CT problem A: rhomin=%.4e subdiv=%d divB_face*dx/Brms=%.2e mass drift=%.2e\n",
                minimum(h[1]), c.tm.subdivision_steps, c.divmax/brms, abs(sum(Float64, h[1])/mass0 - 1))
        # the solver settings are not changed by the retry
        @test c.s.recon === :plm && c.s.riemann === :hll
    end

    @testset "problem B3: robust minmod-PLM/HLL retry accepted" begin
        (dt, ch) = setup(10f0, 1f-3, 0.8f0, 0.1f0)
        sc = H.make_state(:ct, :cpu; u=10f0, p0=1f-3, bscale=0.1f0)
        c = H.advance(sc, :ref; nsteps=8, dt=dt, ch=ch)
        h = fields_to_host(c.s)
        @test c.err === nothing && all(x -> all(isfinite, x), h) && minimum(h[1]) >= gate
        @test H.branch(c) == "robust-PLM/HLL"
        @test c.tm.subdivision_steps == 0
        @test c.tm.plm_hll_fallback_steps >= 1 && c.tm.fallback_steps == c.tm.plm_hll_fallback_steps
        @test length(c.log.msgs) == 1 && occursin("minmod-PLM/HLL fallback", c.log.msgs[1])
        brms = sqrt(sum(abs2, Float64.(h[6]))/length(h[6]))
        @test c.divmax/brms < 1e-5
        @printf("  CT problem B3: robust fallbacks=%d rhomin=%.4e divB_face*dx/Brms=%.2e\n",
                c.tm.plm_hll_fallback_steps, minimum(h[1]), c.divmax/brms)
    end

    @testset "benign step: protection on equals protection off, bitwise" begin
        (dt, ch) = setup(5f0, 1f-3, 0.3f0, 1f0)
        on  = H.advance(H.make_state(:ct, :cpu; u=5f0, p0=1f-3), :ref; nsteps=4, dt=dt, ch=ch)
        off = H.advance(H.make_state(:ct, :cpu; u=5f0, p0=1f-3), :ref; nsteps=4, dt=dt, ch=ch,
                        positivity_fallback=false)
        @test on.tm.fallback_steps == 0 && on.tm.subdivision_steps == 0
        @test all(fields_to_host(on.s)[v] == fields_to_host(off.s)[v] for v in 1:9)
        @test all(getfield(on.s.ct, f) == getfield(off.s.ct, f) for f in (:bxf, :byf, :bzf, :rbx, :rby, :rbz))
    end

    # Rejected steps must leave the COMPLETE pre-step state: cells, faces and their Kahan remainders
    # (restored exactly, not reset), the density perturbation (re-initialised) and residual workspaces.
    for (label, kw, pattern) in (
            ("checkpoint fallback", (checkpoint_fallback_on_rejection=true,), "RADIATION_CHECKPOINT_CFL_FALLBACK"),
            ("subdivision budget exhausted", (positivity_max_subdivisions=0,), "remained inadmissible after 0 subdivisions"))
        @testset "$label: error leaves the pre-step CT state bitwise" begin
            (dtA, chA) = setup(10f0, 1f-3, 1.5f0, 1f0)
            s = H.make_state(:ct, :cpu; u=10f0, p0=1f-3)
            ws = allocate_radiation_workspace(s)
            # benign steps first so the face remainders are nonzero
            (dtb, chb) = setup(10f0, 1f-3, 0.05f0, 1f0)
            for _ in 1:3
                step_radiation_godunov!(s, ws, dtb; drag_impulse=0.1f0*dtb, closure=H.CLOSURE, ch=chb,
                                        integrator=:ref)
            end
            @test any(!iszero, s.ct.rbx) || any(!iszero, s.ct.rby) || any(!iszero, s.ct.rbz)
            cells = fields_to_host(s)
            faces = map(copy, (s.ct.bxf, s.ct.byf, s.ct.bzf, s.ct.rbx, s.ct.rby, s.ct.rbz))
            delta = reshape(cells[1], s.dims) .- ws.rho_mean
            err = try
                step_radiation_godunov!(s, ws, dtA; drag_impulse=0.1f0*dtA, closure=H.CLOSURE, ch=chA,
                                        integrator=:ref, kw...)
                nothing
            catch e
                e
            end
            @test err isa ErrorException && occursin(pattern, sprint(showerror, err))
            after = fields_to_host(s)
            @test all(after[v] == cells[v] for v in 1:9)
            @test all(a == b for (a, b) in zip((s.ct.bxf, s.ct.byf, s.ct.bzf, s.ct.rbx, s.ct.rby, s.ct.rbz), faces))
            @test Array(ws.density_perturbation) == delta
            @test max_divb_face(s)*Float64(s.dx)/0.4 < 1e-5
        end
    end
end
