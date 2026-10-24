# The fused :cube GLM integrator is the reference for the GPU throughput path and the only GLM
# integrator with radiation positivity protection, so it must run (and be tested) on the CPU
# KernelAbstractions backend. KA's CPU lowering emits one workitem loop per `@synchronize`
# section: a local defined in one section is out of scope in the next. These tests would fail
# with `UndefVarError: ox` if the block origin were assigned once at the top of the kernel.
module CubeCPUTests

using Test
using KernelAbstractions
using PMFGLM

const MHD = PMFGLM.MHDKernels
const KA = KernelAbstractions

module FallbackHarness
include(joinpath(@__DIR__, "..", "tools", "ct_fallback_equivalence", "lib.jl"))
end

function smooth_fields(n)
    fields = ntuple(_ -> zeros(Float32, n^3), 9)
    for k in 1:n, j in 1:n, i in 1:n
        x = Float32(2pi * (i - 0.5) / n)
        y = Float32(2pi * (j - 0.5) / n)
        z = Float32(2pi * (k - 0.5) / n)
        u = MHD.prim2cons((1f0 + 0.02f0 * sin(x) * cos(y),
                           0.02f0 * sin(z), 0.02f0 * sin(x), 0.02f0 * sin(y),
                           1f0 + 0.01f0 * cos(z),
                           0.3f0 + 0.02f0 * sin(y), 0.02f0 * sin(z), 0.02f0 * sin(x), 0f0),
                          Float32(5/3))
        index = MHD.linidx(n, n, i, j, k)
        for v in 1:9
            fields[v][index] = u[v]
        end
    end
    fields
end

function make_state(fields, n; recon, riemann)
    s = MHD.allocate_state(MHD.backend(:cpu), Float32, (n,n,n); dx=1/n, gamma=5/3,
                           recon, riemann)
    for v in 1:9
        copyto!(s.U[v], fields[v])
    end
    s
end

# Same tolerances as the GPU cube-vs-CPU-ref parity checks in test_backends.jl.
function assert_fields(reference, candidate; rtol=5f-5, atol=2f-6)
    for v in 1:9
        @test all(isfinite, candidate[v])
        @test maximum(abs, candidate[v] .- reference[v]) <=
              atol + rtol * maximum(abs, reference[v])
    end
    @test minimum(candidate[1]) > 0f0
end

# The 4x4x2-block kernel that Metal launches (lib/MHDKernels/ext/MHDKernelsMetalExt.jl), run
# on the CPU backend.
function step_cube_rect_cpu!(s::MHD.MHDState{T}, dt, ch, decay; drag=nothing) where {T}
    N = s.dims[1]; dtdx = T(dt) / s.dx; cx, cy, cz = MHD.bc_codes(s)
    nbx = N ÷ MHD.RCTBX; nby = N ÷ MHD.RCTBY; nbz = N ÷ MHD.RCTBZ
    physics = (dtdx, s.γ, T(ch), T(decay), s.smallr, s.pfl, s.llf_dmin, s.llf_pmin)
    dc = drag === nothing ? MHD._no_drag_coefficients(T) : MHD._drag_coefficients(T(drag))
    MHD.step_cube_rect_kernel!(s.be, MHD.METAL_CUBE_GS)(
        s.scratch..., s.U..., N, nbx, nby, cx, cy, cz, Val(Float32),
        Val(MHD.recon_code_of(s)), Val(MHD.all_periodic(s)), Val(MHD.riemann_code_of(s)),
        physics, Val(drag !== nothing), dc, Val(false), Val(false), Val(false), Val(false),
        Val(false); ndrange = nbx * nby * nbz * MHD.METAL_CUBE_GS)
    KA.synchronize(s.be)
    s.U, s.scratch = s.scratch, s.U
    s
end

@testset "GLM cube integrator on the CPU backend" begin
    n = 16
    fields = smooth_fields(n)
    mass0 = sum(Float64, fields[1])
    for recon in (:plm, :ppm), riemann in (:hlld, :hll, :llf)
        @testset "$recon / $riemann" begin
            ref = make_state(fields, n; recon, riemann)
            cube = make_state(fields, n; recon, riemann)
            rect = make_state(fields, n; recon, riemann)
            refd = make_state(fields, n; recon, riemann)
            cubed = make_state(fields, n; recon, riemann)
            rectd = make_state(fields, n; recon, riemann)
            dt, speed = MHD.compute_dt(ref; cfl=0.08)
            decay = exp(-MHD.GLM_CR * speed * dt / ref.dx)
            for _ in 1:3
                MHD.step!(ref, dt; ch=speed, integrator=:ref)
                MHD.step_cube!(cube, dt; ch=speed, decay, impl=:ka)
                step_cube_rect_cpu!(rect, dt, speed, decay)
                MHD.step_ref_drag!(refd, dt; ch=speed, decay, drag_impulse=0.05)
                MHD.step_cube_drag!(cubed, dt; ch=speed, decay, drag_impulse=0.05)
                step_cube_rect_cpu!(rectd, dt, speed, decay; drag=0.05)
            end
            host_ref = MHD.fields_to_host(ref)
            host_cube = MHD.fields_to_host(cube)
            assert_fields(host_ref, host_cube)
            assert_fields(host_ref, MHD.fields_to_host(rect))
            assert_fields(MHD.fields_to_host(refd), MHD.fields_to_host(cubed))
            assert_fields(MHD.fields_to_host(refd), MHD.fields_to_host(rectd))
            @test abs(sum(Float64, host_cube[1]) / mass0 - 1) < 3e-6
            @test maximum(abs, host_cube[2] .- fields[2]) > 1f-7   # the state evolved
        end
    end
    @testset "cubic N % 4 == 0 is required" begin
        s = MHD.allocate_state(MHD.backend(:cpu), Float32, (6,6,6); dx=1/6, gamma=5/3)
        @test_throws ErrorException MHD.step_cube!(s, 1f-3; ch=1f0, decay=1f0)
    end
end

# Strong rarefaction (tools/ct_fallback_equivalence): the PLM/HLL trial step is rejected and the
# radiation-coupled cube step falls back. The counters equal the Metal ones recorded by
# tools/ct_fallback_equivalence/bitwise_hash.jl (GLM cube A, B2, B3, B4).
@testset "GLM cube radiation positivity fallback on the CPU backend" begin
    H = FallbackHarness
    gate = 1f-4   # max(smallr, llf_dmin) of the default state
    function run(u, p0, cfl, ns, bscale, integrator; kw...)
        s0 = H.make_state(:glm, :cpu; u, p0, bscale)
        sm = Float32(MHD.max_wavespeed!(MHD.allocate_radiation_workspace(s0).fft_real, s0))
        H.advance(H.make_state(:glm, :cpu; u, p0, bscale), integrator;
                  nsteps=ns, dt=cfl * s0.dx / sm, ch=2sm, kw...)
    end
    @testset "benign step: no fallback, cube agrees with ref" begin
        c = run(5f0, 1f-3, 0.4f0, 8, 1f0, :cube)
        r = run(5f0, 1f-3, 0.4f0, 8, 1f0, :ref)
        @test c.err === nothing && c.tm.fallback_steps == 0 && c.tm.subdivision_steps == 0
        hc = MHD.fields_to_host(c.s); hr = MHD.fields_to_host(r.s)
        assert_fields(hr, hc; rtol=2e-4, atol=1f-5)
    end
    @testset "problem A: subdivision; unprotected :ref reaches the density floor" begin
        c = run(10f0, 1f-3, 1.5f0, 1, 1f0, :cube)
        r = run(10f0, 1f-3, 1.5f0, 1, 1f0, :ref)
        hc = MHD.fields_to_host(c.s)
        @test c.err === nothing && all(x -> all(isfinite, x), hc) && minimum(hc[1]) >= gate
        @test c.tm.subdivision_steps == 1 && c.tm.plm_hll_fallback_steps == 0
        @test minimum(MHD.fields_to_host(r.s)[1]) < gate
    end
    @testset "problem B3: robust minmod-PLM/HLL retry accepted" begin
        c = run(10f0, 1f-3, 0.8f0, 8, 0.1f0, :cube)
        hc = MHD.fields_to_host(c.s)
        @test c.err === nothing && all(x -> all(isfinite, x), hc) && minimum(hc[1]) >= gate
        @test H.branch(c) == "robust-PLM/HLL"
        @test c.tm.subdivision_steps == 0 && c.tm.plm_hll_fallback_steps == 6
    end
    @testset "problem B4: mixed retry and subdivision" begin
        c = run(10f0, 1f-3, 1.0f0, 8, 0.1f0, :cube)
        hc = MHD.fields_to_host(c.s)
        @test c.err === nothing && all(x -> all(isfinite, x), hc) && minimum(hc[1]) >= gate
        @test c.tm.plm_hll_fallback_steps == 4 && c.tm.subdivision_steps == 2
    end
    @testset "checkpoint fallback requested: error leaves the state untouched" begin
        c = run(10f0, 1f-3, 1.5f0, 1, 1f0, :cube; checkpoint_fallback_on_rejection=true)
        @test c.err isa ErrorException
        @test occursin("RADIATION_CHECKPOINT_CFL_FALLBACK", sprint(showerror, c.err))
    end
end

end # module
