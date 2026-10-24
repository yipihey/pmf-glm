# The terminal pressure-implicit step must not depend on thread scheduling.  The centred
# finalize kernel reads rho_new, the force fields and B at neighbouring cells; before the
# buffer-plan fix its outputs aliased rho_new/force (a cross-work-item data race), so results
# changed with the number of threads and between repeated runs.  Each case runs in a fresh
# process and compares a hash of all nine conserved arrays across thread counts.
using Test
using FFTW
using PoissonKernels

@testset "CPU rfft_exponential_source! matches the spectral reference" begin
    N = 8
    st = [sin(2pi * (i + 2j) / N) + 0.1cos(2pi * k / N) + 1 for i in 1:N, j in 1:N, k in 1:N]
    so = [cos(2pi * (j + 3k) / N) + 0.3sin(2pi * i / N) for i in 1:N, j in 1:N, k in 1:N]
    dst = similar(st)
    coeff, dt = 0.7, 0.3
    PoissonKernels.rfft_exponential_source!(dst, st, so; coeff_cells2=coeff, dt=dt)
    h = rfft(st); sh = rfft(so); out = similar(h)
    for kk in CartesianIndices(h)
        i, j, l = Tuple(kk) .- 1
        j = j <= N ÷ 2 ? j : j - N; l = l <= N ÷ 2 ? l : l - N
        x = coeff * (sin(2pi * i / N)^2 + sin(2pi * j / N)^2 + sin(2pi * l / N)^2)
        out[kk] = exp(-x) * h[kk] + (x > 1e-6 ? dt * (1 - exp(-x)) / x : dt) * sh[kk]
    end
    @test maximum(abs, dst .- irfft(out, N)) < 1e-12
    PoissonKernels.rfft_exponential_source!(dst, st, so; coeff_cells2=0.0, dt=dt)
    @test maximum(abs, dst .- (st .+ dt .* so)) < 1e-12      # coeff = 0: forward Euler of the source
    PoissonKernels.rfft_exponential_source!(dst, st, so; coeff_cells2=coeff, dt=0.0)
    @test dst == st
end

@testset "terminal pressure-implicit step is thread-count independent" begin
    root = dirname(Base.active_project())
    harness = joinpath(@__DIR__, "terminal_pressure_harness.jl")
    function result(mode, threads, lim="Inf")
        cmd = `$(Base.julia_cmd()) --project=$root -t $threads $harness $mode $lim`
        line = filter(l -> startswith(l, "RESULT"), readlines(cmd))
        @test length(line) == 1
        return match(r"hash=(\w+)", only(line)).captures[1]
    end
    for (mode, lim) in (("centered", "Inf"), ("centered", "0.05"), ("ssprk2", "Inf"), ("ct", "Inf"))
        ref = result(mode, 1, lim)
        for threads in (3, 8)
            @test result(mode, threads, lim) == ref
        end
    end
end
