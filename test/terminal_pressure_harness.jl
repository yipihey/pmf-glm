# Child-process harness for test_terminal_pressure_determinism.jl: one terminal pressure-implicit
# step (CPU, Float64, 16^3) on a fixed smooth-plus-noise state, printing a hash of all nine
# conserved arrays.  The production runner is `include`d unmodified (it is a script; main() is not
# run).  usage: julia --project=. -t K terminal_pressure_harness.jl <centered|ssprk2|ct> [rel_limit]
ENV["MHD_BACKEND"] = "cpu"; ENV["MHD_PRECISION"] = "f64"
module HarnessRunner
include(joinpath(@__DIR__, "..", "src", "production", "runner.jl"))
end
using Random, SHA
const R = HarnessRunner
mode = Symbol(ARGS[1]); lim = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : Inf
N = 16
be = R.MHDKernels.backend(:cpu)
s = R.MHDKernels.allocate_state(be, Float64, (N, N, N); dx=1.0 / N, gamma=5 / 3,
                                mhd_solver = mode === :ct ? :ct : :glm)
rng = MersenneTwister(7)
for c in 1:N^3
    i = (c - 1) % N; j = ((c - 1) ÷ N) % N; k = (c - 1) ÷ N^2
    x = 2pi * i / N; y = 2pi * j / N; z = 2pi * k / N
    rho = 1 + 0.3sin(x) * cos(y) + 0.2cos(z + x) + 0.05randn(rng)
    s.U[1][c] = rho
    s.U[2][c] = 0.05sin(y + z) + 0.01randn(rng); s.U[3][c] = 0.05cos(z + x); s.U[4][c] = 0.05sin(x - y)
    s.U[6][c] = 0.3 + 0.2sin(y); s.U[7][c] = 0.2cos(z); s.U[8][c] = 0.2sin(x)
    s.U[5][c] = 1.0 + 0.5 * (sum(abs2, (s.U[2][c], s.U[3][c], s.U[4][c])) / rho +
                             sum(abs2, (s.U[6][c], s.U[7][c], s.U[8][c])))
    s.U[9][c] = 0.0
end
mode === :ct && R.MHDKernels.ct_faces_from_cells!(s)
R._terminal_pressure_implicit_step!(be, s, 0.2, 3.0, 5.0, 0.4, lim, lim;
                                    induction=mode, project_b=true)
h = bytes2hex(sha256(reduce(vcat, [reinterpret(UInt8, copy(a)) for a in s.U])))[1:16]
println("RESULT mode=$mode threads=$(Threads.nthreads()) hash=$h sumrho=$(sum(s.U[1])) sumE=$(sum(s.U[5]))")
