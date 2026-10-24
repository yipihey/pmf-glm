# Diagnostic: is the large GLM-vs-CT |B| difference at B3 (cfl 0.8, fallback active) explained by
# time-step error of each solver (distance of each to its own cfl 0.2 solution at the same time)?
bname = Symbol(ARGS[1]); bname === :metal && @eval import Metal
using MHDKernels
include(joinpath(@__DIR__, "lib.jl"))
u, p0, bs = 10f0, 1f-3, 0.1f0
s0 = make_state(:glm, bname; u=u, p0=p0, bscale=bs)
sm = Float32(max_wavespeed!(allocate_radiation_workspace(s0).fft_real, s0)); ch = 2sm
run(solver, integ, cfl, ns) = advance(make_state(solver, bname; u=u, p0=p0, bscale=bs), integ; nsteps=ns, dt=cfl*s0.dx/sm, ch=ch)
g8 = run(:glm, :cube, 0.8f0, 8); g32 = run(:glm, :cube, 0.2f0, 32)
c8 = run(:ct, :ref, 0.8f0, 8);   c32 = run(:ct, :ref, 0.2f0, 32)
pr(l, a, b) = (d = rel_diffs(a.s, b.s); @printf("%-34s rho=%.3e mom=%.3e E=%.3e B=%.3e\n", l, d...))
pr("GLM cfl0.8 vs GLM cfl0.2", g8, g32)
pr("CT  cfl0.8 vs CT  cfl0.2", c8, c32)
pr("GLM cfl0.8 vs CT  cfl0.8", g8, c8)
pr("GLM cfl0.2 vs CT  cfl0.2", g32, c32)
