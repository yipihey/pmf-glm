# CPU: GLM :ref (no fallback exists on :ref) vs CT :ref on the fallback-triggering problems.
using MHDKernels
include(joinpath(@__DIR__, "lib.jl"))
bn = :cpu
smax0 = 10.514f0   # max_wavespeed of the u=10 initial state; ch = 2*smax0 (runner default glm_ch_fac=2)
for (name, u, p0, cfl, ns) in (("A single step cfl=1.5", 10f0, 1f-3, 1.5f0, 1),
                                ("B2 8 steps cfl=0.9",   5f0, 1f-4, 0.9f0, 8),
                                ("B1 8 steps cfl=0.4",  10f0, 1f-3, 0.4f0, 8))
    s0 = make_state(:glm, bn; u=u, p0=p0)
    sm = Float32(max_wavespeed!(allocate_radiation_workspace(s0).fft_real, s0))
    dt = cfl*s0.dx/sm; ch = 2*sm
    @printf("== problem %s: smax0=%.4f dt=%.4e ch=%.4f\n", name, sm, dt, ch)
    report("GLM :ref", advance(make_state(:glm, bn; u=u, p0=p0), :ref; nsteps=ns, dt=dt, ch=ch))
    report("CT  :ref", advance(make_state(:ct,  bn; u=u, p0=p0), :ref; nsteps=ns, dt=dt, ch=ch))
end
