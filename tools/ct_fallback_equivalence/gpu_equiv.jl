# Equivalence of CT :ref (with the new positivity fallback) and GLM :cube (existing fallback), same backend.
# Run: julia --project=environments/metal -t 4 tools/ct_fallback_equivalence/gpu_equiv.jl metal
#
# TOLERANCE, DECLARED BEFORE THE FALLBACK RUNS (this header is committed before the run):
#   d0_X = relative-L1 difference of field X in {rho,|mom|,E,|B|} between GLM :cube and CT :ref on the
#          BENIGN schedule (same problem, cfl 0.4, no fallback triggered) -- the intrinsic scheme-to-scheme
#          difference (cleaned-divergence GLM vs face CT, different integrators, Float32).
#   tol_X = max(3*d0_X, 2e-3).  The fallback problems must satisfy rel-L1_X(GLM,CT) <= tol_X, they must
#          take the SAME branch (robust-PLM/HLL vs subdivision) with the same counters, and CT face
#          divergence must stay <= 1e-5 * Brms/dx-normalised (divB_face*dx/Brms < 1e-5).
#   Rationale: after a fallback both solvers apply the identical robust minmod-PLM/HLL conservative
#   step, so the GLM-vs-CT difference stays the B-treatment difference, not a new error; factor 3 and
#   the 2e-3 floor leave room for the fallback's extra dissipation amplifying that difference.
bname = Symbol(ARGS[1])
bname === :metal && @eval import Metal
using MHDKernels
include(joinpath(@__DIR__, "lib.jl"))
glm_int = :cube
function setup(u, p0, cfl)
    s0 = make_state(:glm, bname; u=u, p0=p0)
    sm = Float32(max_wavespeed!(allocate_radiation_workspace(s0).fft_real, s0))
    (dt = cfl*s0.dx/sm, ch = 2*sm)
end
function pair(label, u, p0, cfl, ns; kw...)
    (dt, ch) = setup(u, p0, cfl)
    @printf("== %s: u=%g p0=%g cfl=%g steps=%d dt=%.4e ch=%.4f\n", label, u, p0, cfl, ns, dt, ch)
    g = advance(make_state(:glm, bname; u=u, p0=p0), glm_int; nsteps=ns, dt=dt, ch=ch, kw...)
    c = advance(make_state(:ct,  bname; u=u, p0=p0), :ref;    nsteps=ns, dt=dt, ch=ch, kw...)
    report("GLM :cube", g); report("CT  :ref", c)
    g, c
end
# calibration: benign
g0, c0 = pair("benign (calibration)", 5f0, 1f-3, 0.4f0, 8)
d0 = rel_diffs(c0.s, g0.s)
tol = map(x -> max(3x, 2e-3), d0)
@printf("d0  : rho=%.3e mom=%.3e E=%.3e B=%.3e\n", d0...)
@printf("tol : rho=%.3e mom=%.3e E=%.3e B=%.3e\n", tol...)
function verdict(label, g, c)
    d = rel_diffs(c.s, g.s)
    ok_tol = all(values(d) .<= values(tol))
    same_branch = branch(g) == branch(c) && g.tm.plm_hll_fallback_steps == c.tm.plm_hll_fallback_steps &&
                  g.tm.subdivision_steps == c.tm.subdivision_steps && g.tm.fallback_steps == c.tm.fallback_steps
    Bh = fields_to_host(c.s); brms = sqrt(sum(abs2, Float64.(Bh[6]))/length(Bh[6]))
    div = c.divmax / brms
    @printf("%s: rel-L1 GLM-vs-CT rho=%.3e mom=%.3e E=%.3e B=%.3e  within_tol=%s  same_branch=%s (GLM %s fb=%d sub=%d | CT %s fb=%d sub=%d)  CT divB*dx/Brms=%.3e (<1e-5: %s)\n",
        label, d..., ok_tol, same_branch, branch(g), g.tm.plm_hll_fallback_steps, g.tm.subdivision_steps,
        branch(c), c.tm.plm_hll_fallback_steps, c.tm.subdivision_steps, div, div < 1e-5)
end
verdict("benign", g0, c0)
ga, ca = pair("A subdivision", 10f0, 1f-3, 1.5f0, 1);  verdict("A", ga, ca)
gb, cb = pair("B1 multi-step", 10f0, 1f-3, 0.4f0, 8); verdict("B1", gb, cb)
gb2, cb2 = pair("B2 multi-step", 5f0, 1f-4, 0.9f0, 8); verdict("B2", gb2, cb2)
# C: checkpoint-fallback error path (problem A): both must throw, and leave the state at its pre-step value.
(dt, ch) = setup(10f0, 1f-3, 1.5f0)
for (nm, solver, integ) in (("GLM :cube", :glm, glm_int), ("CT  :ref", :ct, :ref))
    s = make_state(solver, bname; u=10f0, p0=1f-3)
    before = fields_to_host(s)
    fb = solver === :ct ? (Array(s.ct.bxf), Array(s.ct.byf), Array(s.ct.bzf)) : nothing
    r = advance(s, integ; nsteps=1, dt=dt, ch=ch, checkpoint_fallback_on_rejection=true)
    after = fields_to_host(r.s)
    same = all(after[v] == before[v] for v in 1:9)
    facesame = fb === nothing ? "n/a" : string(all(Array(getfield(r.s.ct, f)) == fb[q] for (q, f) in enumerate((:bxf, :byf, :bzf))))
    msg = r.err === nothing ? "none" : first(split(sprint(showerror, r.err), '\n'))
    @printf("C %s: error=%s\n    cells_restored_bitwise=%s faces_restored_bitwise=%s\n", nm,
            occursin("RADIATION_CHECKPOINT_CFL_FALLBACK", msg), same, facesame)
end
