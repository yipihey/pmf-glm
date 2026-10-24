# Prints a hash of the complete final state (cells, and CT faces + remainders) for several runs, so that two
# source trees can be compared bit for bit: julia --project=<tree>[/environments/metal] bitwise_hash.jl cpu|metal
bname = Symbol(ARGS[1]); bname === :metal && @eval import Metal
using MHDKernels
include(joinpath(@__DIR__, "lib.jl"))
function fullhash(s)
    h = hash(0x1)
    for x in fields_to_host(s); h = hash(Array(x), h); end
    if uses_ct(s)
        for f in (:bxf, :byf, :bzf, :rbx, :rby, :rbz); h = hash(Array(getfield(s.ct, f)), h); end
    end
    h
end
function case(label, solver, integ, u, p0, cfl, ns, bscale; kw...)
    s0 = make_state(:glm, bname; u=u, p0=p0, bscale=bscale)
    sm = Float32(max_wavespeed!(allocate_radiation_workspace(s0).fft_real, s0))
    r = advance(make_state(solver, bname; u=u, p0=p0, bscale=bscale), integ; nsteps=ns, dt=cfl*s0.dx/sm, ch=2sm, kw...)
    @printf("%-34s %s  fallback=%d robust=%d subdiv=%d\n", label, string(fullhash(r.s), base=16), r.tm.fallback_steps, r.tm.plm_hll_fallback_steps, r.tm.subdivision_steps)
end
if bname === :cpu
    case("GLM ref benign", :glm, :ref, 5f0, 1f-3, 0.3f0, 4, 1f0)
    case("GLM ref A", :glm, :ref, 10f0, 1f-3, 1.5f0, 1, 1f0)
    case("GLM ref B3", :glm, :ref, 10f0, 1f-3, 0.8f0, 8, 0.1f0)
    case("CT ref benign", :ct, :ref, 5f0, 1f-3, 0.3f0, 4, 1f0)
    case("CT ref benign, fallback off", :ct, :ref, 5f0, 1f-3, 0.3f0, 4, 1f0; positivity_fallback=false)
else
    case("GLM cube benign", :glm, :cube, 5f0, 1f-3, 0.4f0, 8, 1f0)
    case("GLM cube A (subdivision)", :glm, :cube, 10f0, 1f-3, 1.5f0, 1, 1f0)
    case("GLM cube B1", :glm, :cube, 10f0, 1f-3, 0.4f0, 8, 1f0)
    case("GLM cube B2", :glm, :cube, 5f0, 1f-4, 0.9f0, 8, 1f0)
    case("GLM cube B3 (robust)", :glm, :cube, 10f0, 1f-3, 0.8f0, 8, 0.1f0)
    case("GLM cube B4 (mixed)", :glm, :cube, 10f0, 1f-3, 1.0f0, 8, 0.1f0)
    case("GLM cube A, checkpoint err", :glm, :cube, 10f0, 1f-3, 1.5f0, 1, 1f0; checkpoint_fallback_on_rejection=true)
    case("CT ref benign (Metal)", :ct, :ref, 5f0, 1f-3, 0.3f0, 4, 1f0)
end
