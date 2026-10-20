#!/usr/bin/env julia
using Pkg
root = dirname(@__DIR__)
backend = "cpu"
for arg in ARGS
    startswith(arg,"--backend=") || error("usage: julia scripts/bootstrap.jl --backend=cpu|metal|cuda")
    global backend = split(arg,'=';limit=2)[2]
end
backend in ("cpu","metal","cuda") || error("unsupported backend $backend")
VERSION >= v"1.12" || error("PMFGLM uses active-root [sources]; install Julia 1.12 or newer")
project = backend == "cpu" ? root : joinpath(root,"environments",backend)
Pkg.activate(project)
Pkg.instantiate()
Pkg.precompile()
println("Ready: julia --project=$project --threads=auto scripts/run_pmf.jl examples/smoke.toml --backend=$backend --output=results/first-$backend")
