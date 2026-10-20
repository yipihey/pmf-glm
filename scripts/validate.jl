#!/usr/bin/env julia
# Standard-library Python writes machine-readable results and verifies site links.
root = dirname(@__DIR__)
project = dirname(something(Base.active_project(),joinpath(root,"Project.toml")))
python = get(ENV,"PYTHON","python3")
run(`$python $(joinpath(@__DIR__,"validate.py")) --julia=$(Base.julia_cmd().exec[1]) --project=$project --threads=$(Threads.nthreads()) $ARGS`)
