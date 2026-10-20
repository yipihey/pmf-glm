#!/usr/bin/env julia
using PMFGLM
using TOML
using Dates

isempty(ARGS) && error("usage: julia --project[=environments/metal] -t 4 scripts/run_suite.jl SUITE.toml [--execute]")
path = abspath(first(ARGS))
suite = TOML.parsefile(path)
execute = "--execute" in ARGS
all(arg=="--execute" for arg in ARGS[2:end]) || error("unknown suite argument")
base = load_config(normpath(joinpath(dirname(path),suite["base_config"])))
outroot = normpath(joinpath(dirname(path),get(suite,"output_dir","../results/suite")))
labels = String[]
for case in suite["cases"]
    label = case["label"]
    occursin(r"^[a-zA-Z0-9_-]+$",label) || error("case label must be a simple directory name")
    label in labels && error("duplicate case label $label")
    push!(labels,label)
    settings = deepcopy(base.settings)
    for (section,values) in case
        section=="label" && continue
        values isa AbstractDict || error("suite overrides must be TOML tables")
        merge!(get!(settings,section,Dict{String,Any}()),values)
    end
    config = RunConfig(settings)
    out = joinpath(outroot,label)
    if execute
        println("Starting serialized case ",label," at ",now());flush(stdout)
        run_simulation(config;output_dir=out)
    else
        run_simulation(config;output_dir=out,dry_run=true)
        println(label,": backend=",get(settings["run"],"backend","cpu")," N=",settings["run"]["n"]," -> ",out)
    end
end
execute || println("Validated ",length(labels)," cases. Add --execute to run serially.")
