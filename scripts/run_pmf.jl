#!/usr/bin/env julia
using PMFGLM

function cli(args)
    if isempty(args) || "--help" in args
        println("Usage: julia --project[=environments/metal] -t auto scripts/run_pmf.jl CONFIG.toml --output=RUN_DIR [--backend=cpu|metal|cuda] [--dry-run]")
        return
    end
    config = load_config(first(args))
    output = nothing
    dry = false
    for arg in args[2:end]
        if startswith(arg,"--output=")
            output = split(arg,'=';limit=2)[2]
        elseif startswith(arg,"--backend=")
            get!(config.settings,"run",Dict{String,Any}())["backend"] = split(arg,'=';limit=2)[2]
        elseif arg == "--dry-run"
            dry = true
        else
            error("unknown argument: $arg")
        end
    end
    output === nothing && error("--output=RUN_DIR is required")
    result = run_simulation(config;output_dir=output,dry_run=dry)
    if dry
        import_print = PMFGLM.TOML.print
        import_print(stdout,result;sorted=true)
    else
        println(result)
    end
end
cli(ARGS)
