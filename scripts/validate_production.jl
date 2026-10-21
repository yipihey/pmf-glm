#!/usr/bin/env julia
using PMFGLM
using Serialization
using TOML
using Dates

name = "cpu"
for arg in ARGS
    startswith(arg,"--backend=") || error("usage: scripts/validate_production.jl --backend=cpu|metal|cuda")
    global name = split(arg,'=';limit=2)[2]
end
name in ("cpu","metal","cuda") || error("unsupported backend")
root = dirname(@__DIR__)
mkpath(joinpath(root,"results"))
out = mktempdir(joinpath(root,"results"); prefix="production-$name-",cleanup=false)

function checkpoint_arrays(path)
    open(path) do io
        magic = deserialize(io)
        magic == "MHDKernels PMF checkpoint v1" || error("unexpected checkpoint schema")
        config = deserialize(io)
        runtime = deserialize(io)
        arrays = Dict{String,Any}()
        for _ in 1:deserialize(io)
            key = deserialize(io)
            arrays[key] = deserialize(io)
        end
        (; config, runtime, arrays)
    end
end

function launch(config,path;expected="COMPLETED")
    result = run_simulation(config;output_dir=path)
    result.status == expected || error("production case has unexpected endpoint: $result")
    result
end

started = time()
smoke = load_config(joinpath(root,"examples","cosmology-smoke.toml"))
smoke.settings["run"]["backend"] = name
full = launch(smoke,joinpath(out,"cosmology"))

# Deliberately stop at a cycle budget so the three trajectories have identical
# timesteps and final scale factor. Include chemistry and persistent photons.
base = RunConfig(deepcopy(smoke.settings))
base.settings["cosmology"]["zend"] = 3190.0
base.settings["cosmology"]["max_dln_a"] = 1e-4
base.settings["run"]["max_cycles"] = 4
base.settings["run"]["checkpoint_final"] = true
continuous = launch(base,joinpath(out,"continuous");expected="STOPPED_AT_LIMIT")
base.settings["run"]["max_cycles"] = 2
split = launch(base,joinpath(out,"resumed");expected="STOPPED_AT_LIMIT")
base.settings["run"]["max_cycles"] = 4
base.settings["run"]["resume"] = true
resumed = launch(base,joinpath(out,"resumed");expected="STOPPED_AT_LIMIT")
a = checkpoint_arrays(joinpath(continuous.output_dir,"checkpoint.jls"))
b = checkpoint_arrays(joinpath(resumed.output_dir,"checkpoint.jls"))
keys(a.arrays) == keys(b.arrays) || error("checkpoint field sets differ")
a.runtime.cycle == b.runtime.cycle == 4 || error("restart did not reach cycle four")
a.runtime.a == b.runtime.a || error("restart changed the cosmological clock")
errors = Dict{String,Float64}()
for (key,x) in a.arrays
    y = b.arrays[key]
    all(isfinite,x) && all(isfinite,y) || error("nonfinite state: $key")
    size(x) == size(y) || error("checkpoint field sizes differ: $key")
    wide = eltype(x) <: Complex ? ComplexF64 : Float64
    errors[key] = maximum(abs, wide.(x) .- wide.(y)) /
                  max(maximum(abs,wide.(x)),1.0)
end
# A plotting/diagnostic request must not alter the physical trajectory.
sparse_config = RunConfig(deepcopy(base.settings))
sparse_config.settings["run"]["resume"] = false
sparse_config.settings["diagnostics"]["every"] = 0
sparse_run = launch(sparse_config,joinpath(out,"sparse-diagnostics");expected="STOPPED_AT_LIMIT")
sparse_state = checkpoint_arrays(joinpath(sparse_run.output_dir,"checkpoint.jls"))
a.runtime.a == sparse_state.runtime.a || error("diagnostic cadence changed the clock")
cadence_error = maximum(keys(a.arrays)) do key
    x=a.arrays[key]; y=sparse_state.arrays[key]
    wide=eltype(x) <: Complex ? ComplexF64 : Float64
    maximum(abs,wide.(x).-wide.(y))/max(maximum(abs,wide.(x)),1.0)
end
cadence_error <= 5e-6 || error("diagnostic cadence changed evolution: $cadence_error")
worst = maximum(values(errors))
passed = worst <= 5e-6
record = Dict("schema_version"=>1,"backend"=>name,"status"=>passed ? "PASS" : "FAIL",
    "created_utc"=>string(now(UTC)),"machine"=>gethostname(),
    "source_fingerprint_sha256"=>source_fingerprint(),"julia_version"=>string(VERSION),
    "julia_threads"=>Threads.nthreads(),"wall_seconds"=>time()-started,
    "scope"=>"full-physics endpoint smoke plus four-cycle full-MHD/photon/H-He-chemistry/gravity restart and diagnostic-cadence parity",
    "cosmology_status"=>full.status,"diagnostic_cadence_max_scaled_error"=>cadence_error,"restart_fields"=>length(errors),
    "restart_max_scaled_error"=>worst,"restart_error_tolerance"=>5e-6,
    "run_directory"=>relpath(out,root),"restart_field_errors"=>errors)
destination = joinpath(root,"docs","data","production-$name.toml")
open(io->TOML.print(io,record;sorted=true),destination,"w")
TOML.print(stdout,record;sorted=true)
passed || error("production restart parity failed")
