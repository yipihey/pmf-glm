"SHA256 of all implementation sources and dependency declarations, independent of checkout path."
function source_fingerprint(root::AbstractString=PACKAGE_ROOT)
    files = String[]
    for folder in ("src", "lib", "scripts")
        for (directory, _, names) in walkdir(joinpath(root, folder))
            for name in names
                (endswith(name,".jl") || name == "Project.toml") &&
                    push!(files, relpath(joinpath(directory,name),root))
            end
        end
    end
    push!(files,"Project.toml")
    io = IOBuffer()
    for name in sort(files)
        print(io,name,'\0',bytes2hex(sha256(read(joinpath(root,name)))),'\n')
    end
    bytes2hex(sha256(take!(io)))
end

function _run_metadata(config, env; status="RUNNING")
    project = Base.active_project()
    manifest = project === nothing ? "" : joinpath(dirname(project),"Manifest.toml")
    solver = lowercase(get(env,"MHD_SOLVER","glm"))
    ctnote = solver == "ct" ? Dict{String,Any}(
        "ct_step_glm_limited"=>false,"ct_glm_project_every"=>0,
        "ct_note"=>"CT dt is the plain CFL dt (not shrunk by ch=glm_ch_fac*smax) and B is never " *
                   "projected; GLM radiation runs project every cycle. Expected difference at equal CFL.") :
        Dict{String,Any}()
    merge(Dict("schema_version"=>1,"status"=>status,"mhd_solver"=>solver,"created_utc"=>string(now(UTC)),
        "machine"=>gethostname(),"kernel"=>string(Sys.KERNEL),"architecture"=>string(Sys.ARCH),
        "julia_version"=>string(VERSION),"julia_threads"=>Threads.nthreads(),
        "source_fingerprint_sha256"=>source_fingerprint(),
        "active_project"=>something(project,""),
        "manifest_sha256"=>isfile(manifest) ? bytes2hex(sha256(read(manifest))) : "unresolved",
        "configuration"=>config.settings,"environment"=>env),ctnote)
end

function _completion_status(log::AbstractString,env)
    haskey(env,"MHD_ZSTART") || return "COMPLETED"
    matches = collect(eachmatch(r"COSMO final_z=([+\-0-9.eE]+)",log))
    isempty(matches) && return "INCOMPLETE"
    z = parse(Float64,last(matches).captures[1])
    target = parse(Float64,get(env,"MHD_ZEND","20"))
    z <= target + 1e-6*max(1,target) ? "COMPLETED" : "STOPPED_AT_LIMIT"
end

"CT_INITIAL_FACES numbers from the first run segment of `run.log` (empty if absent)."
function _ct_initial_faces(log::AbstractString)
    m = match(r"CT_INITIAL_FACES cell_b_rel_l2_change=([^ ]+) magnetic_energy_ratio=([^ ]+) divb_face_dx_over_brms=([^ ]+)",log)
    m === nothing && return Dict{String,Any}()
    Dict{String,Any}("ct_initial_cell_b_rel_l2_change"=>parse(Float64,m.captures[1]),
        "ct_initial_magnetic_energy_ratio"=>parse(Float64,m.captures[2]),
        "ct_initial_divb_face_dx_over_brms"=>parse(Float64,m.captures[3]))
end

const RESTART_CONTROL_OPTIONS = Set([
    "MHD_ZEND", "MHD_STEPS", "MHD_MAX_CYCLES", "MHD_CHECKPOINT_RESUME",
    "MHD_CHECKPOINT_EVERY", "MHD_CHECKPOINT_FINAL", "MHD_PRINT_EVERY",
    "MHD_CYCLE_PRINT_EVERY", "MHD_CHECK_EVERY", "MHD_PK_ZS", "MHD_PK_EVERY",
])

function _validate_restart_environment(previous,current)
    changed = sort!([String(key) for key in union(keys(previous),keys(current))
        if !(key in RESTART_CONTROL_OPTIONS) &&
           get(previous,key,nothing) != get(current,key,nothing)])
    isempty(changed) || error("restart changes evolution options: " * join(changed,", ") *
        "; use the original physical settings or start a new run")
    nothing
end

"""
    run_simulation(config; output_dir, dry_run=false)

Run the full production model in a fresh Julia process using the current active
environment. A fresh process isolates the legacy driver's compile-time options
and device caches. `metadata.toml` and `resolved-environment.toml` record inputs;
`run.log` is flushed by the production driver. Existing runs require explicit
`run.resume=true` and a checkpoint. Run directories are never silently replaced.
"""
function run_simulation(c::RunConfig; output_dir::AbstractString, dry_run::Bool=false)
    env = run_environment(c; output_dir)
    dry_run && return env
    out = abspath(output_dir)
    existing = isfile(joinpath(out,"metadata.toml"))
    resuming = get(env,"MHD_CHECKPOINT_RESUME","false") in ("true","1")
    existing && !resuming && error("output already contains a run; choose a new directory or resume=true")
    resuming && !isfile(env["MHD_CHECKPOINT_PATH"]) && error("resume requires checkpoint.jls")
    mkpath(out)
    if existing
        previous = TOML.parsefile(joinpath(out,"metadata.toml"))
        previous["source_fingerprint_sha256"] == source_fingerprint() ||
            error("checkpoint source fingerprint differs; migration requires explicit review")
        _validate_restart_environment(previous["environment"],env)
    end
    metadata = _run_metadata(c,env)
    function write_toml(path,payload)
        open(io->TOML.print(io,payload;sorted=true),path*".tmp","w")
        mv(path*".tmp",path;force=true)
    end
    write_toml(joinpath(out,"metadata.toml"),metadata)
    write_toml(joinpath(out,"resolved-environment.toml"),env)
    active = Base.active_project()
    if active !== nothing
        cp(active,joinpath(out,"Project.toml");force=true)
        manifest = joinpath(dirname(active),"Manifest.toml")
        isfile(manifest) && cp(manifest,joinpath(out,"Manifest.toml");force=true)
    end
    # Prevent unrelated shell MHD options from silently modifying the science case.
    child_env = Dict(k=>v for (k,v) in ENV if !startswith(k,"MHD_"))
    merge!(child_env,env)
    project = dirname(something(Base.active_project(),joinpath(PACKAGE_ROOT,"Project.toml")))
    script = joinpath(PACKAGE_ROOT,"scripts","_production.jl")
    cmd = setenv(`$(Base.julia_cmd()) --project=$project --threads=$(Threads.nthreads()) $script`,child_env)
    println("PMFGLM run: ",out,"\nLog: ",joinpath(out,"run.log")); flush(stdout)
    t0 = time()
    try
        open(joinpath(out,"run.log"),resuming ? "a" : "w") do io
            process = run(pipeline(ignorestatus(cmd);stdout=io,stderr=io))
            success(process) || error("production process exited with status $(process.exitcode); see $(joinpath(out,"run.log"))")
        end
        log = read(joinpath(out,"run.log"),String)
        metadata["status"] = _completion_status(log,env)
        metadata["mhd_solver"] == "ct" && merge!(metadata,_ct_initial_faces(log))
    catch error
        metadata["status"] = "FAILED"
        metadata["error"] = sprint(showerror,error)
        rethrow()
    finally
        metadata["wall_seconds"] = time()-t0
        metadata["finished_utc"] = string(now(UTC))
        write_toml(joinpath(out,"metadata.toml"),metadata)
    end
    (; output_dir=out, status=metadata["status"], wall_seconds=metadata["wall_seconds"])
end
