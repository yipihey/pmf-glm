# Runner-level checks of the selectable constrained-transport (CT) solver. Each case launches
# the production driver in a fresh process on the 16^3 cosmology smoke problem (CPU, Float32).
using TOML

const CT_SMOKE = TOML.parsefile(joinpath(@__DIR__, "..", "examples", "cosmology-smoke.toml"))

function ct_smoke_config(solver; zend=3199.9, resume=false, final_checkpoint=true, max_cycles=32,
                         compensate=false)
    settings = deepcopy(CT_SMOKE)
    compensate && (settings["environment"] = Dict("MHD_RADIATION_MAGNETIC_COMPENSATION" => "1"))
    settings["run"]["max_cycles"] = max_cycles
    settings["run"]["mhd_solver"] = solver
    settings["run"]["checkpoint_final"] = final_checkpoint
    resume && (settings["run"]["resume"] = true)
    settings["cosmology"]["zend"] = zend
    RunConfig(settings)
end

function ct_last_row(dir)
    lines = readlines(joinpath(dir, "history.tsv"))
    header = split(lines[1], '\t')
    Dict(zip(header, split(lines[end], '\t')))
end

function ct_perf(dir)
    lines = readlines(joinpath(dir, "performance.tsv"))
    Dict(zip(split(lines[1], '\t'), split(lines[end], '\t')))
end

mktempdir() do root
    out(name) = joinpath(root, name)
    quiet(f) = redirect_stdout(f, devnull)

    # Uninterrupted CT run to the final redshift.
    full = quiet(() -> run_simulation(ct_smoke_config("ct"; final_checkpoint=false);
                                      output_dir=out("ct-full")))
    @test full.status == "COMPLETED"
    meta = TOML.parsefile(joinpath(out("ct-full"), "metadata.toml"))
    @test meta["mhd_solver"] == "ct"
    @test meta["environment"]["MHD_SOLVER"] == "ct"
    @test meta["ct_step_glm_limited"] == false
    @test haskey(meta, "ct_initial_divb_face_dx_over_brms")
    log = read(joinpath(out("ct-full"), "run.log"), String)
    @test occursin("CT_INITIAL_FACES cell_b_rel_l2_change=", log)
    @test occursin("CONFIG mhd_solver=ct", log)
    perf = ct_perf(out("ct-full"))
    @test perf["mhd_solver"] == "ct"
    @test parse(Float64, perf["divb_face_dx_over_brms"]) < 1e-5     # Float32 round-off
    @test parse(Float64, ct_last_row(out("ct-full"))["divb_face_dx_over_brms"]) < 1e-5
    @test parse(Float64, perf["ct_initial_divb_face_dx_over_brms"]) < 1e-5

    # CT checkpoint round trip: stop after 4 cycles (same step sequence as the full run),
    # resume to the final redshift, compare history rows with the uninterrupted run. Faces are
    # restored exactly, so B, divergence, pressure, temperature and abundances are bitwise
    # identical. The gas/DM density-contrast columns are excluded: after a resume they differ
    # at the 1e-9 level for GLM runs too (pre-existing, not specific to CT).
    half = quiet(() -> run_simulation(ct_smoke_config("ct"; max_cycles=4); output_dir=out("ct-half")))
    @test half.status == "STOPPED_AT_LIMIT"
    @test isfile(joinpath(out("ct-half"), "checkpoint.jls"))
    resumed = quiet(() -> run_simulation(ct_smoke_config("ct"; resume=true); output_dir=out("ct-half")))
    @test resumed.status == "COMPLETED"
    a = ct_last_row(out("ct-full")); b = ct_last_row(out("ct-half"))
    for key in keys(a)
        key in ("delta_b_rms", "delta_dm_rms", "delta2k_amp") && continue
        @test a[key] == b[key]
    end

    # Compensated CT faces (MHD_RADIATION_MAGNETIC_COMPENSATION=1): the face remainders are
    # checkpointed (ct_residual_*), so stop/resume reproduces the uninterrupted run bitwise.
    cfull = quiet(() -> run_simulation(ct_smoke_config("ct"; compensate=true, final_checkpoint=false);
                                       output_dir=out("ctc-full")))
    @test cfull.status == "COMPLETED"
    chalf = quiet(() -> run_simulation(ct_smoke_config("ct"; compensate=true, max_cycles=4);
                                       output_dir=out("ctc-half")))
    @test chalf.status == "STOPPED_AT_LIMIT"
    cresumed = quiet(() -> run_simulation(ct_smoke_config("ct"; compensate=true, resume=true);
                                          output_dir=out("ctc-half")))
    @test cresumed.status == "COMPLETED"
    a = ct_last_row(out("ctc-full")); b = ct_last_row(out("ctc-half"))
    for key in keys(a)
        key in ("delta_b_rms", "delta_dm_rms", "delta2k_amp") && continue
        @test a[key] == b[key]
    end
    @test parse(Float64, a["divb_face_dx_over_brms"]) < 1e-5

    # GLM: metadata and an empty face-divergence column; GLM and CT checkpoints do not mix.
    glm = quiet(() -> run_simulation(ct_smoke_config("glm"; max_cycles=4); output_dir=out("glm-half")))
    @test glm.status == "STOPPED_AT_LIMIT"
    @test TOML.parsefile(joinpath(out("glm-half"), "metadata.toml"))["mhd_solver"] == "glm"
    @test ct_perf(out("glm-half"))["mhd_solver"] == "glm"
    @test isnan(parse(Float64, ct_last_row(out("glm-half"))["divb_face_dx_over_brms"]))
    @test_throws ErrorException quiet(() -> run_simulation(
        ct_smoke_config("ct"; resume=true); output_dir=out("glm-half")))
    @test_throws ErrorException quiet(() -> run_simulation(
        ct_smoke_config("glm"; resume=true); output_dir=out("ct-half")))
end

# The production driver repeats the configuration-time rejections itself (a TOML [environment]
# table can be bypassed by launching scripts/_production.jl directly). Launch the rejected
# combinations in parallel and check the driver's own error text; none may reach allocation.
let
    config = ct_smoke_config("ct"; final_checkpoint=false)
    base = run_environment(config; output_dir=mktempdir())
    script = joinpath(@__DIR__, "..", "scripts", "_production.jl")
    project = dirname(Base.active_project())
    cases = [(Dict("MHD_DRAG_DT_MODE" => "terminal", "MHD_TERMINAL_INDUCTION" => "ssprk2"),
              "does not support MHD_TERMINAL_INDUCTION=ssprk2"),
             (Dict("MHD_DRAG_DT_MODE" => "hybrid", "MHD_TERMINAL_PRESSURE_IMPLICIT" => "1"),
              "requires MHD_TERMINAL_INDUCTION=ct"),
             (Dict("MHD_INTEGRATOR" => "cube"), "requires MHD_INTEGRATOR=ref"),
             (Dict("MHD_GLM_PROJECT_EVERY" => "1"), "does not use GLM B projection")]
    jobs = map(cases) do (extra, _)
        env = Dict(k => v for (k, v) in ENV if !startswith(k, "MHD_"))
        merge!(env, base, extra, Dict("MHD_ALLOW_UNCERTIFIED_TERMINAL" => "1"))
        log = tempname()
        process = run(pipeline(setenv(`$(Base.julia_cmd()) --project=$project $script`, env);
                               stdout=log, stderr=log, ); wait=false)
        (process, log)
    end
    for ((process, log), (_, expected)) in zip(jobs, cases)
        wait(process)
        text = read(log, String)
        @test !success(process)
        @test occursin(expected, text)
        @test !occursin("START allocate_mhd_state", text)
    end
end
