"Validated TOML settings for a reproducible simulation launched in a fresh Julia process."
struct RunConfig
    settings::Dict{String,Any}
    source_path::String
end

const CONFIG_KEYS = Dict(
    "run" => Dict("n"=>"MHD_PMF_N", "backend"=>"MHD_BACKEND", "precision"=>"MHD_PRECISION",
        "steps"=>"MHD_STEPS", "max_cycles"=>"MHD_MAX_CYCLES", "cfl"=>"MHD_CFL",
        "reconstruction"=>"MHD_RECON", "riemann"=>"MHD_RIEMANN",
        "print_every"=>"MHD_PRINT_EVERY", "check_every"=>"MHD_CHECK_EVERY",
        "checkpoint_every"=>"MHD_CHECKPOINT_EVERY", "checkpoint_final"=>"MHD_CHECKPOINT_FINAL",
        "resume"=>"MHD_CHECKPOINT_RESUME", "mhd_solver"=>"MHD_SOLVER"),
    "cosmology" => Dict("zstart"=>"MHD_ZSTART", "zend"=>"MHD_ZEND", "h"=>"MHD_H",
        "omega_m"=>"MHD_OM", "omega_b"=>"MHD_OB", "omega_lambda"=>"MHD_OL",
        "omega_r"=>"MHD_OR", "box_ckpc_h"=>"MHD_TERMINAL_LBOX_CKPC",
        "max_dln_a"=>"MHD_MAXEXP"),
    "streaming" => Dict("vx_kms"=>"MHD_STREAMING_VX_KMS", "vy_kms"=>"MHD_STREAMING_VY_KMS",
        "vz_kms"=>"MHD_STREAMING_VZ_KMS"),
    "magnetic" => Dict("invariant_axis"=>"MHD_PMF_INVARIANT_AXIS", "initial_condition"=>"MHD_PMF_INIT", "b0_ng"=>"MHD_PMF_B0_NG",
        "brms_code"=>"MHD_PMF_BRMS", "seed"=>"MHD_PMF_SEED", "kcut"=>"MHD_PMF_KCUT",
        "kmax"=>"MHD_PMF_KMAX", "kmin"=>"MHD_PMF_KMIN", "mode_lock_n"=>"MHD_PMF_MODE_LOCK_N",
        "curl_symbol"=>"MHD_PMF_CURL_SYMBOL", "curl_reference_n"=>"MHD_PMF_CURL_REFERENCE_N",
        "normalization"=>"MHD_PMF_NORMALIZATION_KIND", "fixed_mode_power"=>"MHD_PMF_FIXED_MODE_POWER",
        "reference_kmax"=>"MHD_PMF_PRESERVE_LOW_KMAX", "reference_kcut"=>"MHD_PMF_PRESERVE_LOW_KCUT",
        "minimum_cells_per_wavelength"=>"MHD_PMF_MIN_CELLS_PER_WAVELENGTH",
        "mode_realization"=>"MHD_PMF_MODE_REALIZATION", "realization_box"=>"MHD_PMF_REALIZATION_BOX"),
    "chemistry" => Dict("mode"=>"MHD_CHEM", "helium"=>"MHD_CHEM_HELIUM",
        "coupling"=>"MHD_CHEM_COUPLING", "expansion"=>"MHD_CHEM_EXPANSION",
        "smoothing"=>"MHD_CHEM_SMOOTH", "smoothing_sigma_cells"=>"MHD_CHEM_SMOOTH_SIGMA_CELLS",
        "dt_fraction"=>"MHD_CHEM_DTFRAC", "hydrogen_storage"=>"MHD_CHEM_H_STORAGE",
        "ionized_fast_zmin"=>"MHD_CHEM_IONIZED_FAST_ZMIN", "velocity_unit_cms"=>"MHD_CHEM_VUNIT",
        "x_hii_initial"=>"MHD_XHII_INIT", "x_h2_initial"=>"MHD_XH2_INIT",
        "nonlocal_table"=>"MHD_CHEM_NONLOCAL_TABLE"),
    "physics" => Dict("gravity"=>"MHD_GRAVITY", "gas_gravity"=>"MHD_GRAVITY_GAS_KICK",
        "gravity_drag_max_impulse"=>"MHD_GRAVITY_DRAG_MAX_IMPULSE",
        "photon_comoving_initially"=>"MHD_RADIATION_PHOTON_COMOVING",
        "photon_adiabatic_initially"=>"MHD_RADIATION_PHOTON_ADIABATIC",
        "compton_drag"=>"MHD_COMPTON_DRAG", "drag_mode"=>"MHD_DRAG_DT_MODE",
        "photon_response"=>"MHD_RADIATION_RESPONSE_MODEL", "photon_max_dln_a"=>"MHD_RADIATION_MAX_DLNA",
        "nonlinear_cfl"=>"MHD_RADIATION_NONLINEAR_CFL", "fft"=>"MHD_FFT",
        "pressure"=>"MHD_P0", "scaling"=>"MHD_COSMOLOGICAL_MHD_SCALING",
        "glm_project_every"=>"MHD_GLM_PROJECT_EVERY", "glm_ch_factor"=>"MHD_GLM_CH_FAC",
        "particle_max_displacement_cells"=>"MHD_PARTICLE_MAX_DISP_CELLS"),
    "diagnostics" => Dict("redshifts"=>"MHD_PK_ZS", "every"=>"MHD_PK_EVERY",
        "adaptive_fraction"=>"MHD_ADAPTIVE_DIAG_REL", "spectrum_bins"=>"MHD_PK_NBINS",
        "angle_bins"=>"MHD_PK_NMU", "pdf_bins"=>"MHD_PDF_NBINS",
        "energy_ledger_every"=>"MHD_RADIATION_ENERGY_LEDGER_EVERY",
        "streaming"=>"MHD_STREAMING_TSV", "frame_redshifts"=>"MHD_DENSITY_FRAME_ZS", "frame_field"=>"MHD_DENSITY_FRAME_FIELD"),
)

RunConfig(settings::AbstractDict) = validate_config(RunConfig(Dict{String,Any}(settings), ""))
load_config(path::AbstractString) = validate_config(RunConfig(TOML.parsefile(path), abspath(path)))
_setting(c, section, key, default) = get(get(c.settings, section, Dict()), key, default)
_envstr(x::Bool) = x ? "true" : "false"
_envstr(x::AbstractVector) = join(_envstr.(x), ",")
_envstr(x) = string(x)

function validate_config(c::RunConfig)
    for (section, values) in c.settings
        section in keys(CONFIG_KEYS) || section == "environment" ||
            throw(ArgumentError("unknown configuration section: $section"))
        values isa AbstractDict || throw(ArgumentError("$section must be a TOML table"))
        for (key, value) in values
            value isa Union{AbstractString,Number,Bool,AbstractVector} ||
                throw(ArgumentError("unsupported value for $section.$key"))
            if section == "environment"
                key in known_environment_keys() || throw(ArgumentError("unknown solver option $key"))
            else
                haskey(CONFIG_KEYS[section], key) || throw(ArgumentError("unknown option $section.$key"))
            end
        end
    end
    n = _setting(c, "run", "n", 16)
    n isa Integer && n >= 8 && iseven(n) || throw(ArgumentError("run.n must be an even integer >=8"))
    be = _setting(c, "run", "backend", "cpu")
    be in ("cpu", "metal", "cuda") || throw(ArgumentError("backend must be cpu, metal, or cuda"))
    precision = _setting(c, "run", "precision", "f32")
    precision in ("f32", "f64") || throw(ArgumentError("precision must be f32 or f64"))
    be != "cpu" && precision != "f32" && throw(ArgumentError("GPU production uses f32"))
    solver = _setting(c, "run", "mhd_solver", "glm")
    solver isa AbstractString && lowercase(solver) in ("glm", "ct") ||
        throw(ArgumentError("run.mhd_solver must be \"glm\" or \"ct\""))
    if lowercase(solver) == "ct"
        raw = Dict{String,String}("MHD_SOLVER"=>"ct")
        for (section, mapping) in CONFIG_KEYS, (key, name) in mapping
            section == "diagnostics" && key == "streaming" && continue
            value = _setting(c, section, key, nothing)
            value === nothing || (raw[name] = _envstr(value))
        end
        for (key, value) in get(c.settings, "environment", Dict())
            raw[key] = _envstr(value)
        end
        raw["MHD_SOLVER"] = "ct"
        _validate_ct_environment(raw)
    end
    cfl = _setting(c, "run", "cfl", 0.2)
    cfl isa Real && isfinite(cfl) && 0 < cfl <= 0.8 || throw(ArgumentError("CFL must be in (0,0.8]"))
    for key in ("steps", "max_cycles")
        value = _setting(c, "run", key, 1)
        value isa Integer && value > 0 || throw(ArgumentError("run.$key must be positive"))
    end
    zs = _setting(c, "cosmology", "zstart", nothing)
    ze = _setting(c, "cosmology", "zend", nothing)
    if zs !== nothing || ze !== nothing
        zs isa Real && ze isa Real && isfinite(zs) && isfinite(ze) && zs > ze >= 0 ||
            throw(ArgumentError("cosmology requires finite zstart > zend >= 0"))
        L = _setting(c, "cosmology", "box_ckpc_h", nothing)
        L isa Real && isfinite(L) && L > 0 || throw(ArgumentError("set a positive box_ckpc_h"))
        B = _setting(c, "magnetic", "b0_ng", nothing)
        B isa Real && isfinite(B) && B > 0 || throw(ArgumentError("set a positive comoving b0_ng"))
    end
    for key in ("vx_kms","vy_kms","vz_kms")
        value = _setting(c,"streaming",key,0)
        value isa Real && !(value isa Bool) && isfinite(value) ||
            throw(ArgumentError("streaming.$key must be finite km/s"))
    end
    invariant = _setting(c,"magnetic","invariant_axis",0)
    invariant isa Integer && !(invariant isa Bool) && invariant in 0:3 ||
        throw(ArgumentError("magnetic.invariant_axis must be 0, 1, 2, or 3"))
    _setting(c,"diagnostics","streaming",false) isa Bool ||
        throw(ArgumentError("diagnostics.streaming must be boolean"))
    _setting(c,"diagnostics","frame_field","gas") in ("gas","dm","both") ||
        throw(ArgumentError("diagnostics.frame_field must be gas, dm, or both"))
    frames = _setting(c,"diagnostics","frame_redshifts",Float64[])
    frames isa AbstractVector && all(z->z isa Real && isfinite(z) && z>=0,frames) ||
        throw(ArgumentError("diagnostics.frame_redshifts must be finite nonnegative redshifts"))
    return c
end

"""
Reject option combinations that the constrained-transport solver (`MHD_SOLVER=ct`) cannot
honour, before any state is allocated. `env` maps `MHD_*` names to strings (unset = default).
The production driver repeats these checks on its own environment.
"""
function _validate_ct_environment(env::AbstractDict)
    lowercase(get(env, "MHD_SOLVER", "glm")) == "ct" || return nothing
    drag = lowercase(get(env, "MHD_DRAG_DT_MODE", ""))
    drag in ("terminal", "hybrid") && throw(ArgumentError(
        "mhd_solver=\"ct\" does not support MHD_DRAG_DT_MODE=$drag: the terminal closures " *
        "overwrite cell-centred B and cannot update CT faces; use radiation, godunov, or explicit"))
    lowercase(get(env, "MHD_INTEGRATOR", "ref")) == "cube" && throw(ArgumentError(
        "mhd_solver=\"ct\" requires MHD_INTEGRATOR=ref; the cube integrators are GLM-only"))
    if haskey(env, "MHD_GLM_PROJECT_EVERY")
        n = tryparse(Int, strip(env["MHD_GLM_PROJECT_EVERY"]))
        n !== nothing && n > 0 && throw(ArgumentError(
            "mhd_solver=\"ct\" does not use GLM B projection: MHD_GLM_PROJECT_EVERY=$n " *
            "would overwrite cell-centred B; leave it unset (CT default 0)"))
    end
    nothing
end

function known_environment_keys()
    source = read(joinpath(@__DIR__, "production", "runner.jl"), String)
    Set(m.captures[1] for m in eachmatch(r"\"(MHD_[A-Z0-9_]+)\"", source))
end

"Resolve every supplied setting to the production driver's public environment interface."
function run_environment(c::RunConfig; output_dir::AbstractString)
    validate_config(c)
    env = Dict("MHD_BACKEND"=>"cpu", "MHD_PRECISION"=>"f32", "MHD_PMF_N"=>"16",
        "MHD_STEPS"=>"4", "MHD_CFL"=>"0.2", "MHD_PK_NMU"=>"1",
        "MHD_CYCLE_PRINT_EVERY"=>"100", "MHD_WARMUP"=>"0",
        "MHD_CHEM_IONIZED_FAST_ZMIN"=>"0", "MHD_CHECKPOINT_FINAL"=>"false",
        "MHD_CHECKPOINT_RESUME"=>"false", "MHD_FFT"=>"auto")
    if _setting(c, "cosmology", "zstart", nothing) !== nothing
        merge!(env, Dict("MHD_P0"=>"auto", "MHD_COMPTON_DRAG"=>"true",
            "MHD_DRAG_DT_MODE"=>"radiation", "MHD_RADIATION_RESPONSE_MODEL"=>"moments_exponential",
            "MHD_RADIATION_NONLINEAR_CFL"=>"0.1", "MHD_RADIATION_MAX_DLNA"=>"2.5e-4",
            "MHD_CHEM"=>"mixing", "MHD_CHEM_EXPANSION"=>"on", "MHD_CHEM_HELIUM"=>"true",
            "MHD_CHEM_H_STORAGE"=>"centered", "MHD_CHEM_SMOOTH"=>"gaussian",
            "MHD_CHEM_SMOOTH_SIGMA_CELLS"=>"8", "MHD_CHEM_COUPLING"=>"strang",
            "MHD_PARTICLE_MAX_DISP_CELLS"=>"1", "MHD_COSMOLOGICAL_MHD_SCALING"=>"jedamzik"))
    end
    for (section, mapping) in CONFIG_KEYS
        for (key, name) in mapping
            value = _setting(c, section, key, nothing)
            section == "diagnostics" && key == "streaming" && continue
            value === nothing || (env[name] = _envstr(value))
        end
    end
    # The physical box is ckpc/h; this upstream option expects ckpc.
    if haskey(env, "MHD_TERMINAL_LBOX_CKPC")
        env["MHD_TERMINAL_LBOX_CKPC"] = string(parse(Float64, env["MHD_TERMINAL_LBOX_CKPC"]) /
                                                    parse(Float64, get(env, "MHD_H", "0.674")))
    end
    out = abspath(output_dir)
    merge!(env, Dict("MHD_HISTORY_TSV"=>joinpath(out,"history.tsv"),
        "MHD_PK_TSV"=>joinpath(out,"spectra.tsv"), "MHD_PDF_TSV"=>joinpath(out,"pdfs.tsv"),
        "MHD_OUT_TSV"=>joinpath(out,"performance.tsv"),
        "MHD_MODE_HISTORY_TSV"=>joinpath(out,"modes.tsv"),
        "MHD_CHECKPOINT_PATH"=>joinpath(out,"checkpoint.jls"),
        "MHD_INITIAL_PK"=>"1", "MHD_FINAL_DIAG"=>"1"))
    if _setting(c,"diagnostics","streaming",false) || any(
        _setting(c,"streaming",key,0) != 0 for key in ("vx_kms","vy_kms","vz_kms"))
        env["MHD_STREAMING_TSV"] = joinpath(out,"streaming.tsv")
    end
    if _setting(c,"diagnostics","frame_redshifts",nothing) !== nothing
        env["MHD_DENSITY_FRAME_DIR"] = joinpath(out,"frames")
    end
    for (key,value) in get(c.settings,"environment",Dict())
        env[key] = _envstr(value)
    end
    haskey(env, "MHD_SOLVER") && (env["MHD_SOLVER"] = lowercase(env["MHD_SOLVER"]))
    _validate_ct_environment(env)
    env
end
