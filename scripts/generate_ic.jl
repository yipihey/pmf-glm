#!/usr/bin/env julia
using PMFGLM
using FFTW
using TOML
using SHA

length(ARGS) == 2 || error("usage: julia --project=. -t 4 scripts/generate_ic.jl CONFIG.toml OUTPUT_DIR")
config = load_config(ARGS[1])
out = abspath(ARGS[2])
ispath(out) && !isempty(readdir(out)) && error("IC output must be new or empty")
mkpath(out)
env = run_environment(config;output_dir=out)
n = parse(Int,env["MHD_PMF_N"])
get(env,"MHD_PMF_INIT","batchelor") == "batchelor" || error("this generator writes Batchelor fields; use tools/reference for matched linear ICs")
FFTW.set_num_threads(parse(Int,get(ENV,"PMF_FFTW_THREADS",string(min(4,Threads.nthreads())))))
brms = parse(Float64,get(env,"MHD_PMF_BRMS","0.001"))
if haskey(env,"MHD_PMF_B0_NG")
    z = parse(Float64,env["MHD_ZSTART"])
    h = parse(Float64,get(env,"MHD_H","0.674"))
    om = parse(Float64,get(env,"MHD_OM","0.315"))
    ob = parse(Float64,get(env,"MHD_OB",string(0.17037*om)))
    vunit = parse(Float64,get(env,"MHD_CHEM_VUNIT","8e5"))
    brms = magnetic_code_amplitude(parse(Float64,env["MHD_PMF_B0_NG"]),z;
                                   h,omega_b=ob,velocity_unit_cms=vunit)
end
kwargs = Dict{Symbol,Any}(:seed=>parse(Int,get(env,"MHD_PMF_SEED","42")),
    :kcut=>parse(Float64,get(env,"MHD_PMF_KCUT",string(0.625pi*n))),
    :brms=>brms,
    :box=>1.0)
for (name,key,convert) in (
    ("MHD_PMF_KMAX",:kmax,x->parse(Float64,x)),
    ("MHD_PMF_KMIN",:kmin,x->parse(Float64,x)),
    ("MHD_PMF_MODE_LOCK_N",:mode_lock_n,x->parse(Int,x)),
    ("MHD_PMF_CURL_REFERENCE_N",:curl_reference_n,x->parse(Int,x)),
    ("MHD_PMF_CURL_SYMBOL",:curl_symbol,Symbol),
    ("MHD_PMF_NORMALIZATION_KIND",:normalization_kind,Symbol),
    ("MHD_PMF_PRESERVE_LOW_KMAX",:preserve_low_kmax,x->parse(Float64,x)),
    ("MHD_PMF_PRESERVE_LOW_KCUT",:preserve_low_kcut,x->parse(Float64,x)),
    ("MHD_PMF_MIN_CELLS_PER_WAVELENGTH",:minimum_cells_per_wavelength,x->parse(Float64,x)),
    ("MHD_PMF_FIXED_MODE_POWER",:fixed_mode_power,x->x in ("1","true")),
    ("MHD_PMF_MODE_REALIZATION",:mode_realization,Symbol),
    ("MHD_PMF_REALIZATION_BOX",:realization_box,x->parse(Float64,x)),
)
    haskey(env,name) && (kwargs[key]=convert(env[name]))
end
Bx,By,Bz,stats = pmf_batchelor_field((n,n,n);kwargs...)
hashes = Dict{String,String}()
for (name,field) in zip(("bx.f32","by.f32","bz.f32"),(Bx,By,Bz))
    path = joinpath(out,name)
    open(io->write(io,Float32.(field)),path,"w")
    hashes[name] = bytes2hex(sha256(read(path)))
end
unit = "MHD code magnetic units at the configured initial redshift"
metadata = Dict("n"=>n,"unit"=>unit,"order"=>"Julia column-major (x fastest)",
    "dtype"=>"Float32","endianness"=>ENDIAN_BOM==0x04030201 ? "little" : "big",
    "source_fingerprint_sha256"=>source_fingerprint(),"sha256"=>hashes,
    "total_brms"=>Float64(stats.total_brms_measured),
    "reference_band_brms"=>Float64(stats.reference_band_brms_measured),
    "normalization_kind"=>stats.normalization_kind,"configuration"=>config.settings)
open(io->TOML.print(io,metadata;sorted=true),joinpath(out,"ic.toml"),"w")
println("Wrote ",out,"; B units: ",unit,"; total Brms=",stats.total_brms_measured)
println("Import at the same start redshift/units with MHD_INITIAL_B_IMPORT_DIR; raw import preserves each Fourier amplitude. Regenerate for a different normalization epoch.")
