using TOML

@test load_config(joinpath(@__DIR__,"..","examples","smoke.toml")) isa RunConfig
@test_throws ArgumentError RunConfig(Dict("run"=>Dict("n"=>7)))
@test_throws ArgumentError RunConfig(Dict("run"=>Dict("backend"=>"unknown")))
@test_throws ArgumentError RunConfig(Dict("run"=>Dict("backend"=>"metal","precision"=>"f64")))
@test_throws ArgumentError RunConfig(Dict("run"=>Dict("cf1"=>0.2)))
@test_throws ArgumentError RunConfig(Dict("environment"=>Dict("MHD_TYPO"=>1)))
@test_throws ArgumentError RunConfig(Dict("cosmology"=>Dict("zstart"=>500,"zend"=>1000)))
@test_throws ArgumentError RunConfig(Dict("cosmology"=>Dict("zstart"=>1000,"zend"=>500,"box_ckpc_h"=>1)))

config = load_config(joinpath(@__DIR__,"..","examples","pmf-recombination.toml"))
env = run_environment(config;output_dir="/tmp/pmf-config-test")
@test env["MHD_PMF_B0_NG"] == "0.003"
@test parse(Float64,env["MHD_TERMINAL_LBOX_CKPC"]) ≈ 1/0.674
@test env["MHD_P0"] == "auto"
@test env["MHD_CHEM_IONIZED_FAST_ZMIN"] == "0.0"
@test env["MHD_CHEM_EXPANSION"] == "on"
@test env["MHD_CHEM_H_STORAGE"] == "centered"
@test env["MHD_RADIATION_RESPONSE_MODEL"] == "moments_exponential"
@test env["MHD_PK_NMU"] == "1"
@test all(startswith(k,"MHD_") for k in keys(env))
@test length(source_fingerprint()) == 64
@test source_fingerprint() == source_fingerprint()
@test PMFGLM._completion_status("COSMO final_z=500.000000 final_a=0.001",Dict("MHD_ZSTART"=>"3200","MHD_ZEND"=>"500")) == "COMPLETED"
@test PMFGLM._completion_status("COSMO final_z=1000.000000",Dict("MHD_ZSTART"=>"3200","MHD_ZEND"=>"500")) == "STOPPED_AT_LIMIT"
@test PMFGLM._completion_status("no final epoch",Dict("MHD_ZSTART"=>"3200")) == "INCOMPLETE"
# At fixed comoving B, physical B scales as a^-2 but the MHD code unit
# includes sqrt(rho), giving a^-1/2 in these initial physical velocity units.
@test magnetic_code_amplitude(0.003,2001)/magnetic_code_amplitude(0.003,1000) ≈ sqrt(2)
@test magnetic_code_amplitude(0.003,1000) > 0
@test_throws ArgumentError magnetic_code_amplitude(0.003,1000;velocity_unit_cms=0)
mktempdir() do dir
    result = run_simulation(config;output_dir=dir,dry_run=true)
    @test result["MHD_HISTORY_TSV"] == joinpath(dir,"history.tsv")
    @test isempty(readdir(dir))
end

# All local source paths must resolve inside this checkout. Active-root source
# declarations are duplicated in the optional environments by Julia's rules.
root = dirname(@__DIR__)
for relative in ("Project.toml","environments/metal/Project.toml","environments/cuda/Project.toml")
    project = joinpath(root,relative)
    data = TOML.parsefile(project)
    for (name, source) in data["sources"]
        path = normpath(joinpath(dirname(project),source["path"]))
        @test startswith(path,root*Base.Filesystem.path_separator) || path == root
        @test isfile(joinpath(path,"Project.toml"))
        @test haskey(data["deps"],name)
    end
end
