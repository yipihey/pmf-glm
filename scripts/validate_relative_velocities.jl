#!/usr/bin/env julia
# End-to-end controlled checks. Output is new simulation evidence, not a PMF campaign.
using Test, Statistics, Serialization, Printf, TOML, SHA
import PMFGLM
ENV["MHD_BACKEND"]="cpu"
ENV["MHD_PRECISION"]=length(ARGS)>=2 ? ARGS[2] : "f32"
include(joinpath(@__DIR__,"..","src","production","runner.jl"))
const output_root=abspath(isempty(ARGS) ? joinpath(@__DIR__,"..","results","relative-velocity-validation") : ARGS[1])
isdir(output_root) && !isempty(readdir(output_root)) && error("choose an empty output directory")
mkpath(output_root)
function make_import(dir; gas_v=0.125, photon_v=0.125, dm_v=0.125, displacement=0.)
    mkpath(dir); n=8
    for i=1:9
        value=i==1 ? 1. : i==2 ? gas_v : i==5 ? 1.5+gas_v^2/2 : 0.
        write(joinpath(dir,"mhd_U_$(i).f32"),fill(Float32(value),n,n,n))
    end
    for d in ("x","y","z")
        positions = d=="x" ? vec(Float32[displacement*sinpi(2*(i-.5)/n)
            for i=1:n,j=1:n,k=1:n]) : zeros(Float32,n^3)
        write(joinpath(dir,"particle_p$(d).f32"),positions)
        value=d=="x" ? dm_v*8e5/_particle_velocity_cms_per_code(1/1201,1000.) : 0.
        write(joinpath(dir,"particle_v$(d).f32"),fill(Float32(value),n^3))
    end
    photons=zeros(ComplexF32,n÷2+1,n,n,4)
    photons[1,1,1,2]=Float32(photon_v*n^3)
    write(joinpath(dir,"photon_hat_batch.c32"),photons)
end

function full_run(tag; rate=0., mode="radiation", maxexp=0.02, gravity=false,
                  zend=600., cap=0.01, displacement=0.)
    out=joinpath(output_root,tag); mkpath(out)
    input=joinpath(out,"input")
    v = displacement==0 ? 0.125 : 0.
    make_import(input;gas_v=v,photon_v=v,dm_v=v,displacement)
    env=Dict("MHD_PMF_N"=>"8","MHD_ZSTART"=>"1200","MHD_ZEND"=>string(zend),
      "MHD_MAX_CYCLES"=>"2000","MHD_WARMUP"=>"0","MHD_CFL"=>"0.2",
      "MHD_CHEM"=>"off","MHD_P0"=>"1","MHD_PMF_B0_NG"=>"1e-12",
      "MHD_PMF_INIT"=>"forcefree","MHD_CHEM_VUNIT"=>"800000",
      "MHD_BOXSIZE"=>"1000","MHD_TERMINAL_LBOX_CKPC"=>string(1000/0.674),
      "MHD_COSMOLOGICAL_MHD_SCALING"=>"jedamzik",
      "MHD_INITIAL_STATE_DIR"=>input,"MHD_COMPTON_DRAG"=>"1",
      "MHD_DRAG_DT_MODE"=>mode,"MHD_DRAG_RATE_CODE"=>string(rate),
      "MHD_DRAG_XE"=>"1",
      "MHD_RADIATION_RESPONSE_MODEL"=>"moments_exponential",
      "MHD_RADIATION_FREE_STREAM_TOL"=>"0","MHD_MAXEXP"=>string(maxexp),
      "MHD_RADIATION_MAX_DLNA"=>string(maxexp),
      "MHD_RADIATION_HIGHZ_MAX_DLNA"=>string(maxexp),
      "MHD_GRAVITY_DRAG_MAX_IMPULSE"=>string(cap),
      "MHD_GRAVITY"=>string(gravity),"MHD_GRAVITY_GAS_KICK"=>"1",
      "MHD_PARTICLE_LATTICE_DISPLACEMENTS"=>"1","MHD_CHECK_EVERY"=>"0",
      "MHD_PRINT_EVERY"=>"2000","MHD_CYCLE_PRINT_EVERY"=>"2000",
      "MHD_INITIAL_PK"=>"0","MHD_FINAL_DIAG"=>"0",
      "MHD_CHECKPOINT_PATH"=>joinpath(out,"final.jls"),
      "MHD_CHECKPOINT_FINAL"=>"1","MHD_CHECKPOINT_RESUME"=>"0")
    # Match the facade: unrelated shell MHD_* options must not alter evidence.
    clean = Dict{String,Union{String,Nothing}}(k=>nothing for k in keys(ENV) if startswith(k,"MHD_"))
    merge!(clean,env)
    withenv(clean...) do
        open(joinpath(out,"run.log"),"w") do io
            redirect_stdout(io) do
                main()
            end
        end
    end
    config,runtime,arrays=open(joinpath(out,"final.jls")) do io
        deserialize(io); cfg=deserialize(io); rt=deserialize(io); count=deserialize(io)
        a=Dict{String,Any}()
        for _=1:count
            name=deserialize(io); a[name]=deserialize(io)
        end
        cfg,rt,a
    end
    open(joinpath(out,"resolved-environment.toml"),"w") do io
        TOML.print(io,env)
    end
    open(joinpath(out,"metadata.toml"),"w") do io
        TOML.print(io,Dict("source_fingerprint_sha256"=>PMFGLM.source_fingerprint(),
            "oracle_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","tools","reference","homogeneous_velocities.py")))),
            "julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),
            "backend"=>string(BACKEND_NAME),"precision"=>string(T),
            "target_z"=>zend,"actual_z"=>a_to_z(runtime.a),
            "scope"=>"controlled velocity transport; fixed ionization; chemistry off"))
    end
    a=runtime.a; unit=_cosmological_velocity_unit(8e5,a,1/1201)
    vb=mean(arrays["mhd_U_2"]./arrays["mhd_U_1"])*unit
    vc=mean(arrays["particle_vx"])*_particle_velocity_cms_per_code(a,1000.)
    vg=haskey(arrays,"photon_hat_batch") ? real(arrays["photon_hat_batch"][1,1,1,2])/8^3*unit : NaN
    println("FULL_RUN tag=",tag," z=",a_to_z(a)," vb=",vb," vc=",vc," vg=",vg,
      " free_only_vmatter=",1e5*(1/1201)/a," free_only_vphoton=100000")
    open(joinpath(out,"velocities.tsv"),"w") do io
        println(io,"z\tvb_cms\tvc_cms\tvg_cms\tcycles")
        println(io,join((a_to_z(a),vb,vc,vg,runtime.cycle),'\t'))
    end
    @test a_to_z(a) ≈ zend atol=1e-7
    return vb,vc,vg
end


@testset "production velocity evolution" begin
    vb,vc,vg=full_run("free_expansion";gravity=true)
    @test vb ≈ vc rtol=2e-5
    @test vg ≈ 1e5 rtol=2e-5
    coarse=full_run("physical_drag";rate=NaN)
    fine=full_run("physical_drag_refined";rate=NaN,maxexp=0.005)
    # Independent Radau physical-velocity oracle, xe=1, z=1200->600;
    # reproduce with python -m tools.reference.homogeneous_velocities.
    python=get(ENV,"PYTHON","python3")
    oracle="from tools.reference.homogeneous_velocities import evolve; u=evolve(); print(u[0],u[2],u[1])"
    reference=Tuple(parse.(Float64,split(read(Cmd(Cmd([python,"-c",oracle]);dir=dirname(@__DIR__)),String))))
    @test fine[2] ≈ reference[2] rtol=2e-5
    @test abs(fine[1]-reference[1]) < abs(coarse[1]-reference[1])
    @test abs(fine[3]-reference[3]) < abs(coarse[3]-reference[3])
    @test fine[1] ≈ reference[1] rtol=2e-4
    @test fine[3] ≈ reference[3] rtol=2e-4

end

# Nonzero periodic Poisson force: initially cold matter and photons, with a
# displaced DM lattice. Compare signed complex baryon-photon slip at a common
# endpoint, independently refining only the gravity/drag cap. The fine solution
# is a temporal reference on this fixed grid, not a spatial/physical oracle.
function gravity_slip(tag)
    path=joinpath(output_root,tag,"final.jls")
    cfg,rt,arrays=open(path) do io
        deserialize(io); cfg=deserialize(io); rt=deserialize(io); count=deserialize(io)
        arrays=Dict{String,Any}()
        for _=1:count
            key=deserialize(io); arrays[key]=deserialize(io)
        end
        cfg,rt,arrays
    end
    n=cfg.N
    unit=_cosmological_velocity_unit(8e5,rt.a,1/1201)
    gas=reshape(arrays["mhd_U_2"]./arrays["mhd_U_1"],n,n,n)
    # The k_y=k_z=0 fundamental of the stored forward FFT; match cell centres.
    photon=arrays["photon_hat_batch"][2,1,1,2] * cis(-pi/n)*2/n^3
    weights=reshape([cis(-2pi*(i-.5)/n) for i=1:n],n,1,1)
    baryon=2sum(gas.*weights)/n^3
    return (baryon-photon)*unit, baryon*unit
end

@testset "nonzero Poisson gravity and drag temporal convergence" begin
    # Declared before running: slip >1e-3 cm/s, successive cap refinement
    # reduces complex error by at least 20%, finest candidate within 2%.
    caps=(0.08,0.04,0.01,0.0025)
    slips=ComplexF64[]
    open(joinpath(output_root,"gravity-convergence.tsv"),"w") do io
        println(io,"cap\tz\tslip_real_cms\tslip_imag_cms\tgas_abs_cms\tobserved_impulse")
        for (i,cap) in enumerate(caps)
            tag="gravity_refinement_$i"
            full_run(tag;rate=NaN,gravity=true,zend=1199.5,cap,displacement=0.00125)
            slip,gas=gravity_slip(tag)
            push!(slips,slip)
            log=read(joinpath(output_root,tag,"run.log"),String)
            observed=parse(Float64,match(r"observed_max=([0-9.eE+-]+)",log).captures[1])
            @test 0 < observed <= cap*(1+1e-6)
            @test abs(gas) > 1e-3 # A nonzero velocity is generated from rest.
            println(io,join((cap,1199.5,real(slip),imag(slip),abs(gas),observed),'\t'))
            flush(io)
        end
    end
    @test abs(slips[end]) > 1e-3
    errors=abs.(slips[1:3].-slips[4])
    @test errors[2] < 0.8errors[1]
    @test errors[3] < 0.8errors[2]
    @test errors[3]/abs(slips[4]) < 0.02
end
