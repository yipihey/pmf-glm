#!/usr/bin/env julia
# Warm MHD-kernel throughput. --mhd-solver=glm (default) keeps the historical behaviour;
# ct measures face-staggered constrained transport (always the :ref integrator); both runs
# GLM (default integrator), GLM on :ref (if that differs: the like-for-like integrator for
# CT) and CT, and prints one table. Single-solver and combined output default to results/.
using PMFGLM
using KernelAbstractions
using TOML
using Dates
using FFTW
using Printf

be_name = :cpu
n = 32
iterations = 10
solver = "glm"
output = nothing
for arg in ARGS
    key,value = split(arg,'=';limit=2)
    key == "--backend" ? (global be_name=Symbol(value)) :
    key == "--n" ? (global n=parse(Int,value)) :
    key == "--iterations" ? (global iterations=parse(Int,value)) :
    key == "--mhd-solver" ? (global solver=lowercase(value)) :
    key == "--output" ? (global output=abspath(value)) : error("unknown option $key")
end
solver in ("glm","ct","both") || error("--mhd-solver must be glm, ct, or both")
if output === nothing
    name = solver == "glm" ? "benchmark.toml" : solver == "ct" ? "benchmark-ct.toml" : "benchmark-compare.toml"
    output = joinpath(@__DIR__,"..","results",name)
end
n >= 8 && n % 4 == 0 || error("benchmark n must be a multiple of four >=8")
iterations > 0 || error("iterations must be positive")
if be_name == :metal
    @eval using Metal
    Metal.functional() || error("Metal hardware is unavailable")
elseif be_name == :cuda
    @eval using CUDA
    CUDA.functional() || error("CUDA hardware is unavailable")
elseif be_name != :cpu
    error("backend must be cpu, metal, or cuda")
end
be = backend(be_name)
device = be_name == :metal ? string(Metal.device()) :
         be_name == :cuda ? CUDA.name(CUDA.device()) : first(Sys.cpu_info()).model
default_integrator = be_name == :cpu ? :ref : :cube

function measure(mhd_solver::Symbol,integrator::Symbol)
    s = allocate_state(be,Float32,(n,n,n);dx=1/n,recon=:ppm,riemann=:hlld,mhd_solver)
    init!() = (MHDKernels.init_turb_field!(s); mhd_solver === :ct && MHDKernels.ct_faces_from_cells!(s))
    init!()
    work = MHDKernels.device_zeros(be,Float32,(n^3,))
    dt,smax = MHDKernels.compute_dt!(work,s;cfl=0.2)
    advance() = step!(s,dt;ch=smax,integrator) # ch is the GLM wave speed; CT ignores it
    advance(); KernelAbstractions.synchronize(be)
    init!() # Warm compilation must not advance the measured IC.
    GC.gc()
    KernelAbstractions.synchronize(be)
    start = time_ns()
    for _ in 1:iterations
        advance()
    end
    KernelAbstractions.synchronize(be)
    elapsed = (time_ns()-start)/1e9
    host = fields_to_host(s)
    finite = all(all(isfinite,a) for a in host)
    positive_density = minimum(host[1]) > 0
    brms = sqrt(sum(v->sum(abs2,Float64.(host[v])),6:8)/n^3)
    divb = Float64(max_divb(s))*Float64(s.dx)/brms # centred-difference diagnostic, any solver
    face = mhd_solver === :ct ? Float64(MHDKernels.max_divb_face(s))*Float64(s.dx)/brms : NaN
    payload = Dict("schema_version"=>1,"backend"=>string(be_name),"n"=>n,
        "iterations"=>iterations,"integrator"=>string(integrator),"mhd_solver"=>string(mhd_solver),
        "wall_seconds"=>elapsed,"mcells_per_second"=>n^3*iterations/elapsed/1e6,
        "julia_threads"=>Threads.nthreads(),"julia_version"=>string(VERSION),
        "created_utc"=>string(now(UTC)),"source_fingerprint_sha256"=>source_fingerprint(),
        "machine"=>gethostname(),"device"=>device,"platform"=>string(Sys.KERNEL,"/",Sys.ARCH),
        "finite"=>finite,"positive_density"=>positive_density,
        "divb_centred_dx_over_brms"=>divb,
        "status"=>finite && positive_density ? "PASS" : "FAIL",
        "scope"=>"warm MHD kernel only; excludes chemistry, FFTs, gravity, diagnostics, startup")
    isnan(face) || (payload["divb_face_dx_over_brms"] = face)
    payload
end

function write_payload(payload,path)
    mkpath(dirname(path))
    open(io->TOML.print(io,payload;sorted=true),path,"w")
end

if solver != "both"
    payload = measure(Symbol(solver),solver == "ct" ? :ref : default_integrator)
    write_payload(payload,output)
    TOML.print(stdout,payload;sorted=true)
    payload["status"] == "PASS" || error("benchmark ended with an invalid MHD state")
else
    cases = [(:glm,default_integrator,"glm")]
    default_integrator == :cube && push!(cases,(:glm,:ref,"glm-ref"))
    push!(cases,(:ct,:ref,"ct"))
    stem,ext = splitext(output)
    payloads = Dict{String,Any}[]
    skipped = Tuple[]
    for (mhd_solver,integrator,label) in cases
        result = try
            measure(mhd_solver,integrator)
        catch err
            # e.g. the GLM :ref kernels exceed Metal's 31 kernel-argument limit; say so, do not hide it.
            reason = first(split(sprint(showerror,err),'\n'))
            println(stderr,"case ",label," NOT_RUN: ",reason)
            push!(skipped,(label,mhd_solver,integrator,reason))
            continue
        end
        result["case_label"] = label
        write_payload(result,string(stem,"-",label,ext))
        push!(payloads,result)
    end
    println("| solver | integrator | backend | N | iterations | Mcell/s | divB centred dx/Brms | divB face dx/Brms | status |")
    println("|---|---|---|---|---|---|---|---|---|")
    for p in payloads
        @printf("| %s | %s | %s | %d | %d | %.3f | %.2e | %s | %s |\n",p["mhd_solver"],p["integrator"],
            p["backend"],p["n"],p["iterations"],p["mcells_per_second"],p["divb_centred_dx_over_brms"],
            haskey(p,"divb_face_dx_over_brms") ? @sprintf("%.2e",p["divb_face_dx_over_brms"]) : "n/a",p["status"])
    end
    for (label,mhd_solver,integrator,reason) in skipped
        println("| ",mhd_solver," | ",integrator," | ",be_name," | ",n," | ",iterations," | NOT_RUN | n/a | n/a | ",reason," |")
    end
    println("Scope: ",first(payloads)["scope"],". GLM rows pass ch=smax; CT is not GLM-limited. Files: ",stem,"-<label>",ext)
    all(p["status"]=="PASS" for p in payloads) || error("benchmark ended with an invalid MHD state")
end
