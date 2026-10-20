#!/usr/bin/env julia
using PMFGLM
using KernelAbstractions
using TOML
using Dates
using FFTW

be_name = :cpu
n = 32
iterations = 10
output = joinpath(@__DIR__,"..","results","benchmark.toml")
for arg in ARGS
    key,value = split(arg,'=';limit=2)
    key == "--backend" ? (global be_name=Symbol(value)) :
    key == "--n" ? (global n=parse(Int,value)) :
    key == "--iterations" ? (global iterations=parse(Int,value)) :
    key == "--output" ? (global output=abspath(value)) : error("unknown option $key")
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
s = allocate_state(be,Float32,(n,n,n);dx=1/n,recon=:ppm,riemann=:hlld)
MHDKernels.init_turb_field!(s)
work = MHDKernels.device_zeros(be,Float32,(n^3,))
dt,smax = MHDKernels.compute_dt!(work,s;cfl=0.2)
integrator = be_name == :cpu ? :ref : :cube
advance() = step!(s,dt;ch=smax,integrator)
advance(); KernelAbstractions.synchronize(be)
MHDKernels.init_turb_field!(s) # Warm compilation must not advance the measured IC.
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
device = be_name == :metal ? string(Metal.device()) :
         be_name == :cuda ? CUDA.name(CUDA.device()) : first(Sys.cpu_info()).model
payload = Dict("schema_version"=>1,"backend"=>string(be_name),"n"=>n,
    "iterations"=>iterations,"integrator"=>string(integrator),
    "wall_seconds"=>elapsed,"mcells_per_second"=>n^3*iterations/elapsed/1e6,
    "julia_threads"=>Threads.nthreads(),"julia_version"=>string(VERSION),
    "created_utc"=>string(now(UTC)),"source_fingerprint_sha256"=>source_fingerprint(),
    "machine"=>gethostname(),"device"=>device,"platform"=>string(Sys.KERNEL,"/",Sys.ARCH),
    "finite"=>finite,"positive_density"=>positive_density,
    "status"=>finite && positive_density ? "PASS" : "FAIL",
    "scope"=>"warm MHD kernel only; excludes chemistry, FFTs, gravity, diagnostics, startup")
mkpath(dirname(output))
open(io->TOML.print(io,payload;sorted=true),output,"w")
TOML.print(stdout,payload;sorted=true)
finite && positive_density || error("benchmark ended with an invalid MHD state")
