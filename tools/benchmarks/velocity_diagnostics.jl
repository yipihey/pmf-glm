using Statistics
ENV["MHD_BACKEND"]="cpu"
length(ARGS)==1 || error("usage: julia --project=. -t 4 tools/benchmarks/velocity_diagnostics.jl src/production/runner.jl")
include(abspath(only(ARGS)))
function measure(n)
    be=MHDKernels.backend(:cpu)
    s=MHDKernels.allocate_state(be,Float32,(n,n,n);dx=1/n)
    fill!(s.U[1],1); fill!(s.U[5],1.5)
    v=Float32[cospi(2*(i-.5)/n) for i=1:n,j=1:n,k=1:n]
    p=(fill(.25f0,n^3),zeros(Float32,n^3),zeros(Float32,n^3))
    reduction=zeros(Float32,n^3)
    mktempdir() do dir
        path=joinpath(dir,"modes.tsv")
        emit()=_write_linear_mode!(path;be,s,rho_dm=zeros_dm,scratch=reduction,
            N=n,gravity=true,cycle=0,measured=0,a=.001,kmode=1,boxsize=1.,hubble=.674,
            particle_velocity_x=vec(v),particle_positions=p,lattice_displacements=true,
            particle_velocity_cms_factor=1.,physical_box_ckpc_h=1.)
        # Retain the DM field, as the real driver does, outside the measured call.
        zeros_dm=zeros(Float32,n^3)
        emit();emit(); GC.gc()
        bytes=minimum([@allocated emit() for _=1:3])
        elapsed=minimum([@elapsed emit() for _=1:3])
        println(n,"\t",bytes,"\t",elapsed)
    end
end
println("N\twarm_host_allocated_bytes\twarm_seconds")
for n in (16,32,64); measure(n); end
