using Test, Statistics
import PMFGLM
ENV["MHD_BACKEND"]="cpu"
include(joinpath(@__DIR__,"..","src","production","runner.jl"))

@testset "photon expansion preserves physical dipole and monopole" begin
    for TT in (Float32,Float64)
        s=MHDKernels.allocate_state(MHDKernels.backend(:cpu),TT,(8,8,8);dx=1/8)
        fill!(s.U[1],1); fill!(s.U[2],0.125); fill!(s.U[3],0); fill!(s.U[4],0)
        fill!(s.U[5],2); foreach(x->fill!(x,0),s.U[6:9])
        photons=MHDKernels.allocate_photon_moment_state(s)
        fill!(photons.hat_batch,complex(TT(0.1),TT(0.2)))
        monopole=copy(photons.delta_hat)
        dipole=copy(photons.velocity_hat[1])
        a0=1/1201; a1=1/601; vunit=8e5
        half=sqrt(_hubble_momentum_decay(a0,a1,a0))
        for _=1:2
            MHDKernels.apply_linear_drag!(s,half)
            MHDKernels.rescale_photon_velocities!(photons,inv(half))
        end
        unit=_cosmological_velocity_unit(vunit,a1,a0)
        @test photons.delta_hat == monopole
        @test photons.velocity_hat[1].*unit ≈ dipole.*vunit rtol=5eps(TT)
        @test mean(s.U[2])*unit ≈ 0.125vunit*a0/a1 rtol=20eps(TT)
        @test_throws ArgumentError MHDKernels.rescale_photon_velocities!(photons,NaN)
    end
end

@testset "gravity source step resolves the fast slip eigenvalue" begin
    for mode in (:godunov,:radiation,:aware,:semiimplicit), R in (0.,0.2,10.)
        cap=_source_drag_impulse_cap(mode,Inf,true,0.01,1+R)
        q=(1+R)*cap
        @test q ≈ 0.01
        # Exact relative bias of repeated final kicks for a constant-force
        # exponential relaxation; a bound on the old stiff-splitting failure.
        @test q/(-expm1(-q))-1 < 0.00501
    end
    @test isinf(_source_drag_impulse_cap(:radiation,Inf,false,0.01))
    @test _source_drag_impulse_cap(:aware,0.001,true,0.01) == 0.001
end

@testset "Eulerian particle velocity follows translated particles" begin
    for TT in (Float32,Float64), lattice in (false,true)
        n=16
        q=TT[(i-0.5)/n for i=1:n,j=1:n,k=1:n]
        qy=TT[(j-0.5)/n for i=1:n,j=1:n,k=1:n]
        qz=TT[(k-0.5)/n for i=1:n,j=1:n,k=1:n]
        v=vec(cos.(2TT(pi).*q)); dx=fill(TT(0.25),n^3); zeroes=zeros(TT,n^3)
        px=lattice ? dx : vec(mod.(q.+TT(0.25),one(TT)))
        py=lattice ? zeroes : vec(qy); pz=lattice ? zeroes : vec(qz)
        velocity=zeros(TT,n^3); density=similar(velocity); scratch=similar(velocity)
        MHDKernels.deposit_particle_velocity!(velocity,density,px,py,pz,v;
                                             N=n,lattice_displacements=lattice)
        @test density ≈ ones(TT,n^3)
        @test sum(density.*velocity) ≈ sum(v) atol=1e-3
        mode=_device_complex_mode!(MHDKernels.backend(:cpu),scratch,velocity,n,1)
        @test mode ≈ -im atol=3e-6
    end
end

@testset "physical box controls DM velocity and mode wavenumber" begin
    n=8; s=MHDKernels.allocate_state(MHDKernels.backend(:cpu),Float32,(n,n,n);dx=1/n)
    fill!(s.U[1],1); foreach(x->fill!(x,0),s.U[2:9]); fill!(s.U[5],1.5)
    q=Float32[(i-0.5)/n for i=1:n,j=1:n,k=1:n]
    v=vec(cos.(2f0*Float32(pi).*q)); dx=fill(0.25f0,n^3); dz=zeros(Float32,n^3)
    mktempdir() do dir
        path=joinpath(dir,"modes.tsv")
        _emit_diagnostics!(be_p=s.be,s=s,ρdm=zeros(Float32,n^3),
            scratch=zeros(Float32,n^3),px=dx,py=dz,pz=dz,vx=v,vy=dz,vz=dz,
            N=n,ncells=n^3,gravity=true,cycle=0,measured=0,a=0.001,
            pk_path="",history_path="",pdf_path="",pdf_counts=nothing,
            pdf_loglo=-1.,pdf_loghi=1.,nmu=1,nbins=4,boxsize=1.,axis=1,
            lattice_displacements=true,mode_history_path=path,
            physical_box_ckpc_h=1000.)
        lines=split.(readlines(path),'\t'); header=lines[1]
        row=only(filter(x->x[1]=="dm_v_x",lines[2:end]))
        getval(name)=parse(Float64,row[findfirst(==(name),header)])
        @test getval("imag") ≈ -_particle_velocity_cms_per_code(.001,1000.) rtol=3e-6
        @test getval("k_comoving_mpc_inv") ≈ 2pi*.674
    end
end

@testset "pre-fix photon checkpoint cannot silently resume" begin
    mktempdir() do dir
        path=joinpath(dir,"old.jls")
        _write_pmf_checkpoint(path,(N=8,),(a=.001,),["x"=>zeros(Float32,1)])
        @test_throws ErrorException _restore_pmf_checkpoint!(path,
            (N=8,velocity_transport_version=2),["x"=>zeros(Float32,1)])
    end
end

include("relative_velocity_checks.jl")
@testset "reused deposition buffers and photon scaling on CPU" begin
    for TT in (Float32,Float64)
        relative_velocity_checks(:cpu,TT)
    end
end
