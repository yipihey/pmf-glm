# Shared CPU and real-device checks; no device scalar indexing.
function streaming_checks(name, T)
    M = PMFGLM.MHDKernels
    K = KernelAbstractions
    n = 8
    be = M.backend(name)
    s = M.allocate_state(be,T,(n,n,n); dx=1/n,gamma=5/3,recon=:plm,riemann=:hll)
    r = T[1+0.2cospi(2*(i-0.5)/n) for i in 1:n,j in 1:n,k in 1:n]
    for (d,field) in enumerate((r,zero.(r),zero.(r),zero.(r),fill(T(2),size(r)),
                               zero.(r),zero.(r),zero.(r),zero.(r)))
        copyto!(s.U[d],vec(field))
    end
    v = (T(0.2),T(-0.3),T(0.4))
    M.boost_gas_velocity!(s,v)
    K.synchronize(be)
    fields = M.fields_to_host(s)
    @test maximum(abs,fields[5] .- (2 .+ vec(r).*sum(abs2,v)/2)) < 2e-6
    for d in 1:3
        @test maximum(abs,fields[d+1] ./ vec(r) .- v[d]) < 1e-6
    end
    positions = ntuple(_->M.device_zeros(be,T,(n^3,)),3)
    velocities = ntuple(d->M.to_device(be,fill(v[d]/T(2),n^3)),3)
    out = ntuple(_->M.device_zeros(be,T,(n^3,)),3)
    # A common physical boost must cancel, despite different code units.
    M.sample_baryon_dm_velocity!(out,s,positions,velocities;gas_unit=3,dm_unit=6)
    K.synchronize(be)
    @test maximum(maximum(abs,M.to_host(x)) for x in out) < 1e-6
    # Probe an x-dependent gas velocity at wrapped positions, not particle labels.
    speed = T[cospi(2*(i-0.5)/n) for i in 1:n,j in 1:n,k in 1:n]
    copyto!(s.U[2],vec(r.*speed))
    for d in 1:3
        copyto!(positions[d],fill(T(d==1 ? -1.25/n : 0),n^3))
    end
    M.sample_baryon_dm_velocity!(out,s,positions,velocities;gas_unit=3,dm_unit=6)
    K.synchronize(be)
    expected = T[3*(0.25speed[mod1(i-2,n),j,k]+0.75speed[mod1(i-1,n),j,k])-3v[1]
                  for i in 1:n,j in 1:n,k in 1:n]
    @test maximum(abs,M.to_host(out[1]).-vec(expected)) < 3e-6
    # Absolute-position convention gives the same interpolation.
    absolute = ntuple(d->T[(d==1 ? i-0.5-1.25 : d==2 ? j-0.5 : k-0.5)/n
                           for i in 1:n,j in 1:n,k in 1:n],3)
    for d in 1:3
        copyto!(positions[d],vec(absolute[d]))
    end
    M.sample_baryon_dm_velocity!(out,s,positions,velocities;
        gas_unit=3,dm_unit=6,lattice_displacements=false)
    K.synchronize(be)
    @test maximum(abs,M.to_host(out[1]).-vec(expected)) < 3e-6

    for axis in 1:3
        stats = M.init_pmf_batchelor!(s;brms=0.1,p0=1,seed=14,kmax=4pi,
                                      invariant_axis=axis)
        K.synchronize(be)
        initial = M.fields_to_host(s)
        @test stats.invariant_axis == axis
        for d in 6:8
            field = reshape(initial[d],n,n,n)
            @test maximum(abs,field.-circshift(field,ntuple(i->i==axis ? 1 : 0,3))) < 2e-7
            @test maximum(abs,field) > 0.001
        end
        M.boost_gas_velocity!(s,(0.1,0.2,0.3))
        M.step!(s,1e-4;ch=1,integrator=name === :cpu ? :ref : :cube)
        K.synchronize(be)
        for field in M.fields_to_host(s)
            cube = reshape(field,n,n,n)
            @test maximum(abs,cube.-circshift(cube,ntuple(i->i==axis ? 1 : 0,3))) < 2e-6
        end
    end
    # Reading photon maps must preserve both the live Fourier modes and future
    # dynamics. FFTW c2r may destroy input even for an out-of-place transform.
    states = ntuple(2) do _
        state = M.allocate_state(be,T,(n,n,n);dx=1/n,gamma=5/3,recon=:plm,riemann=:hll)
        fill!(state.U[1],one(T));fill!(state.U[5],T(2))
        M.boost_gas_velocity!(state,(0.125,0.0625,0.03125))
        state
    end
    workspaces = map(M.allocate_radiation_workspace,states)
    photons = map(M.allocate_photon_moment_state,states)
    for d in 1:2
        M.initialize_photon_moments!(photons[d],workspaces[d],states[d])
    end
    closure = M.RadiationClosure(inertia_ratio=.23,mean_free_path=70.,
        photon_speed=3e4,gas_sound_speed=1.5,response_model=:moments_exponential)
    for _ in 1:2
        before = M.to_host(photons[1].hat_batch)
        M.photon_fields_to_workspace!(workspaces[1],photons[1])
        @test M.to_host(photons[1].hat_batch) == before
        real = M.to_host(workspaces[1].real_batch)
        @test maximum(abs,real[:,:,:,2].-T(.125)) < 2e-6
        for d in 1:2
            M.step_radiation_godunov!(states[d],workspaces[d],1e-4;
                drag_impulse=.05,closure=closure,ch=2,
                integrator=name === :cpu ? :ref : :cube,photon_state=photons[d])
        end
    end
    for (left,right) in zip(M.fields_to_host(states[1]),M.fields_to_host(states[2]))
        @test maximum(abs,left.-right) < 2e-6
    end
    @test maximum(abs,M.to_host(photons[1].hat_batch).-M.to_host(photons[2].hat_batch)) < 2e-6

end
