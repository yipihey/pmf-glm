# Shared checks on CPU and explicitly requested real GPU hardware.
function relative_velocity_checks(name, T)
    M = PMFGLM.MHDKernels
    be = M.backend(name)
    n = 16
    q = T[(i-.5)/n for i=1:n,j=1:n,k=1:n]
    initial = vec(cospi.(2q))
    pos = ntuple(d->M.to_device(be,fill(T(d==1 ? .25 : 0),n^3)),3)
    v = M.to_device(be,initial)
    velocity = M.device_zeros(be,T,(n^3,))
    density = similar(velocity)
    for _=1:2 # Reusing dirty outputs must not retain previous deposited mass.
        M.deposit_particle_velocity!(velocity,density,pos...,v;N=n,lattice_displacements=true)
        host = M.to_host(velocity); mass = M.to_host(density)
        @test mass ≈ ones(T,n^3)
        expected = vec(sinpi.(2q))
        @test maximum(abs,host.-expected) < 2e-6
        @test abs(sum(Float64.(mass).*Float64.(host))-sum(Float64.(initial))) < 1e-4
    end
    # Deliberately empty most cells; zero velocity must retain an empty mask.
    absolute = ntuple(_->M.to_device(be,fill(T(.5/n),n^3)),3)
    M.deposit_particle_velocity!(velocity,density,absolute...,v;N=n)
    @test count(iszero,M.to_host(density)) == n^3-1
    @test all(iszero,M.to_host(velocity)[2:end])

    s = M.allocate_state(be,T,(8,8,8);dx=1/8)
    photons = M.allocate_photon_moment_state(s)
    fill!(photons.hat_batch,complex(T(.1),T(.2)))
    before = M.to_host(photons.hat_batch)
    M.rescale_photon_velocities!(photons,2)
    after = M.to_host(photons.hat_batch)
    @test after[:,:,:,1] == before[:,:,:,1]
    @test after[:,:,:,2:4] == 2before[:,:,:,2:4]
    ws = M.allocate_radiation_workspace(s)
    M.photon_fields_to_workspace!(ws,photons)
    @test M.to_host(photons.hat_batch) == after
    M.photon_fields_to_workspace!(ws,photons)
    @test M.to_host(photons.hat_batch) == after
end
