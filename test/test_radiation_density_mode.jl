# Regression: the persistent photon-coupled density Fourier mode must survive
# the inverse transforms of the radiation corrector. FFTW's multidimensional
# c2r transform destroys its complex input, so a mode increment read back after
# inverse-transforming it is garbage on the CPU (device FFTs preserve it). The
# corrupted "authoritative" density then feeds a grid-scale instability that
# grows by a factor of ~3 per step at small dt and ends in a NaN timestep.
const _RM = PMFGLM.MHDKernels

function _radiation_mode_state(n)
    F = ntuple(_ -> zeros(Float32,n^3),9)
    for k in 1:n, j in 1:n, i in 1:n
        x = Float32(2pi*(i-0.5)/n); y = Float32(2pi*(j-0.5)/n); z = Float32(2pi*(k-0.5)/n)
        u = _RM.prim2cons((1f0+0.02f0*sin(x), 0.02f0*sin(z), 0.02f0*sin(x),
                           0.02f0*sin(y), 1f0+0.01f0*cos(z), 0.3f0,
                           0.02f0*sin(z), 0.02f0*sin(x), 0f0), Float32(5/3))
        for v in 1:9
            F[v][_RM.linidx(n,n,i,j,k)] = u[v]
        end
    end
    s = _RM.allocate_state(_RM.backend(:cpu),Float32,(n,n,n);dx=1/n,gamma=5/3,
                           recon=:ppm,riemann=:hlld)
    for v in 1:9
        copyto!(s.U[v],F[v])
    end
    s
end

@testset "persistent radiation density mode is not destroyed by c2r" begin
    n = 16
    s = _radiation_mode_state(n)
    ws = _RM.allocate_radiation_workspace(s)
    ph = _RM.allocate_photon_moment_state(s)
    _RM.initialize_photon_moments!(ph,ws,s)
    closure = _RM.RadiationClosure(inertia_ratio=0.1f0,mean_free_path=0.01f0,
        photon_speed=10f0,gas_sound_speed=1f0,response_model=:moments_exponential)
    dt, speed = _RM.compute_dt(s;cfl=0.02)
    dt /= 8   # small steps: the regime of the end-to-end native-mode comparison
    worst = 0.0
    for _ in 1:8
        _RM.step_radiation_godunov!(s,ws,dt;drag_impulse=0.02f0,closure=closure,
            ch=speed,integrator=:ref,photon_state=ph)
        delta = Array(reshape(ws.density_perturbation,n,n,n))
        expected = FFTW.rfft(delta,(1,2,3))
        authoritative = Array(ws.authoritative_density_hat)
        worst = max(worst,norm(authoritative-expected)/norm(expected))
    end
    @test all(isfinite,_RM.fields_to_host(s)[1])
    # Fixed code stays below 4e-5 (physical drift of the incrementally updated
    # real density); the destroyed-input bug reaches 1e-2 after eight steps and
    # roughly triples every further step.
    @test worst < 2e-4
end
