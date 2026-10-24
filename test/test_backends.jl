module PMFGLMBackendTests

using Test
using FFTW
using KernelAbstractions
using PMFGLM

const MHD = PMFGLM.MHDKernels
const Poisson = PMFGLM.PoissonKernels
const Chemistry = PMFGLM.ChemistryKernels
const Emission = PMFGLM.EmissionKernels
const KA = KernelAbstractions

function _import_backend(name::Symbol)
    name in (:cpu, :metal, :cuda) ||
        throw(ArgumentError("backend must be cpu, metal, or cuda"))
    if name === :metal
        @eval import Metal
    elseif name === :cuda
        @eval import CUDA
    end
    nothing
end

function _require_backend(name::Symbol)
    if name === :metal
        Metal.functional() || error("requested Metal backend is unavailable")
    elseif name === :cuda
        CUDA.functional() || error("requested CUDA backend is unavailable")
    end
    for package in (MHD, Poisson, Chemistry, Emission)
        package.has_backend(name) || error("$(nameof(package)) did not register requested :$name backend")
    end
    nothing
end

function _smooth_fields(n)
    fields = ntuple(_ -> zeros(Float32, n^3), 9)
    for k in 1:n, j in 1:n, i in 1:n
        x = Float32(2pi * (i - 0.5) / n)
        y = Float32(2pi * (j - 0.5) / n)
        z = Float32(2pi * (k - 0.5) / n)
        rho = 1f0 + 0.02f0 * sin(x) * cos(y)
        pressure = 1f0 + 0.01f0 * cos(z)
        velocity = (0.02f0 * sin(z), 0.02f0 * sin(x), 0.02f0 * sin(y))
        magnetic = (0.3f0 + 0.02f0 * sin(y), 0.02f0 * sin(z), 0.02f0 * sin(x))
        u = MHD.prim2cons((rho, velocity..., pressure, magnetic..., 0f0), Float32(5/3))
        index = MHD.linidx(n,n,i,j,k)
        for v in 1:9
            fields[v][index] = u[v]
        end
    end
    fields
end

function _state(name, fields, n; recon=:ppm, riemann=:hlld)
    s = MHD.allocate_state(MHD.backend(name), Float32, (n,n,n);
                           dx=1/n, gamma=5/3, recon, riemann)
    for v in 1:9
        copyto!(s.U[v], fields[v])
    end
    s
end

function _assert_fields(reference, candidate; rtol=5f-5, atol=2f-6)
    for v in 1:9
        @test all(isfinite, candidate[v])
        @test maximum(abs, candidate[v] .- reference[v]) <=
              atol + rtol * maximum(abs, reference[v])
    end
    @test minimum(candidate[1]) > 0f0
end

_mass(fields) = sum(Float64, fields[1])

function _test_smooth_mhd(name)
    n = 16
    fields = _smooth_fields(n)
    initial_mass = _mass(fields)
    for recon in (:plm, :ppm), riemann in (:hlld, :hll, :llf)
        @testset "$recon / $riemann" begin
            cpu = _state(:cpu, fields, n; recon, riemann)
            device = _state(name, fields, n; recon, riemann)
            dt, speed = MHD.compute_dt(cpu; cfl=0.08)
            for _ in 1:3
                MHD.step!(cpu, dt; ch=speed, integrator=:ref)
                if name === :cpu
                    MHD.step!(device, dt; ch=speed, integrator=:ref)
                else
                    decay = exp(-MHD.GLM_CR * speed * dt / device.dx)
                    MHD.step_cube!(device, dt; ch=speed, decay, impl=:ka)
                end
            end
            host_cpu = MHD.fields_to_host(cpu)
            host_device = MHD.fields_to_host(device)
            _assert_fields(host_cpu, host_device)
            @test abs(_mass(host_device) / initial_mass - 1) < 3e-6
            @test maximum(abs, host_device[2] .- fields[2]) > 1f-7
        end
    end
    if name === :cuda
        @testset "auto uses KA; stale raw is rejected" begin
            auto = _state(name, fields, n)
            explicit = _state(name, fields, n)
            dt, speed = MHD.compute_dt(auto; cfl=0.08)
            kwargs = (ch=speed, decay=exp(-MHD.GLM_CR * speed * dt / auto.dx))
            MHD.step_cube!(auto, dt; kwargs..., impl=:auto)
            MHD.step_cube!(explicit, dt; kwargs..., impl=:ka)
            @test MHD.fields_to_host(auto) == MHD.fields_to_host(explicit)
            before = MHD.fields_to_host(auto)
            @test_throws r"impl=:raw is disabled" MHD.step_cube!(auto, dt; kwargs..., impl=:raw)
            @test MHD.fields_to_host(auto) == before
        end
    end
end

function _fft_field(n)
    Float32[sinpi(2(i-1)/n) + 0.3cospi(2(j-1)/n) + 0.1sinpi(2(k-1)/n)
            for i in 1:n, j in 1:n, k in 1:n]
end

function _test_fft(name)
    be = MHD.backend(name)
    for n in (12,16)
        @testset "FFT size $n" begin
            host = _fft_field(n)
            cpu_hat = zeros(ComplexF32, n÷2+1,n,n)
            MHD._radiation_forward!(cpu_hat,host)
            cpu_plans = MHD._radiation_fft_plans(host,cpu_hat)
            x = MHD.to_device(be,host)
            chat = MHD.device_zeros(be,ComplexF32,size(cpu_hat))
            @test MHD._radiation_fft_plan_key(host,cpu_hat) ==
                  MHD._radiation_fft_plan_key(copy(host),copy(cpu_hat))
            if name !== :cpu
                @test MHD._radiation_fft_plan_key(host,cpu_hat) !=
                      MHD._radiation_fft_plan_key(x,chat)
            end
            MHD._radiation_forward!(chat,x)
            @test MHD.to_host(chat) ≈ cpu_hat rtol=2f-5 atol=2f-4
            MHD._radiation_inverse!(x,chat)
            @test maximum(abs, MHD.to_host(x) .- host) < 3f-6
            # Revisit the same CPU shape after device planning/execution.
            @test MHD._radiation_fft_plans(host,cpu_hat) === cpu_plans
            MHD._radiation_forward!(cpu_hat,host)
            restored = similar(host)
            MHD._radiation_inverse!(restored,cpu_hat)
            @test maximum(abs, restored .- host) < 3f-6
            x64 = Float64.(host)
            h64 = zeros(ComplexF64,size(cpu_hat))
            @test MHD._radiation_fft_plans(x64,h64) !== cpu_plans
            @test_throws DimensionMismatch MHD._radiation_fft_plans(host,zeros(ComplexF32,n,n,n))

            for batch in (2,4)
                input = cat((Float32(d) .* host .+ Float32(d-1) for d in 1:batch)...; dims=4)
                real_batch = MHD.to_device(be,input)
                hat_batch = MHD.device_zeros(be,ComplexF32,(n÷2+1,n,n,batch))
                MHD._radiation_forward_batch!(hat_batch,real_batch)
                expected = FFTW.rfft(input,(1,2,3))
                @test MHD.to_host(hat_batch) ≈ expected rtol=2f-5 atol=2f-3
                if name === :cuda
                    forward, inverse = MHD._radiation_fft_plans(real_batch,hat_batch)
                    @test forward isa CUDA.CUFFT.CuFFTPlan
                    @test inverse.p isa CUDA.CUFFT.CuFFTPlan
                    @test forward.region == (1,2,3)
                    @test size(forward) == size(real_batch)
                end
                MHD._radiation_inverse_batch!(real_batch,hat_batch)
                @test maximum(abs, MHD.to_host(real_batch) .- input) < 2f-5
            end
            if name === :cuda
                forward, inverse = MHD._radiation_fft_plans(x,chat)
                @test forward isa CUDA.CUFFT.CuFFTPlan
                @test inverse.p isa CUDA.CUFFT.CuFFTPlan
            end
        end
    end
end

function _test_poisson(name)
    n = 16
    host = _fft_field(n)
    reference = similar(host)
    Poisson.fft_poisson_rfft!(reference,host; G=1, a=1, boxsize=1)
    be = Poisson.backend(name)
    rho = Poisson.to_device(be,host)
    phi = Poisson.device_zeros(be,Float32,size(host))
    Poisson.fft_poisson_rfft!(phi,rho; G=1, a=1, boxsize=1)
    result = Poisson.to_host(phi)
    @test all(isfinite,result)
    @test maximum(abs, result .- reference) < 2f-6
    @test maximum(abs, result .+ host ./ Float32(4pi^2)) < 2f-6
    @test abs(sum(Float64,result)/length(result)) < 1e-7
end

function _test_photons(name)
    n = 16
    fields = _smooth_fields(n)
    for model in (:moments, :moments_exponential)
        @testset "$model" begin
            cpu = _state(:cpu,fields,n)
            device = _state(name,fields,n)
            wc = MHD.allocate_radiation_workspace(cpu)
            wd = MHD.allocate_radiation_workspace(device)
            pc = MHD.allocate_photon_moment_state(cpu)
            pd = MHD.allocate_photon_moment_state(device)
            MHD.initialize_photon_moments!(pc,wc,cpu)
            MHD.initialize_photon_moments!(pd,wd,device)
            initial_photons = MHD.to_host(pc.hat_batch)
            @test maximum(abs,initial_photons) > 0f0
            closure = MHD.RadiationClosure(inertia_ratio=0.1f0,mean_free_path=0.01f0,
                photon_speed=10f0,gas_sound_speed=1f0,response_model=model)
            lc = MHD.RadiationEnergyLedger()
            ld = MHD.RadiationEnergyLedger()
            dt, speed = MHD.compute_dt(cpu;cfl=0.02)
            for _ in 1:2
                MHD.step_radiation_godunov!(cpu,wc,dt;drag_impulse=0.5f0,closure,
                    ch=speed,integrator=:ref,photon_state=pc,energy_ledger=lc)
                MHD.step_radiation_godunov!(device,wd,dt;drag_impulse=0.5f0,closure,
                    ch=speed,integrator=name === :cpu ? :ref : :cube,
                    photon_state=pd,energy_ledger=ld)
            end
            host_cpu = MHD.fields_to_host(cpu)
            host_device = MHD.fields_to_host(device)
            _assert_fields(host_cpu,host_device)
            @test abs(_mass(host_device)/_mass(fields)-1) < 3e-6
            photons_cpu = MHD.to_host(pc.hat_batch)
            photons_device = MHD.to_host(pd.hat_batch)
            @test all(isfinite,photons_device)
            @test maximum(abs,photons_device .- photons_cpu) <=
                  8f-5 * maximum(abs,photons_cpu) + 2f-5
            @test maximum(abs,photons_device .- initial_photons) > 1f-5
            @test maximum(abs,MHD.to_host(wd.density_perturbation) .-
                              MHD.to_host(wc.density_perturbation)) < 3f-6
            @test lc.samples == ld.samples == 2
            @test isfinite(ld.resolved_coupled_change)
            # This is a small difference between Float32 background energies,
            # not a relative-error observable. Budget one background ULP per
            # sampled step, while retaining the tighter pointwise state gates.
            energy_scale = sum(abs,Float64.(fields[5]))
            energy_roundoff = lc.samples * eps(Float32) * energy_scale
            @test abs(ld.total_change - lc.total_change) <= energy_roundoff
            for (ledger, state) in ((lc,host_cpu),(ld,host_device))
                measured_change = sum(Float64.(state[5]) .- Float64.(fields[5]))
                @test abs(ledger.total_change - measured_change) <= energy_roundoff
            end
        end
    end
end

function _chemistry_inputs(z; helium=false)
    n = 32
    rho = Float32[1 + 0.1sinpi(2(i-1)/n) for i in 1:n]
    xhii = helium ? 0.99f0 : 0.6f0
    helium_mass = helium ? 0.12f0 : 0f0
    temperature = Float32(2.725 * (1+z))
    particles = 0.76f0 * (1+xhii) + 0.24f0/4 + helium_mass/4
    energy = Float32(1.5 * Chemistry.KBOLTZ / Chemistry.MH / 1e16) *
             temperature * particles
    (rho=rho, energy=fill(energy,n), hii=rho .* (0.76f0*xhii),
     h2=rho .* 7.6f-9, heii=rho .* helium_mass,
     neutral=fill(0.76f0*(1-xhii),n), xhii=xhii)
end

function _assert_chemistry(reference, candidate, rho)
    for (expected, actual) in zip(reference,candidate)
        @test all(isfinite,actual)
        @test minimum(actual) >= 0f0
        @test maximum(abs,actual .- expected) <=
              5f-4 * max(maximum(abs,expected),1f-25)
    end
    @test all(candidate[2] .+ candidate[3] .<= 0.76f0 .* rho .+ 2f-6)
    @test minimum(candidate[1]) > 0f0
end

function _test_chemistry(name)
    tables_cpu = Chemistry.build_rate_tables(;backend=:cpu,precision=Float32)
    be = Chemistry.backend(name)
    tables_device = Chemistry.RateTables(
        Chemistry.to_device(be,tables_cpu.logk,Float32),
        tables_cpu.x0,tables_cpu.invdx,tables_cpu.N)

    @testset "Zero-copy full H/H2 mixing network" begin
        z = 1100.0
        inputs = _chemistry_inputs(z)
        density_unit = 1.8788e-29 * 0.674^2 * 0.049 * (1+z)^3
        reference = (copy(inputs.energy),copy(inputs.hii),copy(inputs.h2))
        candidate = map(x -> Chemistry.to_device(be,x,Float32),reference)
        rho_device = Chemistry.to_device(be,inputs.rho,Float32)
        smooth_device = Chemistry.to_device(be,inputs.neutral,Float32)
        kwargs = (a_value=1/(1+z),dt=1e10,density_units=density_unit,
                  length_units=1e8,time_units=1,f_alpha=1,Xe_mean=inputs.xhii,
                  smoothed_is_neutral=true,recfast_hswitch=true,
                  hubble_expansion=true,hubble=67.4,Om=0.315,OL=0.685,
                  fh=0.76,dtfrac=0.1,itcap=1024,precision=Float32)
        Chemistry.solve_chem_mixing_device!(inputs.rho,reference...,inputs.neutral;
            kwargs...,rate_tables=tables_cpu,backend=:cpu)
        Chemistry.solve_chem_mixing_device!(rho_device,candidate...,smooth_device;
            kwargs...,rate_tables=tables_device,backend=name)
        host = map(Chemistry.to_host,candidate)
        _assert_chemistry(reference,host,inputs.rho)
        @test maximum(abs,host[2] .- inputs.hii) > 1f-7
        @test Chemistry.to_host(rho_device) == inputs.rho
    end

    @testset "Full H/He mixing and cooling network" begin
        z = 2200.0
        inputs = _chemistry_inputs(z;helium=true)
        density_unit = 1.8788e-29 * 0.674^2 * 0.049 * (1+z)^3
        reference = map(copy,(inputs.energy,inputs.hii,inputs.h2,inputs.heii))
        candidate = map(copy,reference)
        kwargs = (a_value=1/(1+z),dt=1e10,density_units=density_unit,
                  length_units=1e8,time_units=1,
                  fa_table=Chemistry.FAlphaTable([0.0,1e4],[1.0,1.0]),
                  Xe_mean=inputs.xhii,smoothed_is_neutral=true,
                  recfast_hswitch=true,hubble_expansion=true,
                  hubble=67.4,Om=0.315,OL=0.685,fh=0.76,
                  helium=true,dtfrac=0.1,itcap=1024,precision=Float32)
        # Both public launchers stage these exact host inputs and execute the
        # shared KA network on the explicitly requested backend, including HeI.
        Chemistry.solve_chem_mixing!(inputs.rho,reference[1],reference[2],reference[3],
            inputs.neutral;kwargs...,HeII=reference[4],rate_tables=tables_cpu,backend=:cpu)
        Chemistry.solve_chem_mixing!(inputs.rho,candidate[1],candidate[2],candidate[3],
            inputs.neutral;kwargs...,HeII=candidate[4],rate_tables=tables_device,backend=name)
        _assert_chemistry(reference,candidate,inputs.rho)
        @test all(candidate[4] .<= 0.24f0 .* inputs.rho .+ 2f-6)
        @test maximum(abs,candidate[4] .- inputs.heii) > 1f-7
    end
end

# Constrained transport runs the :ref integrator on every backend (the CT kernel packs
# its arrays and scalars to fit Metal's 31 kernel arguments).
function _test_ct_mhd(name)
    n = 16
    fields = _smooth_fields(n)
    initial_mass = _mass(fields)
    for riemann in (:hlld, :hll)
        @testset "ct / plm / $riemann" begin
            states = map((:cpu, name)) do be
                s = MHD.allocate_state(MHD.backend(be), Float32, (n,n,n); dx=1/n,
                                       gamma=5/3, recon=:plm, riemann, mhd_solver=:ct)
                for v in 1:9
                    copyto!(s.U[v], fields[v])
                end
                MHD.ct_faces_from_cells!(s)
                s
            end
            cpu, device = states
            dt, _ = MHD.compute_dt(cpu; cfl=0.08)
            for s in states
                for _ in 1:3
                    MHD.step!(s, dt)
                end
                MHD.step_drag_godunov!(s, dt; drag_impulse=0.2, ch=0, integrator=:ref)
            end
            host_cpu = MHD.fields_to_host(cpu)
            host_device = MHD.fields_to_host(device)
            _assert_fields(host_cpu, host_device)
            for f in (:bxf, :byf, :bzf)
                a = MHD.to_host(getfield(cpu.ct, f)); b = MHD.to_host(getfield(device.ct, f))
                @test maximum(abs, b .- a) <= 2f-6 + 5f-5 * maximum(abs, a)
            end
            brms = sqrt(sum(v -> sum(abs2, host_device[v]), 6:8) / n^3)
            @test MHD.max_divb_face(device) * Float64(device.dx) / brms < 1e-5
            @test abs(_mass(host_device) / initial_mass - 1) < 3e-6
            # compensated face updates (radiation-coupled steps with track_magnetic) run on the
            # device and agree with the CPU, remainders included
            comp = map((cpu, device)) do s
                c = MHD.device_zeros(s.be, Float32, (9n^3,))
                for _ in 1:3
                    MHD.step_ref_radiation!(s, dt; ch=0f0, decay=1f0, drag_impulse=0.2f0,
                        correction=c, track_density=false, track_magnetic=true,
                        track_energy=false)
                end
                s
            end
            @test any(!iszero, MHD.to_host(comp[2].ct.rbx))
            for f in (:bxf, :byf, :bzf, :rbx, :rby, :rbz)
                a = MHD.to_host(getfield(cpu.ct, f)); b = MHD.to_host(getfield(device.ct, f))
                @test maximum(abs, b .- a) <= 2f-6 + 5f-5 * maximum(abs, a)
            end
            @test MHD.max_divb_face(device) * Float64(device.dx) / brms < 1e-5
        end
    end
end

include("streaming_checks.jl")
include("relative_velocity_checks.jl")

function _run_backend_tests(name)
    name === :cuda && CUDA.allowscalar(false)
    @testset "PMFGLM $name Float32 backend" begin
        @testset "Smooth MHD and mass conservation" _test_smooth_mhd(name)
        @testset "Constrained-transport MHD" _test_ct_mhd(name)
        @testset "Native FFT roundtrip and cache isolation" _test_fft(name)
        @testset "Periodic Poisson" _test_poisson(name)
        @testset "Streaming and planar invariance" streaming_checks(name,Float32)
        @testset "Eulerian particle velocities and photon expansion" relative_velocity_checks(name,Float32)
        @testset "Coupled photon moments" _test_photons(name)
        @testset "Production chemistry and cooling" _test_chemistry(name)
    end
end

function run_backend_tests(name::Symbol=:cpu)
    _import_backend(name)
    # Check bindings as well as extension methods in the post-import world.
    Base.invokelatest(_run_checked_backend_tests,name)
end

function _run_checked_backend_tests(name)
    _require_backend(name)
    _run_backend_tests(name)
end

end # module
