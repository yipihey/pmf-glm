const PK = PMFGLM.PoissonKernels
for T in (Float32,Float64), green in (:spectral,:centered)
    n = 16
    phi_exact = T[sinpi(2*(i-1)/n) for i in 1:n,j in 1:n,k in 1:n]
    k2 = green == :centered ? (n*sin(2pi/n))^2 : (2pi)^2
    rho = -T(k2).*phi_exact
    phi = zeros(T,n,n,n)
    PK.fft_poisson_rfft!(phi,rho;G=1,a=1,boxsize=1,greens=green)
    @test maximum(abs,phi-phi_exact) < (T==Float32 ? 5e-6 : 1e-12)
    alias = copy(rho)
    PK.fft_poisson_rfft!(alias,alias;G=1,a=1,boxsize=1,greens=green)
    @test maximum(abs,alias-phi) < (T==Float32 ? 5e-6 : 1e-12)
    @test abs(mean(phi)) < 1e-7
end

# A uniform DM lattice must deposit uniform density without seeding power.
n = 8
positions = [(Float32((i-0.5)/n),Float32((j-0.5)/n),Float32((k-0.5)/n))
             for i in 1:n,j in 1:n,k in 1:n]
px,py,pz = (vec(getindex.(positions,d)) for d in 1:3)
rho = zeros(Float32,n^3)
PK.cic_deposit!(rho,px,py,pz,px,py,pz,1f0;N=n)
KernelAbstractions.synchronize(PK.backend(:cpu))
@test sum(rho) ≈ Float32(n^3)
@test maximum(abs,rho.-1f0) < 1e-6

# Periodicity and total mass also hold across opposite corners.
PK.cic_deposit!(rho,Float32[0,1],Float32[0,1],Float32[0,1],
                zeros(Float32,2),zeros(Float32,2),zeros(Float32,2),1f0;N=n)
KernelAbstractions.synchronize(PK.backend(:cpu))
@test sum(rho) ≈ 2f0
@test minimum(rho) >= 0

for T in (Float32,Float64)
    n = 16
    wave = T[1+0.1cospi(2*(i-1)/n) for i in 1:n,j in 1:n,k in 1:n]
    smoothed = similar(wave)
    sigma = 2.0
    PK.rfft_smooth_gaussian!(smoothed,wave;sigma_cells=sigma)
    expected = 1 .+ (wave .- 1).*exp(-0.5*(sigma*2pi/n)^2)
    @test maximum(abs,smoothed-expected) < (T == Float32 ? 5e-7 : 1e-12)
    @test mean(smoothed) ≈ mean(wave)
    aliased = copy(wave)
    PK.rfft_smooth_gaussian!(aliased,aliased;sigma_cells=sigma)
    @test aliased ≈ smoothed
    @test_throws ArgumentError PK.rfft_smooth_gaussian!(smoothed,wave;sigma_cells=-1)
end
