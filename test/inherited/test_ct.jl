using Random

# Face-staggered constrained-transport (CT) solver tests (mhd_solver=:ct).
# Provenance of the scheme: Solaceonx/GLMMHDTurb `ct_mhd.jl` (sweep-fixes, commits 9d20e52
# and 7256715; Gardiner & Stone 2005, J. Comput. Phys. 205, 509).

# ── (a) independent plain-Julia CPU port of the fork's ct_mhd.jl step ─────────────
# Copied function by function (moncen/pslope/hanc3d/dflux/hll/gs05/ecc/emf/faceB/bcc);
# CUDA thread indexing replaced by loops and the fork's Float32 constants by Float64.
const FK_SMALLR = 1e-6
const FK_SMALLP = 1e-6
fk_w(i, N) = mod1(i, N)
function fk_moncen(dl, dr)
    dc = 0.5*(dl+dr); s = ifelse(dc >= 0.0, 1.0, -1.0)
    ifelse(dl*dr <= 0.0, 0.0, s*min(2.0*min(abs(dl), abs(dr)), abs(dc)))
end
function fk_cellprim(U, bcc, i, j, k, gamma)
    rho = max(U[i,j,k,1], FK_SMALLR); ir = 1.0/rho
    vx = U[i,j,k,2]*ir; vy = U[i,j,k,3]*ir; vz = U[i,j,k,4]*ir
    bx = bcc[1][i,j,k]; by = bcc[2][i,j,k]; bz = bcc[3][i,j,k]
    p = max((gamma-1.0)*(U[i,j,k,5]-0.5*rho*(vx*vx+vy*vy+vz*vz)-0.5*(bx*bx+by*by+bz*bz)), FK_SMALLP)
    (rho, vx, vy, vz, p, bx, by, bz)
end
fk_psub(a, b, c) = ntuple(t -> a[t]+c*b[t], 8)
fk_pslope(L, M, R) = ntuple(t -> fk_moncen(M[t]-L[t], R[t]-M[t]), 8)
function fk_dflux(q, d, gamma)
    rho, vx, vy, vz, p, bx, by, bz = q
    b2 = bx*bx+by*by+bz*bz; ptot = p+0.5*b2
    E = p/(gamma-1.0)+0.5*rho*(vx*vx+vy*vy+vz*vz)+0.5*b2
    vb = vx*bx+vy*by+vz*bz
    un = d == 1 ? vx : d == 2 ? vy : vz
    bn = d == 1 ? bx : d == 2 ? by : bz
    Fr = rho*un
    Fmx = rho*un*vx - bn*bx + (d == 1 ? ptot : 0.0)
    Fmy = rho*un*vy - bn*by + (d == 2 ? ptot : 0.0)
    Fmz = rho*un*vz - bn*bz + (d == 3 ? ptot : 0.0)
    FE = (E+ptot)*un - bn*vb
    Fbx = un*bx - (d == 1 ? un*bx : vx*bn)
    Fby = un*by - (d == 2 ? un*by : vy*bn)
    Fbz = un*bz - (d == 3 ? un*bz : vz*bn)
    (Fr, Fmx, Fmy, Fmz, FE, Fbx, Fby, Fbz)
end
function fk_fast_speed(q, d, gamma)
    rho = q[1]; bn = d == 1 ? q[6] : d == 2 ? q[7] : q[8]
    c2 = gamma*q[5]/rho; b2 = (q[6]*q[6]+q[7]*q[7]+q[8]*q[8])/rho; dd = 0.5*(b2+c2)
    sqrt(dd+sqrt(max(dd*dd-c2*bn*bn/rho, 0.0)))
end
function fk_toC(q, gamma)
    rho, vx, vy, vz, p, bx, by, bz = q
    (rho, rho*vx, rho*vy, rho*vz, p/(gamma-1.0)+0.5*rho*(vx*vx+vy*vy+vz*vz)+0.5*(bx*bx+by*by+bz*bz), bx, by, bz)
end
function fk_hll(L, R, d, gamma)
    un(q) = (d == 1 ? q[2] : d == 2 ? q[3] : q[4])
    cfL = fk_fast_speed(L, d, gamma); cfR = fk_fast_speed(R, d, gamma)
    SL = min(min(un(L)-cfL, un(R)-cfR), 0.0); SR = max(max(un(L)+cfL, un(R)+cfR), 0.0)
    FL = fk_dflux(L, d, gamma); FR = fk_dflux(R, d, gamma)
    UL = fk_toC(L, gamma); UR = fk_toC(R, gamma); ih = 1.0/(SR-SL)
    ntuple(t -> (SR*FL[t]-SL*FR[t]+SL*SR*(UR[t]-UL[t]))*ih, 8)
end
function fk_halfprim(Uh, gamma)
    rho = max(Uh[1], FK_SMALLR); ir = 1.0/rho
    vx = Uh[2]*ir; vy = Uh[3]*ir; vz = Uh[4]*ir; bx = Uh[6]; by = Uh[7]; bz = Uh[8]
    p = max((gamma-1.0)*(Uh[5]-0.5*rho*(vx*vx+vy*vy+vz*vz)-0.5*(bx*bx+by*by+bz*bz)), FK_SMALLP)
    (rho, vx, vy, vz, p, bx, by, bz)
end
function fk_hanc3d(U, bcc, i, j, k, d, dtdx, N, gamma)
    cp(a, b, c) = fk_cellprim(U, bcc, a, b, c, gamma)
    q = cp(i, j, k)
    sx = fk_pslope(cp(fk_w(i-1, N), j, k), q, cp(fk_w(i+1, N), j, k))
    sy = fk_pslope(cp(i, fk_w(j-1, N), k), q, cp(i, fk_w(j+1, N), k))
    sz = fk_pslope(cp(i, j, fk_w(k-1, N)), q, cp(i, j, fk_w(k+1, N)))
    fxp = fk_dflux(fk_psub(q, sx, 0.5), 1, gamma); fxm = fk_dflux(fk_psub(q, sx, -0.5), 1, gamma)
    fyp = fk_dflux(fk_psub(q, sy, 0.5), 2, gamma); fym = fk_dflux(fk_psub(q, sy, -0.5), 2, gamma)
    fzp = fk_dflux(fk_psub(q, sz, 0.5), 3, gamma); fzm = fk_dflux(fk_psub(q, sz, -0.5), 3, gamma)
    C = fk_toC(q, gamma); h = 0.5*dtdx
    Uh = ntuple(t -> C[t]-h*((fxp[t]-fxm[t])+(fyp[t]-fym[t])+(fzp[t]-fzm[t])), 8)
    (fk_halfprim(Uh, gamma), d == 1 ? sx : d == 2 ? sy : sz)
end
function fk_gs05(Ea0, Eam, Eb0, Ebm, mA0, mAm, mB0, mBm, Er00, Erm0, Er0m, Ermm)
    up(m, lower, upper) = ifelse(m > 0.0, lower, ifelse(m < 0.0, upper, 0.5*(lower+upper)))
    A1 = up(mA0, Erm0-Ebm, Er00-Eb0)
    A3 = up(mAm, Ebm-Ermm, Eb0-Er0m)
    B1 = up(mB0, Er0m-Eam, Er00-Ea0)
    B3 = up(mBm, Eam-Ermm, Ea0-Erm0)
    0.25*((Ea0+Eam+Eb0+Ebm)+(A3-A1)+(B3-B1))
end

# one fork CT step (fixed dt); U is (N,N,N,5), faces are (N,N,N) lower-face arrays
function fk_step!(U, bxf, byf, bzf, dt, dx, N, gamma)
    dtdx = dt/dx
    bcc = (zeros(N,N,N), zeros(N,N,N), zeros(N,N,N))
    for k in 1:N, j in 1:N, i in 1:N
        bcc[1][i,j,k] = 0.5*(bxf[i,j,k]+bxf[fk_w(i+1,N),j,k])
        bcc[2][i,j,k] = 0.5*(byf[i,j,k]+byf[i,fk_w(j+1,N),k])
        bcc[3][i,j,k] = 0.5*(bzf[i,j,k]+bzf[i,j,fk_w(k+1,N)])
    end
    F = ntuple(_ -> zeros(N,N,N,8), 3)
    bnf = (bxf, byf, bzf)
    for d in 1:3, k in 1:N, j in 1:N, i in 1:N
        o = (d == 1 ? 1 : 0, d == 2 ? 1 : 0, d == 3 ? 1 : 0)
        mhL, sL = fk_hanc3d(U, bcc, fk_w(i-o[1],N), fk_w(j-o[2],N), fk_w(k-o[3],N), d, dtdx, N, gamma)
        mhR, sR = fk_hanc3d(U, bcc, i, j, k, d, dtdx, N, gamma)
        L = fk_psub(mhL, sL, 0.5); R = fk_psub(mhR, sR, -0.5)
        bn = bnf[d][i,j,k]
        L = (max(L[1],FK_SMALLR),L[2],L[3],L[4],max(L[5],FK_SMALLP), d==1 ? bn : L[6], d==2 ? bn : L[7], d==3 ? bn : L[8])
        R = (max(R[1],FK_SMALLR),R[2],R[3],R[4],max(R[5],FK_SMALLP), d==1 ? bn : R[6], d==2 ? bn : R[7], d==3 ? bn : R[8])
        f = fk_hll(L, R, d, gamma)
        for t in 1:8; F[d][i,j,k,t] = f[t]; end
    end
    Fx, Fy, Fz = F
    ec = (zeros(N,N,N), zeros(N,N,N), zeros(N,N,N))
    for k in 1:N, j in 1:N, i in 1:N
        mh, _ = fk_hanc3d(U, bcc, i, j, k, 1, dtdx, N, gamma)
        vx = mh[2]; vy = mh[3]; vz = mh[4]; bx = bcc[1][i,j,k]; by = bcc[2][i,j,k]; bz = bcc[3][i,j,k]
        ec[1][i,j,k] = vz*by-vy*bz; ec[2][i,j,k] = vx*bz-vz*bx; ec[3][i,j,k] = vy*bx-vx*by
    end
    ex = zeros(N,N,N); ey = zeros(N,N,N); ez = zeros(N,N,N)
    E(a,b,c,comp) = ec[comp][a,b,c]
    for k in 1:N, j in 1:N, ii in 1:N
        im = fk_w(ii-1,N); jm = fk_w(j-1,N); km = fk_w(k-1,N)
        ez[ii,j,k] = fk_gs05(-Fx[ii,j,k,7],-Fx[ii,jm,k,7],Fy[ii,j,k,6],Fy[im,j,k,6],
                             Fx[ii,j,k,1],Fx[ii,jm,k,1],Fy[ii,j,k,1],Fy[im,j,k,1],
                             E(ii,j,k,3),E(im,j,k,3),E(ii,jm,k,3),E(im,jm,k,3))
        ex[ii,j,k] = fk_gs05(-Fy[ii,j,k,8],-Fy[ii,j,km,8],Fz[ii,j,k,7],Fz[ii,jm,k,7],
                             Fy[ii,j,k,1],Fy[ii,j,km,1],Fz[ii,j,k,1],Fz[ii,jm,k,1],
                             E(ii,j,k,1),E(ii,jm,k,1),E(ii,j,km,1),E(ii,jm,km,1))
        ey[ii,j,k] = fk_gs05(-Fz[ii,j,k,6],-Fz[im,j,k,6],Fx[ii,j,k,8],Fx[ii,j,km,8],
                             Fz[ii,j,k,1],Fz[im,j,k,1],Fx[ii,j,k,1],Fx[ii,j,km,1],
                             E(ii,j,k,2),E(ii,j,km,2),E(im,j,k,2),E(im,j,km,2))
    end
    for k in 1:N, j in 1:N, ii in 1:N
        ip = fk_w(ii+1,N); jp = fk_w(j+1,N); kp = fk_w(k+1,N)
        for t in 1:5
            U[ii,j,k,t] += dtdx*((Fx[ii,j,k,t]-Fx[ip,j,k,t])+(Fy[ii,j,k,t]-Fy[ii,jp,k,t])+(Fz[ii,j,k,t]-Fz[ii,j,kp,t]))
        end
    end
    for k in 1:N, j in 1:N, ii in 1:N
        ip = fk_w(ii+1,N); jp = fk_w(j+1,N); kp = fk_w(k+1,N)
        bxf[ii,j,k] -= dtdx*((ez[ii,jp,k]-ez[ii,j,k])-(ey[ii,j,kp]-ey[ii,j,k]))
        byf[ii,j,k] -= dtdx*((ex[ii,j,kp]-ex[ii,j,k])-(ez[ip,j,k]-ez[ii,j,k]))
        bzf[ii,j,k] -= dtdx*((ey[ip,j,k]-ey[ii,j,k])-(ex[ii,jp,k]-ex[ii,j,k]))
    end
    nothing
end

# edge vector potential of a few Fourier modes, evaluated at the edge positions
function ct_test_potential(N; amp=0.15)
    dx = 1/N; tp = 2π
    A = (zeros(N,N,N), zeros(N,N,N), zeros(N,N,N))
    for k in 1:N, j in 1:N, i in 1:N
        xc = (i-0.5)*dx; xe = (i-1)*dx
        yc = (j-0.5)*dx; ye = (j-1)*dx
        zc = (k-0.5)*dx; ze = (k-1)*dx
        # Ax at (xc, ye, ze), Ay at (xe, yc, ze), Az at (xe, ye, zc)
        A[1][i,j,k] = amp*(sin(tp*ye+0.3)*cos(tp*ze) + 0.5*sin(tp*(ye+ze)+1.0)/tp)/tp
        A[2][i,j,k] = amp*(sin(tp*ze+1.1)*cos(tp*xe+0.4) + 0.4*cos(tp*(xe-ze))/tp)/tp
        A[3][i,j,k] = amp*(sin(tp*xe+0.7)*cos(tp*ye+0.2) + 0.3*sin(tp*(xe+ye)+0.5)/tp)/tp
    end
    A
end

function ct_test_hydro!(s, N, gamma)
    tp = 2π; dx = 1/N
    for k in 1:N, j in 1:N, i in 1:N
        x = (i-0.5)*dx; y = (j-0.5)*dx; z = (k-0.5)*dx
        rho = 1 + 0.2*sin(tp*x+0.3)*cos(tp*y) + 0.1*sin(tp*(y+z)+0.9)
        vx = 0.3*sin(tp*y+0.1) + 0.1*cos(tp*(x+z))
        vy = 0.3*sin(tp*z+0.8) + 0.1*sin(tp*(x-y)+0.2)
        vz = 0.3*sin(tp*x+0.5) + 0.1*cos(tp*(y-z)+0.6)
        p  = 1 + 0.15*cos(tp*(x+y)+0.4) + 0.1*sin(tp*z)
        c = linidx(N, N, i, j, k)
        s.U[1][c] = rho; s.U[2][c] = rho*vx; s.U[3][c] = rho*vy; s.U[4][c] = rho*vz
        s.U[5][c] = p/(gamma-1) + 0.5*rho*(vx*vx+vy*vy+vz*vz)   # magnetic part added by the potential init
        s.U[6][c] = 0; s.U[7][c] = 0; s.U[8][c] = 0; s.U[9][c] = 0
    end
end

@testset "constrained transport (mhd_solver=:ct)" begin

@testset "allocation, accessors, restrictions" begin
    g = allocate_state(backend(:cpu), Float64, (8,8,8); dx=1/8)
    c = allocate_state(backend(:cpu), Float64, (8,8,8); dx=1/8, mhd_solver=:ct)
    @test mhd_solver(g) === :glm && !uses_ct(g) && g.ct === nothing
    @test mhd_solver(c) === :ct && uses_ct(c) && c.ct isa CTState
    @test length(fieldnames(CTState)) == 18
    @test_throws ErrorException allocate_state(backend(:cpu), Float64, (8,8,8); dx=1/8, mhd_solver=:bogus)
    @test_throws ErrorException allocate_state(backend(:cpu), Float64, (8,8,8); dx=1/8,
                                               mhd_solver=:ct, bcs=(:outflow,:periodic,:periodic))
    try
        allocate_state(backend(:cpu), Float64, (8,8,8); dx=1/8, mhd_solver=:ct, bc=:reflecting)
    catch e
        @test occursin("periodic", sprint(showerror, e))
    end
    # cube integrators error clearly for CT (before any grid-size check)
    c4 = allocate_state(backend(:cpu), Float32, (8,8,8); dx=1/8, mhd_solver=:ct)
    for f in (() -> step!(c4, 1f-3; ch=1f0, integrator=:cube),
              () -> step_cube!(c4, 1f-3; ch=1f0, decay=1f0),
              () -> step_cube_drag!(c4, 1f-3; ch=1f0, decay=1f0, drag_impulse=0.1),
              () -> step_drag_godunov!(c4, 1f-3; drag_impulse=0.1, ch=1f0, integrator=:cube),
              () -> step_cube_radiation!(c4, 1f-3; ch=1f0, decay=1f0, drag_impulse=0.1, correction=c4.U[9]))
        e = try f(); nothing catch err; err end
        @test e isa ErrorException
        @test occursin("CT is implemented on the :ref integrator only", sprint(showerror, e))
    end
    # GLM step still demands ch
    @test_throws ErrorException step!(g, 1e-3)
end

@testset "matches the plain-Julia port of the fork's ct_mhd.jl (HLL, PLM, Float64)" begin
    N = 12; gamma = 5/3
    s = allocate_state(backend(:cpu), Float64, (N,N,N); dx=1/N, gamma=gamma, riemann=:hll,
                       recon=:plm, mhd_solver=:ct, smallr=FK_SMALLR, pfl=FK_SMALLP)
    ct_test_hydro!(s, N, gamma)
    A = ct_test_potential(N)
    ct_faces_from_potential!(s, A[1], A[2], A[3])
    @test max_divb_face(s) * s.dx < 1e-13
    # fork state from the same device data
    U = zeros(N,N,N,5)
    for t in 1:5; U[:,:,:,t] .= reshape(copy(s.U[t]), N, N, N); end
    bxf = reshape(copy(s.ct.bxf), N, N, N); byf = reshape(copy(s.ct.byf), N, N, N)
    bzf = reshape(copy(s.ct.bzf), N, N, N)
    bxf0 = copy(bxf)
    dt, smax = compute_dt(s; cfl=0.3)
    nsteps = 5
    for n in 1:nsteps
        step!(s, dt)
        fk_step!(U, bxf, byf, bzf, dt, 1/N, N, gamma)
    end
    rel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())
    errs = [rel(reshape(copy(s.U[t]), N, N, N), U[:,:,:,t]) for t in 1:5]
    ef = [rel(reshape(copy(s.ct.bxf), N, N, N), bxf), rel(reshape(copy(s.ct.byf), N, N, N), byf),
          rel(reshape(copy(s.ct.bzf), N, N, N), bzf)]
    @printf("  fork-port cross-check (%d steps, N=%d, dt=%.4e): rel max |diff|  rho=%.2e mx=%.2e my=%.2e mz=%.2e E=%.2e  bxf=%.2e byf=%.2e bzf=%.2e\n",
            nsteps, N, dt, errs..., ef...)
    @test minimum(reshape(copy(s.U[1]), N, N, N)) > 10*FK_SMALLR
    @test all(<(1e-10), errs)
    @test all(<(1e-10), ef)
    # state genuinely evolved (not a trivial agreement)
    @test rel(reshape(copy(s.ct.bxf), N, N, N), bxf0) > 1e-3
    @test max_divb_face(s) * s.dx < 1e-12
end

@testset "faces from cells: divergence-free, Nyquist planes, small cell change" begin
    for (dims, T) in (((16,16,16), Float64), ((12,16,8), Float64), ((15,9,10), Float64), ((16,16,16), Float32))
        s = allocate_state(backend(:cpu), T, dims; dx=1/maximum(dims), mhd_solver=:ct)
        Nx, Ny, Nz = dims
        # a field with power in every mode including the Nyquist planes
        rng = MersenneTwister(7)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            c = linidx(Nx, Ny, i, j, k)
            s.U[1][c] = 1; s.U[5][c] = 1.5
            s.U[6][c] = 0.3 + 0.1*sin(2π*j/Ny)
            s.U[7][c] = 0.1*cos(2π*k/Nz + 0.3)
            s.U[8][c] = 0.1*sin(2π*i/Nx + 0.1)
        end
        for v in 6:8
            h = Array(s.U[v]); h .+= T(0.05) .* (T.(rand(rng, length(h))) .- T(0.5))   # white noise
            for k in 1:Nz, j in 1:Ny, i in 1:Nx       # (-1)^(i+j+k) pure-Nyquist component
                h[linidx(Nx, Ny, i, j, k)] += T(0.02) * (-1)^(i+j+k)
            end
            copyto!(s.U[v], h)
        end
        info = ct_faces_from_cells!(s)
        tol = T === Float64 ? 1e-13 : 1e-5
        @printf("  faces_from_cells %s %s: cell-B rel L2 change=%.3e, max div_face*dx/Brms=%.3e\n",
                T, dims, info.cell_b_rel_l2_change, info.max_divb_face_dx_over_brms)
        @test info.max_divb_face_dx_over_brms < tol
        @test info.cell_b_rel_l2_change < 0.3      # white noise + Nyquist content is projected
        @test all(iszero, Array(s.U[9]))
    end
    # smooth field: the cell change is small and the pressure is preserved
    N = 24; s = allocate_state(backend(:cpu), Float64, (N,N,N); dx=1/N, gamma=5/3, mhd_solver=:ct)
    init_turb_field!(s)
    h0 = fields_to_host(s)
    p0(h) = (5/3-1) .* (h[5] .- 0.5 .* (h[2].^2 .+ h[3].^2 .+ h[4].^2) ./ h[1] .- 0.5 .* (h[6].^2 .+ h[7].^2 .+ h[8].^2))
    pre = p0(h0)
    info = ct_faces_from_cells!(s)
    @test info.cell_b_rel_l2_change < 1e-12
    @test info.max_divb_face_dx_over_brms < 1e-13
    @test maximum(abs.(p0(fields_to_host(s)) .- pre)) < 1e-13
end

@testset "PMF initializer installs divergence-free faces on a CT state" begin
    s = allocate_state(backend(:cpu), Float32, (16,16,16); dx=1/16, mhd_solver=:ct)
    r = init_pmf_batchelor!(s; brms=0.1, seed=3)
    @test haskey(r, :ct_faces) && r.ct_faces.max_divb_face_dx_over_brms < 1e-5
    step!(s, 1f-3)
    @test max_divb_face(s) * Float64(s.dx) / 0.1 < 1e-5
    @test all(isfinite, fields_to_host(s)[5])
end

@testset "divergence preservation under evolution" begin
    for (T, tol, N) in ((Float64, 1e-12, 24), (Float32, 1e-5, 24))
        s = allocate_state(backend(:cpu), T, (N,N,N); dx=1/N, gamma=5/3, riemann=:hlld, mhd_solver=:ct)
        tp = 2π
        for k in 1:N, j in 1:N, i in 1:N
            x = (i-0.5)/N; y = (j-0.5)/N; z = (k-0.5)/N; c = linidx(N, N, i, j, k)
            rho = 1 + 0.2*sin(tp*x)*cos(tp*z); vx = 0.5*sin(tp*y); vy = 0.5*sin(tp*z+0.4); vz = 0.5*sin(tp*x+0.9)
            bx = 0.4*sin(tp*y+0.2) + 0.2*cos(tp*z); by = 0.4*sin(tp*z) + 0.2*cos(tp*x+0.5); bz = 0.4*sin(tp*x+1.0) + 0.2*cos(tp*y)
            s.U[1][c] = rho; s.U[2][c] = rho*vx; s.U[3][c] = rho*vy; s.U[4][c] = rho*vz
            s.U[5][c] = 0.6/(5/3-1) + 0.5*rho*(vx^2+vy^2+vz^2) + 0.5*(bx^2+by^2+bz^2)
            s.U[6][c] = bx; s.U[7][c] = by; s.U[8][c] = bz
        end
        info = ct_faces_from_cells!(s)
        brms(s) = (h = fields_to_host(s); sqrt(sum(sum(abs2, Float64.(h[v])) for v in 6:8) / length(h[1])))
        divs = Float64[]
        t, n = evolve!(s, 0.35; cfl=0.3, callback=(s, t, n) -> push!(divs, max_divb_face(s)*Float64(s.dx)/brms(s)))
        @printf("  div-preservation %s N=%d: init %.2e; %d steps, max div_face*dx/Brms over run=%.3e (min %.3e)\n",
                T, N, info.max_divb_face_dx_over_brms, n, maximum(divs), minimum(divs))
        @test n >= 50
        @test info.max_divb_face_dx_over_brms < tol
        @test maximum(divs) < tol
        h = fields_to_host(s)
        @test all(isfinite, h[5]) && minimum(h[1]) > 0
        @test maximum(abs.(Array(s.U[9]))) == 0       # no psi
    end
end

# ── (c) convergence ───────────────────────────────────────────────────────────
function ct_alfven_L1(T, N; amp=0.1, p0=0.1, tfinal=0.5, cfl=0.4)
    s = allocate_state(backend(:cpu), T, (N,4,4); dx=1/N, gamma=5/3, use_hlld=true, mhd_solver=:ct)
    init_alfven_wave!(s; amp=amp, B0=1, p0=p0)
    ct_faces_from_cells!(s)
    t, n = evolve!(s, tfinal; cfl=cfl)
    By = fields_to_host(s)[7]
    err = 0.0
    for i in 1:N
        err += abs(Float64(By[i]) - alfven_By_exact((i-0.5)/N, t, 1.0; amp=amp))
    end
    err/N, n, max_divb_face(s)
end

function ct_entropy_L1(T, N; tfinal=1.0, cfl=0.4)
    gamma = 5/3
    s = allocate_state(backend(:cpu), T, (N,4,4); dx=1/N, gamma=gamma, use_hlld=true, mhd_solver=:ct)
    for k in 1:4, j in 1:4, i in 1:N
        c = linidx(N, 4, i, j, k); x = (i-0.5)/N
        rho = 1 + 0.2*sin(2π*x); vx = 1.0; vy = 0.3
        s.U[1][c] = rho; s.U[2][c] = rho*vx; s.U[3][c] = rho*vy; s.U[4][c] = 0
        s.U[5][c] = 1/(gamma-1) + 0.5*rho*(vx^2+vy^2) + 0.5*(0.5^2+0.3^2)
        s.U[6][c] = 0.5; s.U[7][c] = 0.3; s.U[8][c] = 0
    end
    ct_faces_from_cells!(s)
    t, n = evolve!(s, tfinal; cfl=cfl)
    rho = fields_to_host(s)[1]
    err = sum(abs(Float64(rho[i]) - (1 + 0.2*sin(2π*mod((i-0.5)/N - 1.0*t, 1.0)))) for i in 1:N) / N
    err, n
end

@testset "second-order convergence (CT)" begin
    for T in (Float64, Float32)
        # The entropy wave is pre-asymptotic at N=16->32 (order 1.67, identical to GLM, which
        # shares the hydro for uniform B: limiter clipping at the extrema), so it is measured on
        # 32,64,128; the Alfven wave is measured on 16,32,64.
        for (name, f, Ns) in (("Alfven", (N) -> ct_alfven_L1(T, N), (16, 32, 64)),
                              ("entropy", (N) -> ct_entropy_L1(T, N), (32, 64, 128)))
            L1 = Float64[]
            for N in Ns
                r = f(N); push!(L1, r[1])
                @printf("  [CT %s %s] N=%-3d L1=%.4e (%d steps)\n", T, name, N, r[1], r[2])
            end
            orders = [log2(L1[k-1]/L1[k]) for k in 2:3]
            @printf("  [CT %s %s] orders: %.3f %.3f\n", T, name, orders...)
            @test all(isfinite, L1)
            @test all(>=(1.8), orders)
        end
    end
end

# ── (d) Gardiner-Stone field loop ─────────────────────────────────────────────
function ct_loop_run(solver, N)
    s = allocate_state(backend(:cpu), Float32, (N,N,4); dx=1/N, gamma=5/3, riemann=:hlld,
                       recon=:plm, mhd_solver=solver)
    init_field_loop!(s; radius=0.3, amplitude=1e-3, velocity=(1,1,0))
    h0 = fields_to_host(s)
    bnorm = sum(abs, h0[6]) + sum(abs, h0[7]); emag0 = sum(abs2, h0[6]) + sum(abs2, h0[7])
    div0 = solver === :ct ? max_divb_face(s)*Float64(s.dx)/sqrt(emag0/length(h0[6])) : NaN
    t, n = evolve!(s, 1.0; cfl=0.4, glm_ch_fac=2, glm_cr=1, integrator=:ref)
    h1 = fields_to_host(s)
    l1 = (sum(abs, h1[6] .- h0[6]) + sum(abs, h1[7] .- h0[7])) / bnorm
    emag1 = sum(abs2, h1[6]) + sum(abs2, h1[7])
    divn = solver === :ct ? max_divb_face(s)*Float64(s.dx)/sqrt(emag1/length(h1[6])) :
                            Float64(max_divb(s))*Float64(s.dx)/sqrt(emag1/length(h1[6]))
    @printf("  field loop %s N=%d t=%.3f (%d steps): L1=%.3e Emag retention=%.3f div*dx/Brms=%.3e (init %.2e)\n",
            solver, N, t, n, l1, emag1/emag0, divn, div0)
    (l1=l1, ret=emag1/emag0, divn=divn, div0=div0, h=h1)
end

@testset "field-loop advection (CT vs GLM)" begin
    glm = ct_loop_run(:glm, 64)
    ct64 = ct_loop_run(:ct, 64)
    @test ct64.div0 < 1e-5
    @test ct64.divn < 2e-5
    @test all(isfinite, ct64.h[5]) && minimum(ct64.h[1]) > 0
    @test ct64.l1 < 0.7
    @test ct64.ret > 0.2
    ct128 = ct_loop_run(:ct, 128)
    @printf("  field loop CT L1 64->128: %.3e -> %.3e (GLM 64: %.3e)\n", ct64.l1, ct128.l1, glm.l1)
    @test ct128.l1 < ct64.l1
    @test ct128.divn < 2e-5
end

# ── (e) source-coupled smoke tests ───────────────────────────────────────────
function ct_smoke_state(N; T=Float32)
    gamma = T(5)/3
    s = allocate_state(backend(:cpu), T, (N,N,N); dx=1/N, gamma=gamma, riemann=:hll, mhd_solver=:ct)
    for k in 1:N, j in 1:N, i in 1:N
        c = linidx(N, N, i, j, k); ph = 2*T(pi)*(i-1)/N; ph2 = 2*T(pi)*(j-1)/N
        rho = 1 + T(1e-2)*cos(ph); vx = T(1e-2)*sin(ph)
        s.U[1][c] = rho; s.U[2][c] = rho*vx; s.U[3][c] = 0; s.U[4][c] = 0
        bx = T(0.5); by = T(0.05)*sin(ph); bz = T(0.03)*cos(ph2 + 1)
        s.U[6][c] = bx; s.U[7][c] = by; s.U[8][c] = bz
        s.U[5][c] = 1/(gamma-1) + T(0.5)*rho*vx*vx + T(0.5)*(bx*bx+by*by+bz*bz)
    end
    ct_faces_from_cells!(s)
    s
end

@testset "drag and radiation coupled steps with CT" begin
    s = ct_smoke_state(16)
    for _ in 1:4
        step_drag_godunov!(s, 2f-3; drag_impulse=0.5, ch=1f0, integrator=:ref)
    end
    h = fields_to_host(s)
    d = max_divb_face(s)*Float64(s.dx)/sqrt(sum(abs2, h[6])/length(h[6]))
    @printf("  drag-godunov CT smoke: div_face*dx/Brms=%.3e\n", d)
    @test all(x -> all(isfinite, x), h) && minimum(h[1]) > 0
    @test d < 1e-5
    # drag with a moving target through the exponential/strang wrappers
    for f in (step_drag_strang!, step_drag_exponential!)
        s2 = ct_smoke_state(16)
        f(s2, 2f-3; drag_impulse=0.4, ch=1f0, integrator=:ref, target_velocity=(0.01,0,0))
        @test all(x -> all(isfinite, x), fields_to_host(s2))
        @test max_divb_face(s2)*Float64(s2.dx)/0.5 < 1e-5
    end

    N = 16; sr = ct_smoke_state(N)
    ws = allocate_radiation_workspace(sr)
    photons = allocate_photon_moment_state(sr)
    initialize_photon_moments!(photons, ws, sr)
    rc = RadiationClosure(inertia_ratio=0.2f0, mean_free_path=0.001f0, photon_speed=20f0,
                          gas_sound_speed=1f0, response_model=:moments)
    ledger = RadiationEnergyLedger()
    for _ in 1:2
        step_radiation_godunov!(sr, ws, 2f-4; drag_impulse=7f0*2f-4, closure=rc, ch=0f0,
                                integrator=:ref, photon_state=photons, energy_ledger=ledger)
    end
    h = fields_to_host(sr)
    d = max_divb_face(sr)*Float64(sr.dx)/sqrt(sum(abs2, h[6])/length(h[6]))
    @printf("  radiation-godunov CT smoke: div_face*dx/Brms=%.3e (ledger samples=%d)\n", d, ledger.samples)
    @test all(x -> all(isfinite, x), h) && minimum(h[1]) > 0
    @test d < 1e-5
    @test ledger.samples == 2
    @test_throws ErrorException step_radiation_godunov!(sr, ws, 2f-4; drag_impulse=1f-3, closure=rc,
                                                        ch=0f0, integrator=:cube, photon_state=photons)
    # cell-centred terminal induction would overwrite B: refused
    @test_throws ErrorException terminal_ct_induction!(sr, 1f-4; gamma_drag=10, pressure_coeff=1)
end

end # constrained transport
