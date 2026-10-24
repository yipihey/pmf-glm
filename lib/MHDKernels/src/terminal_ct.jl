# Divergence-preserving terminal-regime induction for early-PMF evolution.
# The pressure response is supplied by the caller; this module advances B with
# SSPRK2 using a centered curl plus LLF-equivalent resistivity.  On a periodic
# uniform grid, div(curl(.))=0 and div(laplacian(B))=laplacian(div(B)) for the
# matching centered operators, so an initially solenoidal field stays so without
# a projection FFT.

export terminal_ct_induction!, terminal_magnetic_force!,
       terminal_magnetic_force_split!, terminal_velocity_max!

@inline _tct_lin3(i, j, k, N) = begin
    ii = mod(i - 1, N) + 1
    jj = mod(j - 1, N) + 1
    kk = mod(k - 1, N) + 1
    ((kk - 1) * N + (jj - 1)) * N + ii
end

@inline function _tct_magnetic_force_at(Bx, By, Bz, i, j, k, N, inv2dx)
    T = typeof(inv2dx)
    cxp = _tct_lin3(i + 1, j, k, N); cxm = _tct_lin3(i - 1, j, k, N)
    cyp = _tct_lin3(i, j + 1, k, N); cym = _tct_lin3(i, j - 1, k, N)
    czp = _tct_lin3(i, j, k + 1, N); czm = _tct_lin3(i, j, k - 1, N)

    bxp = Bx[cxp]; byp = By[cxp]; bzp = Bz[cxp]
    bxm = Bx[cxm]; bym = By[cxm]; bzm = Bz[cxm]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    dtxx = (bxp*bxp - T(0.5)*b2p) - (bxm*bxm - T(0.5)*b2m)
    dtyx = byp*bxp - bym*bxm
    dtzx = bzp*bxp - bzm*bxm

    bxp = Bx[cyp]; byp = By[cyp]; bzp = Bz[cyp]
    bxm = Bx[cym]; bym = By[cym]; bzm = Bz[cym]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    dtxy = bxp*byp - bxm*bym
    dtyy = (byp*byp - T(0.5)*b2p) - (bym*bym - T(0.5)*b2m)
    dtzy = bzp*byp - bzm*bym

    bxp = Bx[czp]; byp = By[czp]; bzp = Bz[czp]
    bxm = Bx[czm]; bym = By[czm]; bzm = Bz[czm]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    dtxz = bxp*bzp - bxm*bzm
    dtyz = byp*bzp - bym*bzm
    dtzz = (bzp*bzp - T(0.5)*b2p) - (bzm*bzm - T(0.5)*b2m)

    return (dtxx + dtxy + dtxz) * inv2dx,
           (dtyx + dtyy + dtyz) * inv2dx,
           (dtzx + dtzy + dtzz) * inv2dx
end

@inline function _tct_magnetic_force_split_at(Bx, By, Bz, i, j, k, N, inv2dx)
    T = typeof(inv2dx)
    fx, fy, fz = _tct_magnetic_force_at(Bx, By, Bz, i, j, k, N, inv2dx)
    cxp = _tct_lin3(i + 1, j, k, N); cxm = _tct_lin3(i - 1, j, k, N)
    cyp = _tct_lin3(i, j + 1, k, N); cym = _tct_lin3(i, j - 1, k, N)
    czp = _tct_lin3(i, j, k + 1, N); czm = _tct_lin3(i, j, k - 1, N)
    b2xp = Bx[cxp]*Bx[cxp] + By[cxp]*By[cxp] + Bz[cxp]*Bz[cxp]
    b2xm = Bx[cxm]*Bx[cxm] + By[cxm]*By[cxm] + Bz[cxm]*Bz[cxm]
    b2yp = Bx[cyp]*Bx[cyp] + By[cyp]*By[cyp] + Bz[cyp]*Bz[cyp]
    b2ym = Bx[cym]*Bx[cym] + By[cym]*By[cym] + Bz[cym]*Bz[cym]
    b2zp = Bx[czp]*Bx[czp] + By[czp]*By[czp] + Bz[czp]*Bz[czp]
    b2zm = Bx[czm]*Bx[czm] + By[czm]*By[czm] + Bz[czm]*Bz[czm]
    px = -T(0.5) * (b2xp - b2xm) * inv2dx
    py = -T(0.5) * (b2yp - b2ym) * inv2dx
    pz = -T(0.5) * (b2zp - b2zm) * inv2dx
    return px, py, pz, fx - px, fy - py, fz - pz
end

@kernel function _tct_magnetic_force_k!(fx, fy, fz,
                                        @Const(Bx), @Const(By), @Const(Bz),
                                        N::Int, inv2dx)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        x, y, z = _tct_magnetic_force_at(Bx, By, Bz, i, j, k, N, inv2dx)
        fx[c] = x; fy[c] = y; fz[c] = z
    end
end

@kernel function _tct_magnetic_force_split_k!(px, py, pz, tx, ty, tz,
                                              @Const(Bx), @Const(By), @Const(Bz),
                                              N::Int, inv2dx)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        pxi, pyi, pzi, txi, tyi, tzi =
            _tct_magnetic_force_split_at(Bx, By, Bz, i, j, k, N, inv2dx)
        px[c] = pxi; py[c] = pyi; pz[c] = pzi
        tx[c] = txi; ty[c] = tyi; tz[c] = tzi
    end
end

function terminal_magnetic_force!(s::MHDState, fx, fy, fz;
                                  B=(s.U[6], s.U[7], s.U[8]))
    N = s.dims[1]
    all(==(N), s.dims) || error("terminal magnetic force requires a cubic grid")
    _tct_magnetic_force_k!(s.be)(fx, fy, fz, B..., N, inv(typeof(s.dx)(2) * s.dx);
                                     ndrange=ncells(s))
    return fx, fy, fz
end

"""
    terminal_magnetic_force_split!(s, px, py, pz, tx, ty, tz; B=...)

Compute the centered conservative Maxwell-stress force as magnetic pressure
`-grad(B^2/2)` plus the remaining tension contribution. The remainder uses the
same discrete total-force stencil as [`terminal_magnetic_force!`](@ref), so the
two terms reconstruct the force used by the solver. For a solenoidal field the
tension contribution approaches `(B dot grad)B`.
"""
function terminal_magnetic_force_split!(s::MHDState, px, py, pz, tx, ty, tz;
                                        B=(s.U[6], s.U[7], s.U[8]))
    N = s.dims[1]
    all(==(N), s.dims) || error("terminal magnetic force requires a cubic grid")
    _tct_magnetic_force_split_k!(s.be)(px, py, pz, tx, ty, tz, B..., N,
                                       inv(typeof(s.dx)(2) * s.dx);
                                       ndrange=ncells(s))
    return px, py, pz, tx, ty, tz
end

@inline function _tct_velocity_at(rho, fx, fy, fz, i, j, k, N, inv2dx,
                                  gamma_drag, pressure_coeff, smallr, vcap)
    c = _tct_lin3(i, j, k, N)
    cxp = _tct_lin3(i + 1, j, k, N); cxm = _tct_lin3(i - 1, j, k, N)
    cyp = _tct_lin3(i, j + 1, k, N); cym = _tct_lin3(i, j - 1, k, N)
    czp = _tct_lin3(i, j, k + 1, N); czm = _tct_lin3(i, j, k - 1, N)
    r = max(rho[c], smallr)
    gx = pressure_coeff * (rho[cxp] - rho[cxm]) * inv2dx
    gy = pressure_coeff * (rho[cyp] - rho[cym]) * inv2dx
    gz = pressure_coeff * (rho[czp] - rho[czm]) * inv2dx
    invrg = one(gamma_drag) / (r * max(gamma_drag, eps(gamma_drag)))
    vx = (fx[c] - gx) * invrg
    vy = (fy[c] - gy) * invrg
    vz = (fz[c] - gz) * invrg
    v2 = vx*vx + vy*vy + vz*vz
    if v2 > vcap*vcap
        q = vcap / sqrt(v2)
        vx *= q; vy *= q; vz *= q
    end
    return vx, vy, vz
end

@inline function _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz,
                             i, j, k, N, inv2dx, gamma_drag,
                             pressure_coeff, smallr, vcap)
    c = _tct_lin3(i, j, k, N)
    vx, vy, vz = _tct_velocity_at(rho, fx, fy, fz, i, j, k, N, inv2dx,
                                  gamma_drag, pressure_coeff, smallr, vcap)
    bx = Bx[c]; by = By[c]; bz = Bz[c]
    return vy*bz - vz*by, vz*bx - vx*bz, vx*by - vy*bx
end

@inline function _tct_lap_at(A, i, j, k, N, invdx2)
    c = _tct_lin3(i, j, k, N)
    return (A[_tct_lin3(i + 1, j, k, N)] + A[_tct_lin3(i - 1, j, k, N)] +
            A[_tct_lin3(i, j + 1, k, N)] + A[_tct_lin3(i, j - 1, k, N)] +
            A[_tct_lin3(i, j, k + 1, N)] + A[_tct_lin3(i, j, k - 1, N)] -
            typeof(invdx2)(6) * A[c]) * invdx2
end

@inline function _tct_bilap_at(A, i, j, k, N, invdx2)
    T = typeof(invdx2)
    c = A[_tct_lin3(i, j, k, N)]
    axis1 = A[_tct_lin3(i + 1, j, k, N)] + A[_tct_lin3(i - 1, j, k, N)] +
            A[_tct_lin3(i, j + 1, k, N)] + A[_tct_lin3(i, j - 1, k, N)] +
            A[_tct_lin3(i, j, k + 1, N)] + A[_tct_lin3(i, j, k - 1, N)]
    axis2 = A[_tct_lin3(i + 2, j, k, N)] + A[_tct_lin3(i - 2, j, k, N)] +
            A[_tct_lin3(i, j + 2, k, N)] + A[_tct_lin3(i, j - 2, k, N)] +
            A[_tct_lin3(i, j, k + 2, N)] + A[_tct_lin3(i, j, k - 2, N)]
    diag = A[_tct_lin3(i + 1, j + 1, k, N)] + A[_tct_lin3(i + 1, j - 1, k, N)] +
           A[_tct_lin3(i - 1, j + 1, k, N)] + A[_tct_lin3(i - 1, j - 1, k, N)] +
           A[_tct_lin3(i + 1, j, k + 1, N)] + A[_tct_lin3(i + 1, j, k - 1, N)] +
           A[_tct_lin3(i - 1, j, k + 1, N)] + A[_tct_lin3(i - 1, j, k - 1, N)] +
           A[_tct_lin3(i, j + 1, k + 1, N)] + A[_tct_lin3(i, j + 1, k - 1, N)] +
           A[_tct_lin3(i, j - 1, k + 1, N)] + A[_tct_lin3(i, j - 1, k - 1, N)]
    return (T(42)*c - T(12)*axis1 + axis2 + T(2)*diag) * invdx2 * invdx2
end

@inline function _tct_rhs_at(rho, fx, fy, fz, Bx, By, Bz,
                             i, j, k, N, inv2dx, invdx2, eta, dissipation_order::Int,
                             gamma_drag, pressure_coeff, smallr, vcap)
    exyp = _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz, i, j + 1, k, N, inv2dx,
                       gamma_drag, pressure_coeff, smallr, vcap)
    exym = _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz, i, j - 1, k, N, inv2dx,
                       gamma_drag, pressure_coeff, smallr, vcap)
    exzp = _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k + 1, N, inv2dx,
                       gamma_drag, pressure_coeff, smallr, vcap)
    exzm = _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k - 1, N, inv2dx,
                       gamma_drag, pressure_coeff, smallr, vcap)
    exxp = _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz, i + 1, j, k, N, inv2dx,
                       gamma_drag, pressure_coeff, smallr, vcap)
    exxm = _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz, i - 1, j, k, N, inv2dx,
                       gamma_drag, pressure_coeff, smallr, vcap)
    dx = ((exyp[3] - exym[3]) - (exzp[2] - exzm[2])) * inv2dx
    dy = ((exzp[1] - exzm[1]) - (exxp[3] - exxm[3])) * inv2dx
    dz = ((exxp[2] - exxm[2]) - (exyp[1] - exym[1])) * inv2dx
    if dissipation_order == 4
        return dx - eta * _tct_bilap_at(Bx, i, j, k, N, invdx2),
               dy - eta * _tct_bilap_at(By, i, j, k, N, invdx2),
               dz - eta * _tct_bilap_at(Bz, i, j, k, N, invdx2)
    end
    return dx + eta * _tct_lap_at(Bx, i, j, k, N, invdx2),
           dy + eta * _tct_lap_at(By, i, j, k, N, invdx2),
           dz + eta * _tct_lap_at(Bz, i, j, k, N, invdx2)
end

@kernel function _tct_speed_k!(speed, @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                               N::Int, inv2dx, gamma_drag, pressure_coeff, smallr, vcap)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        vx, vy, vz = _tct_velocity_at(rho, fx, fy, fz, i, j, k, N, inv2dx,
                                      gamma_drag, pressure_coeff, smallr, vcap)
        speed[c] = sqrt(vx*vx + vy*vy + vz*vz)
    end
end

function terminal_velocity_max!(s::MHDState{T}; gamma_drag::Real,
                                pressure_coeff::Real, vcap::Real=Inf) where {T}
    N = s.dims[1]
    fx, fy, fz = s.scratch[2], s.scratch[3], s.scratch[4]
    speed = s.scratch[1]
    inv2dx = inv(T(2) * s.dx)
    terminal_magnetic_force!(s, fx, fy, fz)
    _tct_speed_k!(s.be)(speed, s.U[1], fx, fy, fz, N, inv2dx,
                        T(gamma_drag), T(pressure_coeff), s.smallr, T(vcap);
                        ndrange=ncells(s))
    KA.synchronize(s.be)
    return Float64(maximum(speed))
end

@kernel function _tct_predict_k!(pBx, pBy, pBz,
                                 @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                                 @Const(Bx), @Const(By), @Const(Bz),
                                 N::Int, dt, inv2dx, invdx2, eta, dissipation_order::Int,
                                 gamma_drag, pressure_coeff, smallr, vcap)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        x, y, z = _tct_rhs_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k, N,
                              inv2dx, invdx2, eta, dissipation_order,
                              gamma_drag, pressure_coeff, smallr, vcap)
        pBx[c] = Bx[c] + dt*x
        pBy[c] = By[c] + dt*y
        pBz[c] = Bz[c] + dt*z
    end
end

@kernel function _tct_correct_k!(Bx, By, Bz, E,
                                 @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                                 @Const(pBx), @Const(pBy), @Const(pBz),
                                 N::Int, dt, inv2dx, invdx2, eta, dissipation_order::Int,
                                 gamma_drag, pressure_coeff, smallr, smallE, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        x, y, z = _tct_rhs_at(rho, fx, fy, fz, pBx, pBy, pBz, i, j, k, N,
                              inv2dx, invdx2, eta, dissipation_order,
                              gamma_drag, pressure_coeff, smallr, vcap)
        ox = Bx[c]; oy = By[c]; oz = Bz[c]
        nx = T(0.5) * (ox + pBx[c] + dt*x)
        ny = T(0.5) * (oy + pBy[c] + dt*y)
        nz = T(0.5) * (oz + pBz[c] + dt*z)
        eb0 = T(0.5) * (ox*ox + oy*oy + oz*oz)
        eb1 = T(0.5) * (nx*nx + ny*ny + nz*nz)
        Bx[c] = nx; By[c] = ny; Bz[c] = nz
        E[c] = max(E[c] + eb1 - eb0, smallE)
    end
end

# ── Face-staggered (constrained-transport) terminal induction ────────────────────────────
# For `mhd_solver=:ct` states the same SSPRK2 scheme advances the FACE fields. The cell EMF
# e = v x B is computed exactly as above (cell-centred v and B), averaged to the cell EDGES
# (each edge EMF is the mean of the four cells sharing the edge), and the faces advance by the
# discrete curl of the edge EMFs. The dissipation is the face-component Laplacian (order 2) or
# bilaplacian (order 4) with the identical stencils; both commute with the face divergence
# on a periodic grid, so div b stays at round-off. Cell-centred B is the face average after every
# stage and the total energy follows the magnetic-energy change as in `_tct_correct_k!`.
# Workspace (free outside the Godunov step): ct.ec* (cell EMF), ct.e* (edge EMF), ct.mf*
# (predicted faces).

@kernel function _tctc_cell_emf_k!(ecx, ecy, ecz,
                                   @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                                   @Const(Bx), @Const(By), @Const(Bz),
                                   N::Int, inv2dx, gamma_drag, pressure_coeff, smallr, vcap)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        x, y, z = _tct_emf_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k, N, inv2dx,
                              gamma_drag, pressure_coeff, smallr, vcap)
        ecx[c] = x; ecy[c] = y; ecz[c] = z
    end
end

# Cell EMF v x B from prescribed cell velocity arrays (radiation terminal closure).
@kernel function _tctc_cell_emf_v_k!(ecx, ecy, ecz, @Const(vx), @Const(vy), @Const(vz),
                                     @Const(Bx), @Const(By), @Const(Bz), N::Int)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        x, y, z = _rtct_emf_at(vx, vy, vz, Bx, By, Bz, i, j, k, N)
        ecx[c] = x; ecy[c] = y; ecz[c] = z
    end
end

# Edge EMF = mean of the four cells sharing the edge. Lower-face convention of ct.jl:
# ez at (i-1/2,j-1/2,k), ex at (i,j-1/2,k-1/2), ey at (i-1/2,j,k-1/2).
@kernel function _tctc_edge_k!(ex, ey, ez, @Const(ecx), @Const(ecy), @Const(ecz), N::Int)
    c = @index(Global)
    @inbounds begin
        T = eltype(ex)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        c0 = c
        cim = _tct_lin3(i - 1, j, k, N); cjm = _tct_lin3(i, j - 1, k, N)
        ckm = _tct_lin3(i, j, k - 1, N)
        cimjm = _tct_lin3(i - 1, j - 1, k, N)
        cjmkm = _tct_lin3(i, j - 1, k - 1, N)
        ckmim = _tct_lin3(i - 1, j, k - 1, N)
        q = T(0.25)
        ez[c] = q * (ecz[c0] + ecz[cim] + ecz[cjm] + ecz[cimjm])
        ex[c] = q * (ecx[c0] + ecx[cjm] + ecx[ckm] + ecx[cjmkm])
        ey[c] = q * (ecy[c0] + ecy[cim] + ecy[ckm] + ecy[ckmim])
    end
end

# d/dt of the faces: curl of the edge EMFs (e = v x B) plus face dissipation of `A*`.
@inline function _tctc_rate_at(ex, ey, ez, Ax, Ay, Az, i, j, k, N, invdx, invdx2,
                               eta, dissipation_order::Int)
    c  = _tct_lin3(i, j, k, N)
    ip = _tct_lin3(i + 1, j, k, N); jp = _tct_lin3(i, j + 1, k, N)
    kp = _tct_lin3(i, j, k + 1, N)
    dx = ((ez[jp] - ez[c]) - (ey[kp] - ey[c])) * invdx
    dy = ((ex[kp] - ex[c]) - (ez[ip] - ez[c])) * invdx
    dz = ((ey[ip] - ey[c]) - (ex[jp] - ex[c])) * invdx
    if dissipation_order == 4
        return dx - eta * _tct_bilap_at(Ax, i, j, k, N, invdx2),
               dy - eta * _tct_bilap_at(Ay, i, j, k, N, invdx2),
               dz - eta * _tct_bilap_at(Az, i, j, k, N, invdx2)
    end
    return dx + eta * _tct_lap_at(Ax, i, j, k, N, invdx2),
           dy + eta * _tct_lap_at(Ay, i, j, k, N, invdx2),
           dz + eta * _tct_lap_at(Az, i, j, k, N, invdx2)
end

@kernel function _tctc_predict_k!(pbx, pby, pbz, @Const(bxf), @Const(byf), @Const(bzf),
                                  @Const(ex), @Const(ey), @Const(ez),
                                  N::Int, dt, invdx, invdx2, eta, dissipation_order::Int)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        x, y, z = _tctc_rate_at(ex, ey, ez, bxf, byf, bzf, i, j, k, N, invdx, invdx2,
                                eta, dissipation_order)
        pbx[c] = bxf[c] + dt*x
        pby[c] = byf[c] + dt*y
        pbz[c] = bzf[c] + dt*z
    end
end

# SSPRK2 corrector on the faces; edge EMFs belong to the predicted state. `COMP` selects
# Kahan-compensated accumulation (mirrors `_rtct_correct_k!`'s compensated branch).
@kernel function _tctc_correct_k!(bxf, byf, bzf, rbx, rby, rbz,
                                  @Const(pbx), @Const(pby), @Const(pbz),
                                  @Const(ex), @Const(ey), @Const(ez),
                                  N::Int, dt, invdx, invdx2, eta, dissipation_order::Int,
                                  compensated::Val{COMP}) where {COMP}
    c = @index(Global)
    @inbounds begin
        T = eltype(bxf)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        x, y, z = _tctc_rate_at(ex, ey, ez, pbx, pby, pbz, i, j, k, N, invdx, invdx2,
                                eta, dissipation_order)
        ox = bxf[c]; oy = byf[c]; oz = bzf[c]
        if COMP
            bxf[c] = _compensated_update!(rbx, c, ox, T(0.5) * ((pbx[c] - ox) + dt*x), Val(true))
            byf[c] = _compensated_update!(rby, c, oy, T(0.5) * ((pby[c] - oy) + dt*y), Val(true))
            bzf[c] = _compensated_update!(rbz, c, oz, T(0.5) * ((pbz[c] - oz) + dt*z), Val(true))
        else
            bxf[c] = T(0.5) * (ox + pbx[c] + dt*x)
            byf[c] = T(0.5) * (oy + pby[c] + dt*y)
            bzf[c] = T(0.5) * (oz + pbz[c] + dt*z)
        end
    end
end

# Cell B <- face average; E += magnetic-energy change (floor smallE), as `_tct_correct_k!`.
@kernel function _tctc_cell_energy_k!(Bx, By, Bz, E, @Const(bxf), @Const(byf), @Const(bzf),
                                      N::Int, smallE)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        nx = T(0.5) * (bxf[c] + bxf[_tct_lin3(i + 1, j, k, N)])
        ny = T(0.5) * (byf[c] + byf[_tct_lin3(i, j + 1, k, N)])
        nz = T(0.5) * (bzf[c] + bzf[_tct_lin3(i, j, k + 1, N)])
        ox = Bx[c]; oy = By[c]; oz = Bz[c]
        eb0 = T(0.5) * (ox*ox + oy*oy + oz*oz)
        eb1 = T(0.5) * (nx*nx + ny*ny + nz*nz)
        Bx[c] = nx; By[c] = ny; Bz[c] = nz
        E[c] = max(E[c] + eb1 - eb0, smallE)
    end
end

# Edge EMFs from the cell EMFs held in ct.ec*.
function _tctc_edges!(s::MHDState, N::Int)
    ct = s.ct
    _tctc_edge_k!(s.be)(ct.ex, ct.ey, ct.ez, ct.ecx, ct.ecy, ct.ecz, N; ndrange=ncells(s))
    return nothing
end

# Predictor: faces -> ct.mf* (stage state), cell B of the predictor -> scratch[6:8].
function _tctc_predict!(s::MHDState{T}, N, dt, invdx, invdx2, eta, order) where {T}
    ct = s.ct
    _tctc_predict_k!(s.be)(ct.mfx, ct.mfy, ct.mfz, ct.bxf, ct.byf, ct.bzf,
                           ct.ex, ct.ey, ct.ez, N, dt, invdx, invdx2, eta, order;
                           ndrange=ncells(s))
    _ct_bcc_k!(s.be, 256)(s.scratch[6], s.scratch[7], s.scratch[8],
                          ct.mfx, ct.mfy, ct.mfz, N, N, N; ndrange=ncells(s))
    return nothing
end

# Corrector: faces advance, then cell B / E follow.
function _tctc_correct!(s::MHDState{T}, N, dt, invdx, invdx2, eta, order, comp::Val) where {T}
    ct = s.ct
    _tctc_correct_k!(s.be)(ct.bxf, ct.byf, ct.bzf, ct.rbx, ct.rby, ct.rbz,
                           ct.mfx, ct.mfy, ct.mfz, ct.ex, ct.ey, ct.ez,
                           N, dt, invdx, invdx2, eta, order, comp; ndrange=ncells(s))
    _tctc_cell_energy_k!(s.be)(s.U[6], s.U[7], s.U[8], s.U[5], ct.bxf, ct.byf, ct.bzf,
                               N, T(1e-30); ndrange=ncells(s))
    return nothing
end

# One forward-Euler face update from cell EMFs already stored in ct.ec* (used by the
# first-order velocity-step terminal paths of the production runner): edge average, faces +=
# dt curl(e) (Kahan-compensated through ct.rb*), then cell B <- face average with E += (|B_face|^2 - |B_cell|^2)/2, replacing the
# provisional cell-centred B (and its energy) that the caller's cell kernel wrote.
@kernel function _tctc_euler_k!(bxf, byf, bzf, rbx, rby, rbz,
                                @Const(ex), @Const(ey), @Const(ez), N::Int, dt, invdx)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N*N) + 1
        ip = _tct_lin3(i + 1, j, k, N); jp = _tct_lin3(i, j + 1, k, N)
        kp = _tct_lin3(i, j, k + 1, N)
        bxf[c] = _compensated_update!(rbx, c, bxf[c], dt * ((ez[jp] - ez[c]) - (ey[kp] - ey[c])) * invdx, Val(true))
        byf[c] = _compensated_update!(rby, c, byf[c], dt * ((ex[kp] - ex[c]) - (ez[ip] - ez[c])) * invdx, Val(true))
        bzf[c] = _compensated_update!(rbz, c, bzf[c], dt * ((ey[ip] - ey[c]) - (ex[jp] - ex[c])) * invdx, Val(true))
    end
end

function terminal_ct_euler_from_cell_emf!(s::MHDState{T}, dt::Real) where {T}
    uses_ct(s) || error("terminal_ct_euler_from_cell_emf! needs a CT state")
    N = s.dims[1]; ct = s.ct
    _tctc_edges!(s, N)
    _tctc_euler_k!(s.be)(ct.bxf, ct.byf, ct.bzf, ct.rbx, ct.rby, ct.rbz, ct.ex, ct.ey, ct.ez, N, T(dt), inv(s.dx);
                         ndrange=ncells(s))
    _ct_cell_from_faces_k!(s.be, 256)(s.U[5], s.U[6], s.U[7], s.U[8], s.U[9],
                                      ct.bxf, ct.byf, ct.bzf, N, N, N; ndrange=ncells(s))
    KA.synchronize(s.be)
    return nothing
end

"""
    terminal_ct_induction!(s, dt; gamma_drag, pressure_coeff, cfl=0.4,
                           dissipation=0.05, dissipation_order=4,
                           vcap=Inf, max_subcycles=256)

Advance only the terminal-regime induction equation. Density and pressure are
held at the caller-provided macro-step state in `s`; Lorentz force and velocity
are recomputed at both SSPRK2 stages. Returns `(nsubcycles, vmax, eta)`.

For `mhd_solver=:ct` states the faces advance (see the face-staggered section above); with
`compensate=true` (default, CT only) each face update is Kahan-compensated through the persistent
face remainders `s.ct.rbx/rby/rbz`, which keeps the Float32 face divergence at round-off over
thousands of sub-ulp updates. GLM states ignore `compensate`.
"""
function terminal_ct_induction!(s::MHDState{T}, dt::Real;
                                gamma_drag::Real, pressure_coeff::Real,
                                cfl::Real=0.4, dissipation::Real=0.05,
                                dissipation_order::Int=4,
                                vcap::Real=Inf, max_subcycles::Int=256,
                                compensate::Bool=true) where {T}
    N = s.dims[1]
    all(==(N), s.dims) || error("terminal CT requires a cubic grid")
    0 < cfl <= 1 || error("terminal CT cfl must be in (0,1]")
    0 <= dissipation <= 1 || error("terminal CT dissipation must be in [0,1]")
    dissipation_order in (2, 4) || error("terminal CT dissipation_order must be 2 or 4")
    fx, fy, fz = s.scratch[2], s.scratch[3], s.scratch[4]
    inv2dx = inv(T(2) * s.dx)
    invdx2 = inv(s.dx * s.dx)
    vmax = terminal_velocity_max!(s; gamma_drag=gamma_drag,
                                  pressure_coeff=pressure_coeff, vcap=vcap)
    nsub = max(1, ceil(Int, Float64(dt) * vmax /
                           max(Float64(cfl) * Float64(s.dx), eps(Float64))))
    nsub <= max_subcycles || error("terminal CT needs $nsub subcycles (limit $max_subcycles)")
    dts = T(dt / nsub)
    eta = dissipation_order == 4 ?
          T(dissipation / 12) * T(vmax) * s.dx^3 :
          T(dissipation) * T(vmax) * s.dx
    if uses_ct(s)
        _terminal_ct_face_subcycles!(s, nsub, dts, eta, dissipation_order;
                                     gamma_drag, pressure_coeff, vcap, compensate)
        KA.synchronize(s.be)
        return nsub, vmax, Float64(eta)
    end
    for _ in 1:nsub
        terminal_magnetic_force!(s, fx, fy, fz)
        _tct_predict_k!(s.be)(s.scratch[6], s.scratch[7], s.scratch[8],
                              s.U[1], fx, fy, fz, s.U[6], s.U[7], s.U[8],
                              N, dts, inv2dx, invdx2, eta, dissipation_order,
                              T(gamma_drag), T(pressure_coeff), s.smallr, T(vcap);
                              ndrange=ncells(s))
        _tct_magnetic_force_k!(s.be)(fx, fy, fz,
                                     s.scratch[6], s.scratch[7], s.scratch[8],
                                     N, inv2dx; ndrange=ncells(s))
        _tct_correct_k!(s.be)(s.U[6], s.U[7], s.U[8], s.U[5],
                              s.U[1], fx, fy, fz,
                              s.scratch[6], s.scratch[7], s.scratch[8],
                              N, dts, inv2dx, invdx2, eta, dissipation_order,
                              T(gamma_drag), T(pressure_coeff), s.smallr,
                              T(1e-30), T(vcap); ndrange=ncells(s))
    end
    KA.synchronize(s.be)
    return nsub, vmax, Float64(eta)
end

# CT variant of the subcycle loop (see the face-staggered section above).
function _terminal_ct_face_subcycles!(s::MHDState{T}, nsub, dts, eta, order;
                                      gamma_drag, pressure_coeff, vcap, compensate) where {T}
    N = s.dims[1]; ct = s.ct
    fx, fy, fz = s.scratch[2], s.scratch[3], s.scratch[4]
    inv2dx = inv(T(2) * s.dx); invdx = inv(s.dx); invdx2 = inv(s.dx * s.dx)
    gd = T(gamma_drag); pc = T(pressure_coeff); vc = T(vcap)
    for _ in 1:nsub
        terminal_magnetic_force!(s, fx, fy, fz)
        _tctc_cell_emf_k!(s.be)(ct.ecx, ct.ecy, ct.ecz, s.U[1], fx, fy, fz,
                                s.U[6], s.U[7], s.U[8], N, inv2dx, gd, pc, s.smallr, vc;
                                ndrange=ncells(s))
        _tctc_edges!(s, N)
        _tctc_predict!(s, N, dts, invdx, invdx2, eta, order)
        _tct_magnetic_force_k!(s.be)(fx, fy, fz, s.scratch[6], s.scratch[7], s.scratch[8],
                                     N, inv2dx; ndrange=ncells(s))
        _tctc_cell_emf_k!(s.be)(ct.ecx, ct.ecy, ct.ecz, s.U[1], fx, fy, fz,
                                s.scratch[6], s.scratch[7], s.scratch[8],
                                N, inv2dx, gd, pc, s.smallr, vc; ndrange=ncells(s))
        _tctc_edges!(s, N)
        _tctc_correct!(s, N, dts, invdx, invdx2, eta, order, Val(compensate))
    end
    return nothing
end
