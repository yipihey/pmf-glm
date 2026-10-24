# ── Face-staggered constrained transport (CT), an alternative to Dedner GLM ────
#
# SCHEME. The hydrodynamic part is shared with the GLM path: cell-centred Godunov,
# PLM(MonCen) with the full slope, a 3-direction MUSCL-Hancock predictor, and the
# selected Riemann solver (HLLD/HLL/LLF). What differs is the induction equation:
#
#   * B lives on the FACES of the grid (`CTState.bxf/byf/bzf`; `bxf[c]` is Bx on the
#     lower-x face (i-1/2,j,k) of cell c, etc.). The normal field at a face is that one
#     continuous value: both Riemann states are overwritten with it, there is no
#     Riemann problem for B_n, and no ψ (`riemann_ct`).
#   * The Godunov transverse-B fluxes of the faces are the face EMFs. The edge EMF is
#     built from the four surrounding face EMFs plus upwind (contact-mode) derivative
#     corrections selected by the sign of the face mass flux, from the cell-centred
#     reference EMF E = -v x B (v of the predicted half step, B at the start of the
#     step) — the "CT-contact" scheme of Gardiner & Stone 2005 (`_ct_gs05`).
#   * Faces are advanced by the discrete curl of the edge EMFs (`_ct_faceB_k!`), which
#     keeps the face divergence at round-off for all time; cell-centred B (used by the
#     hydro) is the average of the two faces (`_ct_bcc_k!`). Total energy is advanced by
#     the Godunov energy flux alone (no energy correction).
#
# PROVENANCE. A KernelAbstractions, precision-generic port of the fixed CT prototype on
# the Solaceonx fork of GLMMHDTurb (`ct_mhd.jl`, branch `sweep-fixes`): commit 9d20e52
# (full MonCen slope + 3-direction Hancock predictor) and commit 7256715 (Gardiner &
# Stone 2005 upwinded edge EMFs with a cell-centred reference EMF). Gardiner & Stone
# 2005, J. Comput. Phys. 205, 509. Deviations from the fork: the reference EMF velocity
# includes the same drag/radiation source modifications the face states receive
# (`_facep_sources`); generic (Nx,Ny,Nz) grids; pmf-glm's pressure/density floors.
#
# MEMORY. 3 face arrays + 15 workspace arrays (6 face-flux arrays, 3 mass-flux arrays,
# 3 cell EMFs, 3 edge EMFs) of N^3 on top of GLM's 18 (9 state + 9 scratch). The ψ slot
# stays allocated but is identically zero.
#
# LIMITS. Periodic boundaries only; `:ref` integrator only (the shared-memory cube
# kernels error for CT states); the terminal-regime induction paths (which overwrite
# cell-centred B) error for CT states.

export max_divb_face, ct_faces_from_cells!, ct_faces_from_potential!


function _require_glm(s::MHDState, what::AbstractString)
    uses_ct(s) && error("$what overwrites cell-centred B and is not available for " *
                        "mhd_solver=:ct states (constrained transport keeps B on faces)")
    nothing
end

function _require_ct_integrator(s::MHDState, what::AbstractString)
    uses_ct(s) && error("$what: CT is implemented on the :ref integrator only " *
                        "(mhd_solver=:ct states cannot use the :cube integrators)")
    nothing
end

# ── Gardiner & Stone (2005) CT-contact edge EMF ──────────────────────────────
# The edge sits at (a-1/2, b-1/2) in the plane of the two directions a, b across it.
# Ea0, Eam are the EMFs on the a-faces (a-1/2, b) and (a-1/2, b-1); Eb0, Ebm those on
# the b-faces (a, b-1/2) and (a-1, b-1/2); mA*, mB* the mass fluxes through those
# faces; Er** the cell-centred EMFs of cells (a+da, b+db), da, db in {0,-1}.
@inline function _ct_gs05(Ea0::T,Eam::T,Eb0::T,Ebm::T,mA0::T,mAm::T,mB0::T,mBm::T,
                          Er00::T,Erm0::T,Er0m::T,Ermm::T) where {T}
    h = T(0.5)
    up(m,lower,upper) = ifelse(m>zero(T),lower,ifelse(m<zero(T),upper,h*(lower+upper)))
    A1 = up(mA0,Erm0-Ebm,Er00-Eb0)
    A3 = up(mAm,Ebm-Ermm,Eb0-Er0m)
    B1 = up(mB0,Er0m-Eam,Er00-Ea0)
    B3 = up(mBm,Eam-Ermm,Ea0-Erm0)
    T(0.25)*((Ea0+Eam+Eb0+Ebm)+(A3-A1)+(B3-B1))
end

# Edge EMFs from the lower-face Godunov magnetic fluxes (E_z on x-faces = -F_x[By],
# on y-faces = +F_y[Bx]; E_x on y-faces = -F_y[Bz], on z-faces = +F_z[By];
# E_y on z-faces = -F_z[Bx], on x-faces = +F_x[Bz]).
@kernel function _ct_emf_k!(ex,ey,ez,@Const(mfx),@Const(mfy),@Const(mfz),
        @Const(fxby),@Const(fxbz),@Const(fybx),@Const(fybz),@Const(fzbx),@Const(fzby),
        @Const(ecx),@Const(ecy),@Const(ecz),Nx::Int,Ny::Int,Nz::Int)
    c=@index(Global,Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        L(a,b,d)=_periodic_corr_index(Nx,Ny,Nz,a,b,d)
        c0=c; cim=L(i-1,j,k); cjm=L(i,j-1,k); ckm=L(i,j,k-1)
        cimjm=L(i-1,j-1,k); cjmkm=L(i,j-1,k-1); ckmim=L(i-1,j,k-1)
        # ez at (i-1/2,j-1/2,k): a=x, b=y
        ez[c]=_ct_gs05(-fxby[c0],-fxby[cjm],fybx[c0],fybx[cim],
                       mfx[c0],mfx[cjm],mfy[c0],mfy[cim],
                       ecz[c0],ecz[cim],ecz[cjm],ecz[cimjm])
        # ex at (i,j-1/2,k-1/2): a=y, b=z
        ex[c]=_ct_gs05(-fybz[c0],-fybz[ckm],fzby[c0],fzby[cjm],
                       mfy[c0],mfy[ckm],mfz[c0],mfz[cjm],
                       ecx[c0],ecx[cjm],ecx[ckm],ecx[cjmkm])
        # ey at (i-1/2,j,k-1/2): a=z, b=x
        ey[c]=_ct_gs05(-fzbx[c0],-fzbx[cim],fxbz[c0],fxbz[ckm],
                       mfz[c0],mfz[cim],mfx[c0],mfx[ckm],
                       ecy[c0],ecy[ckm],ecy[cim],ecy[ckmim])
    end
end

# Face update by the discrete curl of the edge EMFs.
@kernel function _ct_faceB_k!(bxf,byf,bzf,@Const(ex),@Const(ey),@Const(ez),
        Nx::Int,Ny::Int,Nz::Int,dtdx::T) where {T}
    c=@index(Global,Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        L(a,b,d)=_periodic_corr_index(Nx,Ny,Nz,a,b,d)
        ip=L(i+1,j,k); jp=L(i,j+1,k); kp=L(i,j,k+1)
        bxf[c]-=dtdx*((ez[jp]-ez[c])-(ey[kp]-ey[c]))
        byf[c]-=dtdx*((ex[kp]-ex[c])-(ez[ip]-ez[c]))
        bzf[c]-=dtdx*((ey[ip]-ey[c])-(ex[jp]-ex[c]))
    end
end

# Cell-centred B = average of the two faces.
@kernel function _ct_bcc_k!(o6,o7,o8,@Const(bxf),@Const(byf),@Const(bzf),
        Nx::Int,Ny::Int,Nz::Int)
    c=@index(Global,Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        L(a,b,d)=_periodic_corr_index(Nx,Ny,Nz,a,b,d)
        T=eltype(o6)
        o6[c]=T(0.5)*(bxf[c]+bxf[L(i+1,j,k)])
        o7[c]=T(0.5)*(byf[c]+byf[L(i,j+1,k)])
        o8[c]=T(0.5)*(bzf[c]+bzf[L(i,j,k+1)])
    end
end

"""
    ct_update!(s, dtdx)

Second half of a CT step, run after the Godunov kernel and the `U`/`scratch` swap: edge
EMFs (Gardiner & Stone 2005 CT-contact) from the workspace fluxes, face update by the
discrete curl, and cell-centred B (`s.U[6:8]`) as the face average. `s.scratch` keeps the
full old state.
"""
function ct_update!(s::MHDState{T}, dtdx::Real) where {T}
    ct=s.ct; ct === nothing && error("ct_update! needs a CT state")
    Nx,Ny,Nz=s.dims; nc=ncells(s)
    _ct_emf_k!(s.be,256)(ct.ex,ct.ey,ct.ez,ct.mfx,ct.mfy,ct.mfz,ct.fxby,ct.fxbz,
        ct.fybx,ct.fybz,ct.fzbx,ct.fzby,ct.ecx,ct.ecy,ct.ecz,Nx,Ny,Nz;ndrange=nc)
    _ct_faceB_k!(s.be,256)(ct.bxf,ct.byf,ct.bzf,ct.ex,ct.ey,ct.ez,Nx,Ny,Nz,T(dtdx);
                           ndrange=nc)
    _ct_bcc_k!(s.be,256)(s.U[6],s.U[7],s.U[8],ct.bxf,ct.byf,ct.bzf,Nx,Ny,Nz;ndrange=nc)
    KA.synchronize(s.be)
    s
end

# ── Diagnostics ──────────────────────────────────────────────────────────────
@kernel function _ct_divb_face_k!(d,@Const(bxf),@Const(byf),@Const(bzf),
        Nx::Int,Ny::Int,Nz::Int,invdx::T) where {T}
    c=@index(Global,Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        L(a,b,e)=_periodic_corr_index(Nx,Ny,Nz,a,b,e)
        d[c]=abs(((bxf[L(i+1,j,k)]-bxf[c])+(byf[L(i,j+1,k)]-byf[c])+
                  (bzf[L(i,j,k+1)]-bzf[c]))*invdx)
    end
end

"""
    max_divb_face(s) -> Float64

Maximum over cells of the exact face divergence `|Σ_d (b_d[upper] - b_d[lower])/dx|` of a
CT state — the constraint the CT solver preserves to round-off. (The cell-centred
[`max_divb`](@ref) is a different, centred-difference quantity.)
"""
function max_divb_face(s::MHDState{T}) where {T}
    uses_ct(s) || error("max_divb_face needs a CT state (mhd_solver=:ct)")
    Nx,Ny,Nz=s.dims; ct=s.ct
    d=device_zeros(s.be,T,(ncells(s),))
    _ct_divb_face_k!(s.be,256)(d,ct.bxf,ct.byf,ct.bzf,Nx,Ny,Nz,one(T)/s.dx;ndrange=ncells(s))
    KA.synchronize(s.be)
    Float64(maximum(d))
end

# ── Initialization ───────────────────────────────────────────────────────────
# cell B <- face average; E <- E + (|B_new|^2 - |B_old|^2)/2 (thermal pressure fixed); psi <- 0
@kernel function _ct_cell_from_faces_k!(U5,U6,U7,U8,U9,@Const(bxf),@Const(byf),@Const(bzf),
        Nx::Int,Ny::Int,Nz::Int)
    c=@index(Global,Linear)
    @inbounds begin
        T=eltype(U5)
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        L(a,b,e)=_periodic_corr_index(Nx,Ny,Nz,a,b,e)
        bx=T(0.5)*(bxf[c]+bxf[L(i+1,j,k)])
        by=T(0.5)*(byf[c]+byf[L(i,j+1,k)])
        bz=T(0.5)*(bzf[c]+bzf[L(i,j,k+1)])
        ox=U6[c]; oy=U7[c]; oz=U8[c]
        U5[c]+=T(0.5)*((bx*bx+by*by+bz*bz)-(ox*ox+oy*oy+oz*oz))
        U6[c]=bx; U7[c]=by; U8[c]=bz; U9[c]=zero(T)
    end
end

function _ct_install_faces!(s::MHDState{T}, bxf, byf, bzf) where {T}
    ct=s.ct; Nx,Ny,Nz=s.dims
    copyto!(ct.bxf,bxf); copyto!(ct.byf,byf); copyto!(ct.bzf,bzf)
    _ct_cell_from_faces_k!(s.be,256)(s.U[5],s.U[6],s.U[7],s.U[8],s.U[9],
        ct.bxf,ct.byf,ct.bzf,Nx,Ny,Nz;ndrange=ncells(s))
    KA.synchronize(s.be)
    s
end

# Half-cell phase θ = k dx/2 of rfft/fft index m (0-based) on an N-point axis; Nyquist
# (even N, m = N/2) is reported separately because cos θ = 0 there.
@inline function _ct_half_phase(m::Int,N::Int)
    nyq = iseven(N) && m == N÷2
    mm = m > N/2 ? m-N : m
    (nyq ? 0.0 : π*mm/N, nyq)
end

"""
    ct_faces_from_cells!(s::MHDState) -> NamedTuple

Build a discretely divergence-free face field from the cell-centred `B = s.U[6:8]` of a CT
state (host Float64 FFTW), changing the cell field as little as possible.

The CT cell field is the average of its two faces, so with `θ_d = k_d dx/2` a face mode
`f_d = b_d exp(-iθ_d)/cos θ_d` averages back to exactly `b_d`, and its face divergence
`Σ_d (exp(2iθ_d)-1) f_d/dx = (2i/dx) Σ_d tan θ_d b_d`. The cell field is therefore
projected orthogonally onto `t·b = 0` with `t_d = tan θ_d` (the smallest L2 change of the
cell field that admits divergence-free faces), and the faces are built from the result.
On a Nyquist plane of axis d (`cos θ_d = 0`) the face average of component d vanishes for
any face field, so that component is dropped there. Both operations are
Hermitian-symmetric, so the faces are real and divergence-free to round-off. (A plain
half-cell shift of the cell field followed by a face projection instead filters every
component by `cos θ_d` on the way back to the cell average: 7% of the magnetic energy of a
Batchelor field cut at 8 cells per wavelength.)

The centred-curl and continuum PMF fields are solenoidal under different symbols
(`sin 2θ`, `θ`) than `tan θ`, so the projection changes them by O((k dx)^2) near the
cutoff. Then `s.U[6:8]` become the face averages, `s.U[5]` is adjusted so the thermal
pressure is unchanged (`E += ½|B_new|² - ½|B_old|²`), and `s.U[9] = 0`.

Returns `(cell_b_rel_l2_change, magnetic_energy_ratio, max_divb_face_dx_over_brms)`.
"""
function ct_faces_from_cells!(s::MHDState{T}) where {T}
    uses_ct(s) || error("ct_faces_from_cells! needs a CT state (mhd_solver=:ct)")
    Nx,Ny,Nz=s.dims; dx=Float64(s.dx)
    Bold=ntuple(d->reshape(Float64.(to_host(s.U[5+d])),Nx,Ny,Nz),3)
    Bh=ntuple(d->rfft(Bold[d]),3)
    Fh=ntuple(d->similar(Bh[d]),3)
    nxr=Nx÷2+1
    px=[_ct_half_phase(m,Nx) for m in 0:nxr-1]
    py=[_ct_half_phase(m,Ny) for m in 0:Ny-1]
    pz=[_ct_half_phase(m,Nz) for m in 0:Nz-1]
    for kk in 1:Nz, jj in 1:Ny, ii in 1:nxr
        (θx,nx),(θy,ny),(θz,nz)=px[ii],py[jj],pz[kk]
        bx=nx ? zero(ComplexF64) : Bh[1][ii,jj,kk]
        by=ny ? zero(ComplexF64) : Bh[2][ii,jj,kk]
        bz=nz ? zero(ComplexF64) : Bh[3][ii,jj,kk]
        tx=tan(θx); ty=tan(θy); tz=tan(θz)
        t2=tx*tx+ty*ty+tz*tz
        if t2>0
            dot=(tx*bx+ty*by+tz*bz)/t2
            bx-=tx*dot; by-=ty*dot; bz-=tz*dot
        end
        Bh[1][ii,jj,kk]=bx; Bh[2][ii,jj,kk]=by; Bh[3][ii,jj,kk]=bz
        Fh[1][ii,jj,kk]=bx*cis(-θx)/cos(θx)
        Fh[2][ii,jj,kk]=by*cis(-θy)/cos(θy)
        Fh[3][ii,jj,kk]=bz*cis(-θz)/cos(θz)
    end
    faces=ntuple(d->vec(irfft(Fh[d],Nx)),3)
    _ct_install_faces!(s,to_device(s.be,faces[1],T),to_device(s.be,faces[2],T),
                       to_device(s.be,faces[3],T))
    Bnew=ntuple(d->reshape(Float64.(to_host(s.U[5+d])),Nx,Ny,Nz),3)
    num=sum(sum(abs2,Bnew[d].-Bold[d]) for d in 1:3)
    den=sum(sum(abs2,Bold[d]) for d in 1:3)
    enew=sum(sum(abs2,Bnew[d]) for d in 1:3)
    brms=sqrt(enew/length(Bnew[1]))
    (cell_b_rel_l2_change = den>0 ? sqrt(num/den) : 0.0,
     magnetic_energy_ratio = den>0 ? enew/den : 1.0,
     max_divb_face_dx_over_brms = brms>0 ? max_divb_face(s)*dx/brms : 0.0)
end

@kernel function _ct_curl_k!(bxf,byf,bzf,@Const(Ax),@Const(Ay),@Const(Az),
        Nx::Int,Ny::Int,Nz::Int,invdx::T) where {T}
    c=@index(Global,Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        L(a,b,e)=_periodic_corr_index(Nx,Ny,Nz,a,b,e)
        bxf[c]=((Az[L(i,j+1,k)]-Az[c])-(Ay[L(i,j,k+1)]-Ay[c]))*invdx
        byf[c]=((Ax[L(i,j,k+1)]-Ax[c])-(Az[L(i+1,j,k)]-Az[c]))*invdx
        bzf[c]=((Ay[L(i+1,j,k)]-Ay[c])-(Ax[L(i,j+1,k)]-Ax[c]))*invdx
    end
end

"""
    ct_faces_from_potential!(s::MHDState, Ax, Ay, Az)

Set the CT face field to the discrete curl of an EDGE-centred vector potential (arrays of
size `s.dims` or length `ncells`; cell `c=(i,j,k)` holds `Ax` at `(i,j-1/2,k-1/2)`, `Ay` at
`(i-1/2,j,k-1/2)`, `Az` at `(i-1/2,j-1/2,k)`, with the same lower-face convention as the
face fields). The face divergence is zero to round-off by construction. Then, as in
[`ct_faces_from_cells!`](@ref), cell B becomes the face average, `E` is adjusted so the
thermal pressure is unchanged, and `ψ = 0`.
"""
function ct_faces_from_potential!(s::MHDState{T}, Ax, Ay, Az) where {T}
    uses_ct(s) || error("ct_faces_from_potential! needs a CT state (mhd_solver=:ct)")
    Nx,Ny,Nz=s.dims
    for A in (Ax,Ay,Az)
        length(A)==ncells(s) || error("vector-potential array has $(length(A)) entries, expected $(ncells(s))")
    end
    dev(A)=vec(to_device(s.be,A isa Array ? A : Array(A),T))
    dAx=dev(Ax); dAy=dev(Ay); dAz=dev(Az)
    nc=ncells(s)
    bx=device_zeros(s.be,T,(nc,)); by=device_zeros(s.be,T,(nc,)); bz=device_zeros(s.be,T,(nc,))
    _ct_curl_k!(s.be,256)(bx,by,bz,dAx,dAy,dAz,Nx,Ny,Nz,one(T)/s.dx;ndrange=nc)
    KA.synchronize(s.be)
    _ct_install_faces!(s,bx,by,bz)
end
