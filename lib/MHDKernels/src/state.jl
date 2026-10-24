# ── MHDState: 9 SoA conserved fields + a ping-pong scratch set + run parameters ─
# Fields are flat column-major length-Nx·Ny·Nz device vectors; cell (i,j,k) lives at
#   idx = (k-1)*Nx*Ny + (j-1)*Nx + i   (1-based).
# Boundaries are PERIODIC for now (the smooth-wave convergence gate needs nothing
# else); `bc` is carried for the outflow/reflecting extension. The conserved order is
# U=(ρ,ρvx,ρvy,ρvz,E,Bx,By,Bz,ψ); ψ is the GLM cleaning potential (damped, not
# conserved). γ is the adiabatic index; the floors/Riemann-switch params feed
# `riemann`. `be` is the KA backend the fields live on.

export MHDState, allocate_state, nvars, ncells, linidx, fields_to_host
export CTState, uses_ct, mhd_solver

const NVAR = 9

"""
    CTState{A}

Face-staggered constrained-transport (CT) storage attached to an [`MHDState`](@ref)
allocated with `mhd_solver=:ct` (see `ct.jl` for the scheme). All fields are flat
length-`ncells` device vectors indexed like the cell fields, cell `c=(i,j,k)`:

  * `bxf, byf, bzf` — the evolved face fields. `bxf[c]` is `Bx` on the LOWER-x face
    `(i-1/2,j,k)` of cell `c`; likewise `byf` at `(i,j-1/2,k)`, `bzf` at `(i,j,k-1/2)`.
  * `mfx, mfy, mfz` — per-step mass flux through the lower x/y/z face of each cell.
  * `fxby, fxbz` — lower-x-face magnetic fluxes `F_x[By]`, `F_x[Bz]`;
    `fybx, fybz` — lower-y-face `F_y[Bx]`, `F_y[Bz]`; `fzbx, fzby` — lower-z-face
    `F_z[Bx]`, `F_z[By]`.
  * `ecx, ecy, ecz` — cell-centred reference EMF `E = -v x B`.
  * `ex, ey, ez` — edge EMFs: `ez[c]` at `(i-1/2,j-1/2,k)`, `ex[c]` at
    `(i,j-1/2,k-1/2)`, `ey[c]` at `(i-1/2,j,k-1/2)`.

All 18 are contiguous views, in this order, of ONE device buffer `packed` of length
`18*ncells`: the fused Godunov kernel takes that single buffer (slot `q` of cell `c` at
`(q-1)*ncells + c`), because Metal allows a kernel 31 buffer arguments and counts every
scalar argument as one: with 18 separate CT arrays it failed with 38/31.

(The struct lives in `state.jl` because `MHDState` has a field of this type.)
"""
struct CTState{A,V}
    packed::A
    bxf::V; byf::V; bzf::V
    mfx::V; mfy::V; mfz::V
    fxby::V; fxbz::V; fybx::V; fybz::V; fzbx::V; fzby::V
    ecx::V; ecy::V; ecz::V
    ex::V; ey::V; ez::V
end

function CTState(packed::AbstractVector, nc::Int)
    length(packed) == 18nc || error("CT packed buffer has $(length(packed)) entries, expected $(18nc)")
    v = ntuple(q -> view(packed, (q-1)*nc+1:q*nc), 18)
    CTState{typeof(packed),eltype(v)}(packed, v...)
end

"`ct_arrays(ct)` — the 18 CT field views as a tuple, in field order."
ct_arrays(c::CTState) = ntuple(i -> getfield(c, i + 1), 18)

mutable struct MHDState{T,A,B}
    U::NTuple{NVAR,A}        # conserved
    scratch::NTuple{NVAR,A}  # ping-pong target for the integrator
    dims::NTuple{3,Int}      # (Nx, Ny, Nz)
    dx::T
    γ::T
    smallr::T
    pfl::T
    llf_dmin::T
    llf_pmin::T
    use_hlld::Bool
    riemann::Symbol
    bcs::NTuple{3,Symbol}    # per-axis: :periodic / :outflow / :reflecting
    recon::Symbol            # :plm (MonCen), :plm_minmod, or :ppm (local CW84)
    be::B
    ct::Union{Nothing,CTState{A}}   # (CTState{A,V} for any view type V) face-staggered CT storage (`mhd_solver=:ct`), else nothing
end

# GLM-only construction (the historic 14-field form).
MHDState{T,A,B}(U, scratch, dims, dx, γ, smallr, pfl, llf_dmin, llf_pmin, use_hlld,
                riemann, bcs, recon, be) where {T,A,B} =
    MHDState{T,A,B}(U, scratch, dims, dx, γ, smallr, pfl, llf_dmin, llf_pmin, use_hlld,
                    riemann, bcs, recon, be, nothing)

"`uses_ct(s)` — true when `s` carries face-staggered CT storage (`mhd_solver=:ct`)."
uses_ct(s::MHDState) = s.ct !== nothing

"`mhd_solver(s)` — `:ct` (constrained transport) or `:glm` (Dedner GLM cleaning, the default)."
mhd_solver(s::MHDState) = uses_ct(s) ? :ct : :glm

nvars(::MHDState) = NVAR
ncells(s::MHDState) = prod(s.dims)
@inline linidx(Nx::Int, Ny::Int, i::Int, j::Int, k::Int) = ((k-1)*Ny + (j-1))*Nx + i

"`bc_codes(s)` → the per-axis BC integer codes (cx,cy,cz) for the kernels."
bc_codes(s::MHDState) = (bc_code(s.bcs[1]), bc_code(s.bcs[2]), bc_code(s.bcs[3]))

"`recon_code_of(s)` → reconstruction code (0=MonCen PLM, 1=PPM, 2=minmod PLM)."
recon_code_of(s::MHDState) = recon_code(s.recon)

"`riemann_code_of(s)` → the compile-time Riemann code (:hlld, :hll, :llf)."
riemann_code_of(s::MHDState) = riemann_code(s.riemann)

"`all_periodic(s)` → true when every axis is periodic (enables the cube's fast load path)."
all_periodic(s::MHDState) = all(==(:periodic), s.bcs)

"""
    allocate_state(be, T, dims; dx, gamma=5/3, smallr=1e-6, pfl=1e-7,
                   llf_dmin=1e-4, llf_pmin=0, use_hlld=true, riemann=nothing,
                   bc=:periodic, bcs=(bc,bc,bc), recon=:plm, mhd_solver=:glm)

Allocate an `MHDState` of element type `T` and shape `dims=(Nx,Ny,Nz)` on backend
`be` (zero-filled; fill it with a problem initialiser). `dx` is the uniform cell
width (domain assumed `[0,L]^3` with L = Nx·dx). `bc` sets all three axes; `bcs`
overrides per-axis (e.g. `bcs=(:outflow,:periodic,:periodic)` for a 1-D shock tube).

`mhd_solver` selects the divergence treatment: `:glm` (default; Dedner cleaning with
the ψ field) or `:ct` (face-staggered constrained transport, see `ct.jl`; allocates
3 face + 15 workspace arrays and requires all-periodic BCs and the `:ref` integrator).
"""
function allocate_state(be, ::Type{T}, dims::NTuple{3,Int};
                        dx::Real, gamma::Real = 5//3,
                        smallr::Real = 1e-6, pfl::Real = 1e-7,
                        llf_dmin::Real = 1e-4, llf_pmin::Real = 0,
                        use_hlld::Bool = true, riemann = nothing, bc::Symbol = :periodic,
                        bcs::NTuple{3,Symbol} = (bc, bc, bc), recon::Symbol = :plm,
                        mhd_solver::Symbol = :glm) where {T}
    mhd_solver in (:glm, :ct) ||
        error("unknown mhd_solver :$mhd_solver (have :glm, :ct)")
    mhd_solver === :ct && !all(==(:periodic), bcs) &&
        error("mhd_solver=:ct requires all-periodic boundaries (bcs=$bcs); constrained transport is periodic-only")
    nc = prod(dims)
    U  = ntuple(_ -> device_zeros(be, T, (nc,)), NVAR)
    sc = ntuple(_ -> device_zeros(be, T, (nc,)), NVAR)
    rsym = riemann === nothing ? (use_hlld ? :hlld : :llf) : Symbol(riemann)
    ct = mhd_solver === :ct ? CTState(device_zeros(be, T, (18nc,)), nc) : nothing
    MHDState{T,eltype(U),typeof(be)}(U, sc, dims, T(dx), T(gamma), T(smallr), T(pfl),
                                     T(llf_dmin), T(llf_pmin), use_hlld, rsym, bcs, recon, be, ct)
end

"`fields_to_host(s)` → an `NTuple{9,Array}` host copy of the conserved fields."
fields_to_host(s::MHDState) = ntuple(v -> to_host(s.U[v]), NVAR)
