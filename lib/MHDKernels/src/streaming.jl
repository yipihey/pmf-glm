# Peculiar baryon minus DM velocities, sampled at the actual particle positions.
export boost_gas_velocity!, sample_baryon_dm_velocity!

@kernel function _boost_gas_velocity_k!(rho, mx, my, mz, E, ux, uy, uz)
    p = @index(Global)
    @inbounds begin
        r = rho[p]
        E[p] += ux*mx[p] + uy*my[p] + uz*mz[p] + r*(ux^2+uy^2+uz^2)/2
        mx[p] += r*ux
        my[p] += r*uy
        mz[p] += r*uz
    end
end

"Add a uniform code velocity while preserving gas internal and magnetic energy."
function boost_gas_velocity!(s::MHDState{T}, velocity) where T
    length(velocity) == 3 && all(isfinite, velocity) ||
        throw(ArgumentError("boost requires three finite components"))
    _boost_gas_velocity_k!(s.be)(s.U[1],s.U[2],s.U[3],s.U[4],s.U[5],
                                T.(velocity)...; ndrange=ncells(s))
    s
end

@kernel function _sample_baryon_dm_velocity_k!(outx, outy, outz,
        @Const(rho), @Const(mx), @Const(my), @Const(mz),
        @Const(px), @Const(py), @Const(pz), @Const(vx), @Const(vy), @Const(vz),
        N::Int, lattice::Bool, gas_unit, dm_unit, smallr)
    p = @index(Global)
    @inbounds begin
        T = eltype(outx)
        q = p-1
        # Work with offsets separately to retain tiny lattice displacements.
        bx = lattice ? q % N : 0
        by = lattice ? (q ÷ N) % N : 0
        bz = lattice ? q ÷ (N*N) : 0
        ux = T(N)*px[p] - (lattice ? zero(T) : T(0.5))
        uy = T(N)*py[p] - (lattice ? zero(T) : T(0.5))
        uz = T(N)*pz[p] - (lattice ? zero(T) : T(0.5))
        ix = floor(Int,ux); iy = floor(Int,uy); iz = floor(Int,uz)
        fx = ux-T(ix); fy = uy-T(iy); fz = uz-T(iz)
        gx = zero(T); gy = zero(T); gz = zero(T)
        for k in 0:1, j in 0:1, i in 0:1
            c = mod(bx+ix+i,N) + N*mod(by+iy+j,N) + N*N*mod(bz+iz+k,N) + 1
            w = (i==0 ? 1-fx : fx)*(j==0 ? 1-fy : fy)*(k==0 ? 1-fz : fz)
            r = max(rho[c],smallr)
            gx += w*mx[c]/r; gy += w*my[c]/r; gz += w*mz[c]/r
        end
        outx[p] = gas_unit*gx-dm_unit*vx[p]
        outy[p] = gas_unit*gy-dm_unit*vy[p]
        outz[p] = gas_unit*gz-dm_unit*vz[p]
    end
end

"""Sample v_b - v_c at DM positions using periodic CIC of gas velocity.

Outputs have the units supplied by gas_unit and dm_unit (normally cm/s).
Equal-mass particle averages are DM-mass weighted, not Eulerian volume averages;
these arrays must not be FFTed on the original lattice after particles move.
"""
function sample_baryon_dm_velocity!(out, s, positions, velocities;
        lattice_displacements::Bool=true, gas_unit::Real, dm_unit::Real)
    N = s.dims[1]
    s.dims == (N,N,N) || throw(ArgumentError("streaming sampler requires a cube"))
    all(x->isfinite(x) && x>0,(gas_unit,dm_unit)) ||
        throw(ArgumentError("velocity units must be finite and positive"))
    length(out)==length(positions)==length(velocities)==3 ||
        throw(ArgumentError("require three vector components"))
    all(a->length(a)==ncells(s),(out...,positions...,velocities...)) ||
        throw(ArgumentError("sampler requires one particle per original lattice cell"))
    T = eltype(s.U[1])
    _sample_baryon_dm_velocity_k!(s.be)(out...,s.U[1],s.U[2],s.U[3],s.U[4],
        positions...,velocities...,N,lattice_displacements,T(gas_unit),T(dm_unit),s.smallr;
        ndrange=ncells(s))
    out
end
