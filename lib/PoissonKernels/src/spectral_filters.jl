# Native real FFT smoothing for host arrays and CuArrays. Metal's extension
# provides its fused MPSGraph method with the same cell-space Gaussian symbol.
const _GAUSSIAN_RFFT_CACHE = Dict{Any,Any}()

@kernel function _gaussian_rfft_filter_k!(hat, n1::Int, n2::Int, n3::Int,
                                         sigma::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        h1 = n1 ÷ 2 + 1
        q = c - 1
        i = q % h1
        j = (q ÷ h1) % n2
        k = q ÷ (h1*n2)
        j = j <= n2 ÷ 2 ? j : j-n2
        k = k <= n3 ÷ 2 ? k : k-n3
        q2 = (T(2pi)*T(i)/T(n1))^2 + (T(2pi)*T(j)/T(n2))^2 +
             (T(2pi)*T(k)/T(n3))^2
        hat[c] *= exp(-T(0.5)*sigma*sigma*q2)
    end
end

function rfft_smooth_gaussian!(dst::AbstractArray{T,3}, src::AbstractArray{T,3};
                                sigma_cells::Real, boxsize=1.0) where {T<:AbstractFloat}
    size(dst) == size(src) || throw(DimensionMismatch("Gaussian smoothing shapes differ"))
    isfinite(sigma_cells) && sigma_cells >= 0 || throw(ArgumentError("sigma_cells must be finite and nonnegative"))
    sigma_cells == 0 && return copyto!(dst,src)
    plan = get!(_GAUSSIAN_RFFT_CACHE,(typeof(src),size(src))) do
        hat = similar(src,Complex{T},_rfft_dims(size(src)))
        (hat,plan_rfft(src),plan_irfft(hat,size(src,1)))
    end
    hat,forward,inverse = plan
    be = KA.get_backend(src)
    mul!(hat,forward,src)
    _gaussian_rfft_filter_k!(be,256)(hat,size(src)...,T(sigma_cells);ndrange=length(hat))
    KA.synchronize(be)
    mul!(dst,inverse,hat)
    dst
end

# CPU/CUDA exponential-integrator step for the terminal pressure-implicit solve:
#   out_k = exp(-x(k)) state_k + dt (1 - exp(-x(k)))/x(k) source_k,
#   x(k) = coeff_cells2 (sin^2 qx + sin^2 qy + sin^2 qz),
# i.e. the centred finite-difference diffusion symbol, identical to the Metal
# MPSGraph method (`_exponential_source_filters_k!` in the Metal extension).
# `dst` may alias neither input in a way that matters: both are transformed first.
const _EXPSRC_RFFT_CACHE = Dict{Any,Any}()

@kernel function _exponential_source_rfft_k!(hs, hq, n1::Int, n2::Int, n3::Int,
                                             coeff::T, dt::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        h1 = n1 ÷ 2 + 1
        q = c - 1
        i = q % h1
        j = (q ÷ h1) % n2
        k = q ÷ (h1*n2)
        j = j <= n2 ÷ 2 ? j : j-n2
        k = k <= n3 ÷ 2 ? k : k-n3
        q2 = sin(T(2pi)*T(i)/T(n1))^2 + sin(T(2pi)*T(j)/T(n2))^2 +
             sin(T(2pi)*T(k)/T(n3))^2
        x = max(coeff, zero(T)) * q2
        e = exp(-x)
        fac = x > T(1.0f-6) ? dt * (one(T) - e) / x :
              dt * (one(T) - T(0.5) * x + x*x / T(6))
        hq[c] = e * hq[c] + fac * hs[c]
    end
end

function rfft_exponential_source!(dst::Array{T,3}, state::Array{T,3}, source::Array{T,3};
                                  coeff_cells2::Real, dt::Real,
                                  boxsize=1.0) where {T<:AbstractFloat}
    size(dst) == size(state) == size(source) ||
        throw(DimensionMismatch("exponential-source shapes differ"))
    isfinite(coeff_cells2) && coeff_cells2 >= 0 ||
        throw(ArgumentError("coeff_cells2 must be finite and non-negative"))
    isfinite(dt) && dt >= 0 || throw(ArgumentError("dt must be finite and non-negative"))
    dt == 0 && return copyto!(dst, state)
    plan = get!(_EXPSRC_RFFT_CACHE, (T, size(state))) do
        hq = similar(state, Complex{T}, _rfft_dims(size(state)))
        hs = similar(hq)
        (hq, hs, plan_rfft(state), plan_irfft(hq, size(state, 1)))
    end
    hq, hs, forward, inverse = plan
    mul!(hq, forward, state)
    mul!(hs, forward, source)
    be = KA.get_backend(state)
    _exponential_source_rfft_k!(be, 256)(hs, hq, size(state)..., T(coeff_cells2), T(dt);
                                         ndrange=length(hq))
    KA.synchronize(be)
    mul!(dst, inverse, hq)
    dst
end
