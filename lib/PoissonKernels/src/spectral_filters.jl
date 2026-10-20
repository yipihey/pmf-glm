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
