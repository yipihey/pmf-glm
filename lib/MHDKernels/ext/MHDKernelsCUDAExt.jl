module MHDKernelsCUDAExt

using MHDKernels
using CUDA

function __init__()
    if CUDA.functional()
        MHDKernels.register_backend!(:cuda, CUDABackend())
    end
end

MHDKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

# The historical native cube predates the Val-based Riemann interface and the
# compensated/source-aware updates. Production uses the shared KA algorithm.
function MHDKernels._cube_launch!(::Val{:auto}, be::CUDABackend,
        s::MHDKernels.MHDState, dt::Real, ch::Real, decay::Real)
    MHDKernels._cube_launch!(Val(:ka), be, s, dt, ch, decay)
end

function MHDKernels._cube_launch!(::Val{:raw}, ::CUDABackend,
        ::MHDKernels.MHDState, ::Real, ::Real, ::Real)
    error("CUDA impl=:raw is disabled: the historical native kernel is stale; " *
          "use impl=:ka or impl=:auto for the fused KernelAbstractions kernel")
end

MHDKernels._radiation_fft_backend_key(x::CuArray) =
    (CUDABackend, CUDA.device(x), CUDA.context(x))

# Transform only spatial axes. The fourth dimension is the photon/velocity
# batch, not a fourth spatial axis. Planning on CuArray dispatches to cuFFT;
# no host staging or per-channel device copies are involved.
function MHDKernels._radiation_forward_batch!(
        chat::CuArray{Complex{T},4}, x::CuArray{T,4}) where {T}
    forward, _ = MHDKernels._radiation_fft_plans(x, chat)
    MHDKernels.FFTW.mul!(chat, forward, x)
    chat
end

function MHDKernels._radiation_inverse_batch!(
        x::CuArray{T,4}, chat::CuArray{Complex{T},4}) where {T}
    _, inverse = MHDKernels._radiation_fft_plans(x, chat)
    MHDKernels.FFTW.mul!(x, inverse, chat)
    x
end

end # module
