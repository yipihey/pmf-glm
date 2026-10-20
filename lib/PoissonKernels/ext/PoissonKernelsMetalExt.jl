"""
    PoissonKernelsMetalExt

Package extension that lights up the Metal (Apple GPU) backend for `PoissonKernels`.
Loaded automatically when `Metal` is present in the environment. Registers the
`:metal` backend and specialises the device-array helpers onto `MtlArray`.
Metal is Float32-only, so callers must request `Float32` element types.
"""
module PoissonKernelsMetalExt

using PoissonKernels
using Metal
using KernelAbstractions: @kernel, @index, @Const, @atomic

const MG = Metal.MPSGraphs
using Metal.MPSGraphs: MPSGraph, MPSGraphFFTDescriptor, MPSGraphTensor,
    MPSGraphTensorData, placeholderTensor, id, NSDictionary, NSArray, NSNumber,
    NSString

function __init__()
    # Only register a usable GPU if the system actually has one (CI on Apple
    # hardware without a functional Metal device should degrade gracefully).
    if Metal.functional()
        PoissonKernels.register_backend!(:metal, Metal.MetalBackend())
    end
end

# `Metal.zeros(T, dims)` allocates a zero-filled MtlArray on the default device.
PoissonKernels.device_zeros(::Metal.MetalBackend, ::Type{T}, dims::Dims) where {T} =
    Metal.zeros(T, dims)

struct _MPSRFFTPlan
    graph::Any
    rho::Any
    gk::Any
    coef::Any
    out::Any
    Gk::Any
    coefd::Any
    coefh::Vector{Float32}
end

const _MPS_RFFT_CACHE = Dict{Any,_MPSRFFTPlan}()

struct _MPSForwardRFFTPlan
    graph::Any
    rho::Any
    out::Any
    chat::Any
end

const _MPS_FORWARD_RFFT_CACHE = Dict{Any,_MPSForwardRFFTPlan}()

struct _MPSSmoothGaussianPlan
    graph::Any
    src::Any
    filt::Any
    out::Any
    filtd::Any
end

const _MPS_SMOOTH_GAUSSIAN_CACHE = Dict{Any,_MPSSmoothGaussianPlan}()

struct _MPSImplicitDiffusionPlan
    graph::Any
    src::Any
    filt::Any
    out::Any
    filtd::Any
end

const _MPS_IMPLICIT_DIFFUSION_CACHE = Dict{Any,_MPSImplicitDiffusionPlan}()

struct _MPSExponentialSourcePlan
    graph::Any
    state::Any
    source::Any
    decay::Any
    sourcefac::Any
    out::Any
    decayd::Any
    sourcefacd::Any
end

const _MPS_EXPONENTIAL_SOURCE_CACHE = Dict{Any,_MPSExponentialSourcePlan}()

function _fft_descriptor(; inverse::Bool, scaling = MG.MPSGraphFFTScalingModeNone)
    desc = MPSGraphFFTDescriptor(MG.@objc [MPSGraphFFTDescriptor descriptor]::id{MPSGraphFFTDescriptor})
    desc.inverse = inverse
    desc.scalingMode = scaling
    desc.roundToOddHermitean = false
    return desc
end

function _real_to_hermitean(g, x, axes, desc)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} realToHermiteanFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"rfft"::id{NSString}]::id{MPSGraphTensor})
end

function _hermitean_to_real(g, x, axes, desc)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} HermiteanToRealFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"irfft"::id{NSString}]::id{MPSGraphTensor})
end

function _real_part(g, x)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} realPartOfTensor:x::id{MPSGraphTensor} name:"real"::id{NSString}]::id{MPSGraphTensor})
end

function _imag_part(g, x)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} imaginaryPartOfTensor:x::id{MPSGraphTensor} name:"imag"::id{NSString}]::id{MPSGraphTensor})
end

function _complex_tensor(g, r, i)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} complexTensorWithRealTensor:r::id{MPSGraphTensor} imaginaryTensor:i::id{MPSGraphTensor} name:"complex"::id{NSString}]::id{MPSGraphTensor})
end

function _mps_rfft_plan(::Type{Float32}, N::NTuple{3,Int},
                        L::NTuple{3,Float64}, greens::Symbol)
    get!(_MPS_RFFT_CACHE, (Float32, N, L, greens)) do
        half = PoissonKernels._rfft_dims(N)
        gh = greens === :spectral ? PoissonKernels._greens_periodic(Float32, N, L) :
             greens === :discrete7 ? PoissonKernels._greens_discrete7(Float32, N, L) :
             greens === :centered ? PoissonKernels._greens_centered(Float32, N, L) :
             error("fft_poisson_rfft!: greens must be :spectral, :discrete7, or :centered")

        graph = MPSGraph()
        rho = placeholderTensor(graph, N, Float32, "rho")
        gk = placeholderTensor(graph, half, Float32, "gk")
        coef = placeholderTensor(graph, (1,), Float32, "coef")

        axes = NSArray(NSNumber.([0, 1, 2]))
        fdesc = _fft_descriptor(inverse=false)
        idesc = _fft_descriptor(inverse=true, scaling=MG.MPSGraphFFTScalingModeSize)

        rhok = _real_to_hermitean(graph, rho, axes, fdesc)
        gcoef = MG.multiplicationWithPrimaryTensor(graph, gk, coef, "gcoef")
        phr = MG.multiplicationWithPrimaryTensor(graph, _real_part(graph, rhok), gcoef, "mulr")
        phi = MG.multiplicationWithPrimaryTensor(graph, _imag_part(graph, rhok), gcoef, "muli")
        phik = _complex_tensor(graph, phr, phi)
        out = _hermitean_to_real(graph, phik, axes, idesc)

        _MPSRFFTPlan(graph, rho, gk, coef, out, Metal.MtlArray(gh),
                     Metal.MtlArray(Float32[1]), Float32[1])
    end
end

function _mps_rfft_plan(::Type{T}, N::NTuple{3,Int},
                        L::NTuple{3,Float64}, greens::Symbol) where {T}
    error("Metal MPSGraph FFT supports Float32 only, got $T for size $N")
end

function _mps_forward_rfft_plan(::Type{Float32}, N::NTuple{3,Int})
    get!(_MPS_FORWARD_RFFT_CACHE, (Float32, N)) do
        half = PoissonKernels._rfft_dims(N)
        graph = MPSGraph()
        rho = placeholderTensor(graph, N, Float32, "rho")
        axes = NSArray(NSNumber.([0, 1, 2]))
        out = _real_to_hermitean(graph, rho, axes, _fft_descriptor(inverse=false))
        _MPSForwardRFFTPlan(graph, rho, out, Metal.zeros(ComplexF32, half))
    end
end

function _mps_forward_rfft!(x::Metal.MtlArray{Float32,3}, chat=nothing)
    P = _mps_forward_rfft_plan(Float32, size(x))
    chat === nothing && (chat = P.chat)
    MG.@autoreleasepool begin
        feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(P.rho => MPSGraphTensorData(x))
        results = Dict{MPSGraphTensor,MPSGraphTensorData}(P.out => MPSGraphTensorData(chat))
        cmdbuf = MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cmdbuf, P.graph, NSDictionary(feeds), NSDictionary(results),
                   MG.nil, MG.MPSGraphExecutionDescriptor())
        MG.commit!(cmdbuf)
        MG.wait_completed(cmdbuf)
    end
    return chat
end

function _gaussian_filter_half(::Type{Float32}, N::NTuple{3,Int}, sigma_cells::Float64)
    half = PoissonKernels._rfft_dims(N)
    n = N[1]
    filt = Array{Float32}(undef, half)
    for k in 0:N[3]-1, j in 0:N[2]-1, i in 0:half[1]-1
        kz = k <= N[3] ÷ 2 ? k : k - N[3]
        ky = j <= N[2] ÷ 2 ? j : j - N[2]
        kx = i
        q = (2π / n) * sqrt(kx*kx + ky*ky + kz*kz)
        filt[i + 1, j + 1, k + 1] = Float32(exp(-0.5 * (sigma_cells * q)^2))
    end
    return filt
end

function _mps_smooth_gaussian_plan(::Type{Float32}, N::NTuple{3,Int}, sigma_cells::Float64)
    get!(_MPS_SMOOTH_GAUSSIAN_CACHE, (Float32, N, sigma_cells)) do
        half = PoissonKernels._rfft_dims(N)
        filt_h = _gaussian_filter_half(Float32, N, sigma_cells)

        graph = MPSGraph()
        src = placeholderTensor(graph, N, Float32, "src")
        filt = placeholderTensor(graph, half, Float32, "filt")

        axes = NSArray(NSNumber.([0, 1, 2]))
        fdesc = _fft_descriptor(inverse=false)
        idesc = _fft_descriptor(inverse=true, scaling=MG.MPSGraphFFTScalingModeSize)

        srck = _real_to_hermitean(graph, src, axes, fdesc)
        outr = MG.multiplicationWithPrimaryTensor(graph, _real_part(graph, srck), filt, "smooth_r")
        outi = MG.multiplicationWithPrimaryTensor(graph, _imag_part(graph, srck), filt, "smooth_i")
        outk = _complex_tensor(graph, outr, outi)
        out = _hermitean_to_real(graph, outk, axes, idesc)

        _MPSSmoothGaussianPlan(graph, src, filt, out, Metal.MtlArray(filt_h))
    end
end

function _mps_smooth_gaussian_plan(::Type{T}, N::NTuple{3,Int}, sigma_cells::Float64) where {T}
    error("Metal MPSGraph Gaussian smoothing supports Float32 only, got $T for size $N")
end

function PoissonKernels.rfft_smooth_gaussian!(dst::Metal.MtlArray{T,3},
                                              src::Metal.MtlArray{T,3};
                                              sigma_cells::Real,
                                              boxsize=1.0) where {T}
    N = size(src)
    size(dst) == N || error("rfft_smooth_gaussian!: dst size $(size(dst)) != src size $N")
    all(==(N[1]), N) || error("rfft_smooth_gaussian!: cubic grid required, got $N")
    sigma = Float64(sigma_cells)
    sigma >= 0 || error("rfft_smooth_gaussian!: sigma_cells must be non-negative")
    if sigma == 0
        copyto!(dst, src)
        return dst
    end
    P = _mps_smooth_gaussian_plan(T, N, sigma)
    MG.@autoreleasepool begin
        feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.src => MPSGraphTensorData(src),
            P.filt => MPSGraphTensorData(P.filtd),
        )
        results = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.out => MPSGraphTensorData(dst),
        )
        cmdbuf = MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cmdbuf, P.graph, NSDictionary(feeds), NSDictionary(results),
                   MG.nil, MG.MPSGraphExecutionDescriptor())
        MG.commit!(cmdbuf)
        MG.wait_completed(cmdbuf)
    end
    return dst
end

@kernel function _implicit_diffusion_filter_k!(filt, n::Int32, coeff::Float32)
    p = @index(Global)
    h = (n >>> 1) + Int32(1)
    q = Int32(p - 1)
    i = q % h
    q = q ÷ h
    j = q % n
    k = q ÷ n
    @inbounds begin
        kz = k <= (n >>> 1) ? k : k - n
        ky = j <= (n >>> 1) ? j : j - n
        kx = i
        q2 = Float32((kx*kx + ky*ky + kz*kz)) * (Float32(2π) / Float32(n))^2
        filt[p] = inv(Float32(1) + max(coeff, Float32(0)) * q2)
    end
end

@kernel function _exponential_source_filters_k!(decay, sourcefac, n::Int32,
                                                coeff::Float32, dt::Float32)
    p = @index(Global)
    h = (n >>> 1) + Int32(1)
    q = Int32(p - 1)
    i = q % h
    q = q ÷ h
    j = q % n
    k = q ÷ n
    @inbounds begin
        kz = k <= (n >>> 1) ? k : k - n
        ky = j <= (n >>> 1) ? j : j - n
        kx = i
        qx = Float32(2π) * Float32(kx) / Float32(n)
        qy = Float32(2π) * Float32(ky) / Float32(n)
        qz = Float32(2π) * Float32(kz) / Float32(n)
        # Match the centered finite-difference pressure/source operator used by
        # the terminal update: D(k)^2 = sin(k dx)^2 in cell coordinates.
        q2 = sin(qx)^2 + sin(qy)^2 + sin(qz)^2
        x = max(coeff, Float32(0)) * q2
        e = exp(-x)
        decay[p] = e
        sourcefac[p] = x > Float32(1.0f-6) ? dt * (Float32(1) - e) / x :
                       dt * (Float32(1) - Float32(0.5) * x + x*x / Float32(6))
    end
end

function _mps_implicit_diffusion_plan(::Type{Float32}, N::NTuple{3,Int})
    get!(_MPS_IMPLICIT_DIFFUSION_CACHE, (Float32, N)) do
        half = PoissonKernels._rfft_dims(N)
        graph = MPSGraph()
        src = placeholderTensor(graph, N, Float32, "src")
        filt = placeholderTensor(graph, half, Float32, "filt")

        axes = NSArray(NSNumber.([0, 1, 2]))
        fdesc = _fft_descriptor(inverse=false)
        idesc = _fft_descriptor(inverse=true, scaling=MG.MPSGraphFFTScalingModeSize)

        srck = _real_to_hermitean(graph, src, axes, fdesc)
        outr = MG.multiplicationWithPrimaryTensor(graph, _real_part(graph, srck), filt, "diffuse_r")
        outi = MG.multiplicationWithPrimaryTensor(graph, _imag_part(graph, srck), filt, "diffuse_i")
        outk = _complex_tensor(graph, outr, outi)
        out = _hermitean_to_real(graph, outk, axes, idesc)

        _MPSImplicitDiffusionPlan(graph, src, filt, out, Metal.zeros(Float32, half))
    end
end

function _mps_exponential_source_plan(::Type{Float32}, N::NTuple{3,Int})
    get!(_MPS_EXPONENTIAL_SOURCE_CACHE, (Float32, N)) do
        half = PoissonKernels._rfft_dims(N)
        graph = MPSGraph()
        state = placeholderTensor(graph, N, Float32, "state")
        source = placeholderTensor(graph, N, Float32, "source")
        decay = placeholderTensor(graph, half, Float32, "decay")
        sourcefac = placeholderTensor(graph, half, Float32, "sourcefac")

        axes = NSArray(NSNumber.([0, 1, 2]))
        fdesc = _fft_descriptor(inverse=false)
        idesc = _fft_descriptor(inverse=true, scaling=MG.MPSGraphFFTScalingModeSize)

        statek = _real_to_hermitean(graph, state, axes, fdesc)
        sourcek = _real_to_hermitean(graph, source, axes, fdesc)
        outr_state = MG.multiplicationWithPrimaryTensor(graph, _real_part(graph, statek), decay, "exp_state_r")
        outi_state = MG.multiplicationWithPrimaryTensor(graph, _imag_part(graph, statek), decay, "exp_state_i")
        outr_source = MG.multiplicationWithPrimaryTensor(graph, _real_part(graph, sourcek), sourcefac, "exp_source_r")
        outi_source = MG.multiplicationWithPrimaryTensor(graph, _imag_part(graph, sourcek), sourcefac, "exp_source_i")
        outr = MG.additionWithPrimaryTensor(graph, outr_state, outr_source, "exp_r")
        outi = MG.additionWithPrimaryTensor(graph, outi_state, outi_source, "exp_i")
        outk = _complex_tensor(graph, outr, outi)
        out = _hermitean_to_real(graph, outk, axes, idesc)

        _MPSExponentialSourcePlan(graph, state, source, decay, sourcefac, out,
                                  Metal.zeros(Float32, half), Metal.zeros(Float32, half))
    end
end

function _mps_exponential_source_plan(::Type{T}, N::NTuple{3,Int}) where {T}
    error("Metal MPSGraph exponential source supports Float32 only, got $T for size $N")
end

function _mps_implicit_diffusion_plan(::Type{T}, N::NTuple{3,Int}) where {T}
    error("Metal MPSGraph implicit diffusion supports Float32 only, got $T for size $N")
end

function PoissonKernels.rfft_implicit_diffuse!(dst::Metal.MtlArray{T,3},
                                               src::Metal.MtlArray{T,3};
                                               coeff_cells2::Real,
                                               boxsize=1.0) where {T}
    N = size(src)
    size(dst) == N || error("rfft_implicit_diffuse!: dst size $(size(dst)) != src size $N")
    all(==(N[1]), N) || error("rfft_implicit_diffuse!: cubic grid required, got $N")
    coeff = Float64(coeff_cells2)
    coeff >= 0 || error("rfft_implicit_diffuse!: coeff_cells2 must be non-negative")
    if coeff == 0
        copyto!(dst, src)
        return dst
    end
    P = _mps_implicit_diffusion_plan(T, N)
    _implicit_diffusion_filter_k!(Metal.MetalBackend())(P.filtd, Int32(N[1]), Float32(coeff);
                                                        ndrange=length(P.filtd))
    Metal.synchronize()
    MG.@autoreleasepool begin
        feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.src => MPSGraphTensorData(src),
            P.filt => MPSGraphTensorData(P.filtd),
        )
        results = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.out => MPSGraphTensorData(dst),
        )
        cmdbuf = MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cmdbuf, P.graph, NSDictionary(feeds), NSDictionary(results),
                   MG.nil, MG.MPSGraphExecutionDescriptor())
        MG.commit!(cmdbuf)
        MG.wait_completed(cmdbuf)
    end
    return dst
end

function PoissonKernels.rfft_exponential_source!(dst::Metal.MtlArray{T,3},
                                                 state::Metal.MtlArray{T,3},
                                                 source::Metal.MtlArray{T,3};
                                                 coeff_cells2::Real,
                                                 dt::Real,
                                                 boxsize=1.0) where {T}
    N = size(state)
    size(dst) == N || error("rfft_exponential_source!: dst size $(size(dst)) != state size $N")
    size(source) == N || error("rfft_exponential_source!: source size $(size(source)) != state size $N")
    all(==(N[1]), N) || error("rfft_exponential_source!: cubic grid required, got $N")
    coeff = Float64(coeff_cells2)
    coeff >= 0 || error("rfft_exponential_source!: coeff_cells2 must be non-negative")
    dtt = Float64(dt)
    isfinite(dtt) && dtt >= 0 || error("rfft_exponential_source!: dt must be non-negative")
    if dtt == 0
        copyto!(dst, state)
        return dst
    end
    P = _mps_exponential_source_plan(T, N)
    _exponential_source_filters_k!(Metal.MetalBackend())(P.decayd, P.sourcefacd,
                                                         Int32(N[1]), Float32(coeff),
                                                         Float32(dtt);
                                                         ndrange=length(P.decayd))
    Metal.synchronize()
    MG.@autoreleasepool begin
        feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.state => MPSGraphTensorData(state),
            P.source => MPSGraphTensorData(source),
            P.decay => MPSGraphTensorData(P.decayd),
            P.sourcefac => MPSGraphTensorData(P.sourcefacd),
        )
        results = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.out => MPSGraphTensorData(dst),
        )
        cmdbuf = MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cmdbuf, P.graph, NSDictionary(feeds), NSDictionary(results),
                   MG.nil, MG.MPSGraphExecutionDescriptor())
        MG.commit!(cmdbuf)
        MG.wait_completed(cmdbuf)
    end
    return dst
end

function PoissonKernels.clear_fft_scratch!(::Metal.MetalBackend)
    Metal.synchronize()
    if isdefined(Metal, :unsafe_free!)
        for P in values(_MPS_FORWARD_RFFT_CACHE)
            P.chat === nothing || Metal.unsafe_free!(P.chat)
        end
        for P in values(_MPS_SMOOTH_GAUSSIAN_CACHE)
            P.filtd === nothing || Metal.unsafe_free!(P.filtd)
        end
        for P in values(_MPS_IMPLICIT_DIFFUSION_CACHE)
            P.filtd === nothing || Metal.unsafe_free!(P.filtd)
        end
        for P in values(_MPS_EXPONENTIAL_SOURCE_CACHE)
            P.decayd === nothing || Metal.unsafe_free!(P.decayd)
            P.sourcefacd === nothing || Metal.unsafe_free!(P.sourcefacd)
        end
    end
    empty!(_MPS_FORWARD_RFFT_CACHE)
    empty!(_MPS_SMOOTH_GAUSSIAN_CACHE)
    empty!(_MPS_IMPLICIT_DIFFUSION_CACHE)
    empty!(_MPS_EXPONENTIAL_SOURCE_CACHE)
    return nothing
end

@kernel function _pkmu_bin_rfft_k!(psum, ksum, cnt, @Const(chat),
                                   n::Int32, nb::Int32, nμ::Int32, axis::Int32,
                                   kf::Float32, kny::Float32, scale::Float32,
                                   count_modes::Int32)
    p = @index(Global)
    h = (n >>> 1) + Int32(1)
    q = Int32(p - 1)
    i = q % h
    q = q ÷ h
    j = q % n
    k = q ÷ n
    @inbounds begin
        kz = k <= (n >>> 1) ? k : k - n
        ky = j <= (n >>> 1) ? j : j - n
        kx = i
        km2 = kx*kx + ky*ky + kz*kz
        if km2 > 0
            km = sqrt(Float32(km2))
            if km <= kny
                ka = axis == Int32(1) ? kx : axis == Int32(2) ? ky : kz
                b = Int32(1) + unsafe_trunc(Int32, floor((km / kny) * Float32(nb)))
                b = min(max(b, Int32(1)), nb)
                mb = Int32(1) + unsafe_trunc(Int32, floor((abs(Float32(ka)) / km) * Float32(nμ)))
                mb = min(max(mb, Int32(1)), nμ)
                mult = (i == 0 || i == (n >>> 1)) ? Float32(1) : Float32(2)
                mult_i = (i == 0 || i == (n >>> 1)) ? UInt32(1) : UInt32(2)
                z = chat[p]
                pow = mult * scale * (real(z)*real(z) + imag(z)*imag(z))
                @atomic psum[Int(b), Int(mb)] + pow
                if count_modes == Int32(1)
                    @atomic ksum[Int(b)] + (mult * km * kf)
                    @atomic cnt[Int(b), Int(mb)] + mult_i
                end
            end
        end
    end
end

function PoissonKernels.fft_poisson_rfft!(φ::Metal.MtlArray{T,3},
                                          ρ::Metal.MtlArray{T,3};
                                          G::Real=1.0, a::Real=1.0,
                                          boxsize=1.0,
                                          greens::Symbol=:spectral) where {T}
    N = size(ρ)
    size(φ) == N || error("fft_poisson_rfft!: φ size $(size(φ)) != ρ size $N")
    trace = get(ENV, "CIC_MPS_RFFT_TRACE", "0") == "1"
    live0 = if trace
        s0 = Metal.alloc_stats
        max(0, s0.alloc_bytes - s0.free_bytes) / 2^30
    else
        0.0
    end
    t0 = time()
    L = boxsize isa Number ? ntuple(_ -> Float64(boxsize), 3) :
        ntuple(d -> Float64(boxsize[d]), 3)
    P = _mps_rfft_plan(T, N, L, greens)
    tplan = time()
    P.coefh[1] = Float32(G) / Float32(a)
    copyto!(P.coefd, P.coefh)
    tcoef = time()

    tprep = tcoef
    tenc = tcoef
    twait = 0.0
    MG.@autoreleasepool begin
        feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.rho => MPSGraphTensorData(ρ),
            P.gk => MPSGraphTensorData(P.Gk),
            P.coef => MPSGraphTensorData(P.coefd),
        )
        results = Dict{MPSGraphTensor,MPSGraphTensorData}(
            P.out => MPSGraphTensorData(φ),
        )
        cmdbuf = MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        tprep = time()
        MG.encode!(cmdbuf, P.graph, NSDictionary(feeds), NSDictionary(results),
                   MG.nil, MG.MPSGraphExecutionDescriptor())
        MG.commit!(cmdbuf)
        tenc = time()
        MG.wait_completed(cmdbuf)
        twait = time() - tenc
    end
    if trace
        s = Metal.alloc_stats
        live = max(0, s.alloc_bytes - s.free_bytes) / 2^30
        total = time() - t0
        println("  MPS_RFFT_TRACE default N=$(N[1]) live=$(round(live0; digits=2))->$(round(live; digits=2)) GiB",
                " plan=$(round(tplan-t0; digits=4))",
                " coef=$(round(tcoef-tplan; digits=4))",
                " prep=$(round(tprep-tcoef; digits=4))",
                " encode=$(round(tenc-tprep; digits=4))",
                " wait=$(round(twait; digits=4))",
                " total=$(round(total; digits=4))")
        flush(stdout)
    end
    return φ
end

function _power_spectrum_aniso_mps(fields::Tuple; boxsize=1.0, nmu::Integer=4,
                                   nbins::Integer=0, axis::Integer=1,
                                   kmax::Union{Nothing,Real}=nothing)
    isempty(fields) && error("power_spectrum_aniso_gpu needs at least one field")
    f1 = fields[1]
    N = size(f1)
    all(size(f) == N for f in fields) || error("power_spectrum_aniso_gpu field sizes must match")
    all(f -> f isa Metal.MtlArray{Float32,3}, fields) ||
        error("Metal power_spectrum_aniso_gpu supports Float32 MtlArray fields")
    all(==(N[1]), N) || error("power_spectrum_aniso_gpu needs a cubic grid, got $N")

    n = N[1]
    L = Float64(boxsize isa Number ? boxsize : boxsize[1])
    nb = nbins > 0 ? Int(nbins) : n ÷ 2
    nμ = Int(nmu)
    kf = 2π / L
    kny_grid = n / 2
    kny = kmax === nothing ? kny_grid : Float64(kmax) / kf
    0 < kny <= kny_grid ||
        error("kmax must satisfy 0 < kmax <= grid Nyquist $(kny_grid*kf), got $kmax")
    invN3 = 1.0 / Float64(n)^3
    V = L^3

    Psum = zeros(Float64, nb, nμ)
    psumd = Metal.zeros(Float32, nb, nμ)
    ksumd = Metal.zeros(Float32, nb)
    cntd = Metal.zeros(UInt32, nb, nμ)
    scale = Float32(V * invN3 * invN3)
    ndr = prod(PoissonKernels._rfft_dims(N))

    for (ifield, f) in pairs(fields)
        chat = _mps_forward_rfft!(f)
        _pkmu_bin_rfft_k!(Metal.MetalBackend())(psumd, ksumd, cntd, chat,
            Int32(n), Int32(nb), Int32(nμ), Int32(axis),
            Float32(kf), Float32(kny), scale, ifield == 1 ? Int32(1) : Int32(0);
            ndrange = ndr)
        Metal.synchronize()
    end

    Psum .= Float64.(Array(psumd))
    ksum = Float64.(Array(ksumd))
    cnt = Int.(Array(cntd))
    tot = vec(sum(cnt, dims=2))
    kc = [tot[b] > 0 ? ksum[b]/tot[b] : NaN for b in 1:nb]
    P = [cnt[b,m] > 0 ? Psum[b,m]/cnt[b,m] : NaN for b in 1:nb, m in 1:nμ]
    return (k = kc, P = P, Nmodes = cnt)
end

function PoissonKernels.power_spectrum_aniso_gpu(field::Metal.MtlArray{Float32,3};
                                                 boxsize=1.0, nmu::Integer=4,
                                                 nbins::Integer=0, axis::Integer=1,
                                                 kmax::Union{Nothing,Real}=nothing)
    _power_spectrum_aniso_mps((field,); boxsize=boxsize, nmu=nmu, nbins=nbins,
                              axis=axis, kmax=kmax)
end

function PoissonKernels.power_spectrum_aniso_gpu(fields::Tuple{Vararg{Metal.MtlArray{Float32,3}}};
                                                 boxsize=1.0, nmu::Integer=4,
                                                 nbins::Integer=0, axis::Integer=1,
                                                 kmax::Union{Nothing,Real}=nothing)
    _power_spectrum_aniso_mps(fields; boxsize=boxsize, nmu=nmu, nbins=nbins,
                              axis=axis, kmax=kmax)
end

end # module
