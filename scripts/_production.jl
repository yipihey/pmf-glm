using PMFGLM
using FFTW
fft_threads = parse(Int,get(ENV,"PMF_FFTW_THREADS",string(min(4,Threads.nthreads()))))
fft_threads > 0 || error("PMF_FFTW_THREADS must be positive")
# The IC generator also sets FFTW's global plan thread count. Keep that choice
# consistent with the public launcher setting rather than resetting it to four.
ENV["MHD_PMF_FFT_THREADS"] = string(fft_threads)
FFTW.set_num_threads(fft_threads)
include(joinpath(@__DIR__,"..","src","production","runner.jl"))
main()
