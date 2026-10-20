using PMFGLM
using FFTW
FFTW.set_num_threads(parse(Int,get(ENV,"PMF_FFTW_THREADS",string(min(4,Threads.nthreads())))))
include(joinpath(@__DIR__,"..","src","production","runner.jl"))
main()
