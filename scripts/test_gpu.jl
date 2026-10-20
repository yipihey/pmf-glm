# Use the active optional environment; this script never activates or installs packages.
function requested_backend(args)
    length(args) == 1 && startswith(only(args),"--backend=") ||
        error("usage: julia --project=<optional-env> scripts/test_gpu.jl --backend=metal|cuda")
    name = Symbol(split(only(args),'=';limit=2)[2])
    name in (:metal,:cuda) || error("--backend must be metal or cuda")
    name
end

backend_name = requested_backend(ARGS)
if backend_name === :metal
    import Metal
else
    import CUDA
end
include(joinpath(@__DIR__,"..","test","test_backends.jl"))
Base.invokelatest(PMFGLMBackendTests.run_backend_tests,backend_name)
