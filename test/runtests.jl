using Test
using PMFGLM
using KernelAbstractions
using FFTW
using Statistics

@testset "configuration, units, and reproducible inputs" begin
    include("test_configuration.jl")
end

@testset "CFL stability limits of the unsplit scheme" begin
    include("test_cfl_stability.jl")
end

@testset "terminal pressure-implicit determinism and CPU exponential solve" begin
    include("test_terminal_pressure_determinism.jl")
end

@testset "constrained-transport solver at the runner level" begin
    include("test_ct_runner.jl")
end

@testset "periodic gravity and particles" begin
    include("test_gravity.jl")
end

include("streaming_checks.jl")
@testset "streaming and planar three-component evolution" begin
    for T in (Float32,Float64)
        streaming_checks(:cpu,T)
    end
end

module CubeCPUTestsHost
include("test_cube_cpu.jl")
end

module InheritedMHDTests
include("inherited/runtests.jl")
end

module RecombinationTests
include("inherited/test_recombination_mixing.jl")
end

module CheckpointTests
include("inherited/test_pmf_checkpoint_helpers.jl")
end

module RelativeVelocityTests
include("test_relative_velocities.jl")
end
