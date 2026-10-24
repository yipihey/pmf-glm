# The accepted CFL ceilings (MAX_STABLE_CFL in src/configuration.jl) come from the measured
# linear stability of the unsplit 3D MUSCL-Hancock scheme.  Re-measure the two edges for each
# solver on a short run: stable at the accepted ceiling, blown up well above the measured limit.
using Test
include(joinpath(@__DIR__, "..", "tools", "cfl_stability", "cfl_stability.jl"))

@testset "CFL ceiling is stable, 2x the ceiling is not ($solver)" for solver in (:glm, :ct)
    ceiling = PMFGLM.MAX_STABLE_CFL[String(solver)]
    ok = cfl_stability_run(solver, ceiling; nsteps=600)
    @test !ok.blown && ok.steps == 600
    @test ok.growth_max < 100            # physical wave plus noise; measured about 4 (GLM), 10 (CT)
    bad = cfl_stability_run(solver, 2ceiling; nsteps=600)
    @test bad.blown && bad.steps < 600   # measured: GLM 35 steps, CT 55 steps at cfl 0.5
end
