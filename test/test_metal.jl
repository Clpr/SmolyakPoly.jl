# Manual-only test: run on an Apple Silicon Mac with Metal.jl installed.
# This file is intentionally not included by test/runtests.jl, especially on
# Windows where Metal cannot execute.
using Test
import Metal
using SmolyakPoly

@testset "Metal sidecar" begin
    domain = BoxDomain([-1.0, -1.0], [1.0, 1.0])
    grid = SmolyakGrid([3, 3], 4, domain)
    basis = ChebyshevBasisSpec([2, 2], 3)
    result = fit(x -> x[1]^2 + x[2], FitPlan(grid, basis); verbose = false)
    points = Float32[0.2 -0.4; -0.5 0.75; 1 -1]
    evaluator = prepare(result; T = Float32, backend = :metal)
    output1 = evaluator(points)
    output2 = evaluator(points)
    @test output1 isa Metal.MtlArray
    @test Array(output1) ≈ result(Float32, points) rtol = 5f-5
    @test Array(output2) ≈ Array(output1)
    @test basis_matrix(evaluator, points) isa Metal.MtlArray
    @test_throws ArgumentError prepare(result; T = Float64, backend = :metal)
    @test :metal in available_backends()
end
