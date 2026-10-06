using Test
using SmolyakPoly

if Base.find_package("CUDA") === nothing
    @info "CUDA.jl is not installed; CUDA tests skipped"
else
    import CUDA
    if !CUDA.functional()
        @info "CUDA hardware/runtime is not functional; CUDA tests skipped"
    else
        @testset "CUDA sidecar" begin
            domain = BoxDomain([-1.0, -1.0], [1.0, 1.0])
            grid = SmolyakGrid([3, 3], 4, domain)
            basis = ChebyshevBasisSpec([2, 2], 3)
            result = fit(x -> x[1]^2 + x[2], FitPlan(grid, basis);
                         verbose = false)
            points = Float32[0.2 -0.4; -0.5 0.75; 1 -1]
            evaluator = prepare(result; T = Float32, backend = :cuda)
            output1 = evaluator(points)
            output2 = evaluator(points)
            @test output1 isa CUDA.CuArray
            @test Array(output1) ≈ result(Float32, points) rtol = 5f-5
            @test Array(output2) ≈ Array(output1)
            matrix = basis_matrix(evaluator, points)
            @test matrix isa CUDA.CuArray
            @test Array(matrix * evaluator.coeffs) ≈ Array(output1) rtol = 5f-5
            @test :cuda in available_backends()
        end
    end
end
