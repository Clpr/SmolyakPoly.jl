@testset "ChebyshevBasisSpec" begin
    basis = ChebyshevBasisSpec([2, 3], 2)
    @test basis isa ChebyshevBasisSpec{2}
    @test dimension(basis) == 2
    @test length(basis) == 6
    @test all(index -> index[1] <= 2 && index[2] <= 3 && sum(index) <= 2,
              basis.indices)
    @test basis.indices[1] == (0, 0)
    @test (2, 0) in basis.indices
    @test !((0, 3) in basis.indices)

    domain = BoxDomain([-1.0, -1.0], [1.0, 1.0])
    grid = SmolyakGrid([2, 1], 2, domain)
    independent = ChebyshevBasisSpec([1, 1], 1)
    @test grid.indices != independent.indices
    plan = FitPlan(grid, independent)
    result = fit(x -> x[1], plan; verbose = false)
    vector = basis_vector(result, [0.25, -0.5])
    for (k, alpha) in pairs(independent.indices)
        expected = (alpha[1] == 0 ? 1.0 : 0.25) *
                   (alpha[2] == 0 ? 1.0 : -0.5)
        @test vector[k] ≈ expected
    end
    @test occursin("ChebyshevBasisSpec", sprint(show, basis))
end

