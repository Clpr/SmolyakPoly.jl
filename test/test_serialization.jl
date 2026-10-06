@testset "JSON serialization" begin
    mktempdir() do directory
        domain = BoxDomain(Float32[-1, 0], Float32[1, 2])
        grid = SmolyakGrid([2, 2], 3, domain)
        basis = ChebyshevBasisSpec([1, 2], 2)
        result = fit(x -> x[1] + x[2]^2, FitPlan(grid, basis); verbose = false)

        domain_path = joinpath(directory, "domain.json")
        grid_path = joinpath(directory, "grid.json")
        basis_path = joinpath(directory, "basis.json")
        result_path = joinpath(directory, "result.json")
        @test save(domain, domain_path) == domain_path
        save(grid, grid_path)
        save(basis, basis_path)
        save(result, result_path)

        loaded_domain = BoxDomain(domain_path)
        loaded_grid = SmolyakGrid(grid_path)
        loaded_basis = ChebyshevBasisSpec(basis_path)
        loaded_result = SmolyakApproximation(result_path)
        @test loaded_domain.lb == domain.lb
        @test loaded_grid.indices == grid.indices
        @test loaded_grid.nodes == grid.nodes
        @test loaded_basis.indices == basis.indices
        @test coefficients(loaded_result) == coefficients(result)
        @test loaded_result([0.25, 1.25]) == result([0.25, 1.25])
        @test occursin("\"format_version\": 1", read(result_path, String))

        bad_path = joinpath(directory, "bad.json")
        write(bad_path, "{\"format_version\":2,\"kind\":\"BoxDomain\"}")
        @test_throws ArgumentError BoxDomain(bad_path)
        @test_throws ArgumentError save(prepare(result),
                                        joinpath(directory, "runtime.json"))
    end
end

