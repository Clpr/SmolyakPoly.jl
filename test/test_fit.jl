@testset "Fitting and refitting" begin
    domain = BoxDomain([-1.0, -2.0], [1.0, 2.0])
    grid = SmolyakGrid([3, 3], 4, domain)
    basis = ChebyshevBasisSpec([2, 1], 3)
    plan = FitPlan(grid, basis)
    target(x) = 1.25 + 0.5x[1] - 0.75x[2] + 2x[1]^2
    values = evaluate_nodes(target, grid; verbose = false)
    threaded = evaluate_nodes(target, grid;
                              multi_threading = true, verbose = false)
    @test values == threaded
    result = fit(plan, values; verbose = false)
    convenient = fit(target, plan; verbose = false)
    @test coefficients(result) ≈ coefficients(convenient) atol = 1e-13
    @test result.node_rmse < 1e-12
    @test result.max_node_error < 1e-12

    new_target(x) = -2.0 + x[1] + 0.25x[2]
    new_values = evaluate_nodes(new_target, grid; verbose = false)
    fresh = fit(plan, new_values; verbose = false)
    original_grid = result.grid
    original_basis = result.basis
    @test fit!(result, plan, new_values; verbose = false) === result
    @test result.grid === original_grid
    @test result.basis === original_basis
    @test coefficients(result) ≈ coefficients(fresh) atol = 1e-13
    @test result.node_rmse ≈ fresh.node_rmse

    incompatible_grid = SmolyakGrid([4, 3], 4, domain)
    incompatible_plan = FitPlan(incompatible_grid, basis)
    @test_throws ArgumentError fit!(result, incompatible_plan,
                                    zeros(length(incompatible_grid)); verbose = false)
    @test_throws DimensionMismatch fit(plan, values[1:end-1]; verbose = false)
end

