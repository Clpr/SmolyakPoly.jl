@testset "CPU evaluation" begin
    domain = BoxDomain([-1.0, -1.0], [1.0, 1.0])
    grid = SmolyakGrid([4, 4], 5, domain)
    basis = ChebyshevBasisSpec([2, 2], 3)
    plan = FitPlan(grid, basis)
    target(x) = 0.5 + x[1]^2 + 2x[2]
    result = fit(target, plan; verbose = false)
    point = [0.2, -0.4]
    points = [0.2 -0.4; -0.5 0.75; 1.0 -1.0]
    @test result(point) ≈ target(point) atol = 1e-12
    @test result(points) ≈ [target(@view points[i, :]) for i in 1:3] atol = 1e-12
    @test result(Float32, point) isa Float32
    @test result(Float32, points) isa Vector{Float32}
    @test length(result(points)) == 3
    @test_throws DomainError result([1.1, 0.0])
    @test result(project(domain, [1.1, 0.0])) ≈ target([1.0, 0.0])

    vector64 = basis_vector(result, point)
    matrix64 = basis_matrix(result, points)
    @test dot(vector64, coefficients(result)) ≈ result(point) atol = 1e-13
    @test matrix64 * coefficients(result) ≈ result(points) atol = 1e-13
    vector32 = basis_vector(Float32, result, point)
    matrix32 = basis_matrix(Float32, result, points)
    @test dot(vector32, Float32.(coefficients(result))) ≈
          result(Float32, point) rtol = 2f-6
    @test matrix32 * Float32.(coefficients(result)) ≈
          result(Float32, points) rtol = 2f-6

    evaluator = prepare(result)
    evaluator32 = prepare(result; T = Float32)
    @test evaluator(point) ≈ result(point)
    @test evaluator(points) ≈ result(points)
    @test evaluator32(point) ≈ result(Float32, point) rtol = 2f-6
    @test basis_matrix(evaluator, points) * evaluator.coeffs ≈ evaluator(points)
    @test prepare(result; backend = :auto)(point) ≈ result(point)
    @test available_backends() == (:cpu,)
    @test_throws ArgumentError prepare(result; backend = :missing)

    @test @inferred(dimension(result)) == 2
    @test @inferred(result(point)) ≈ target(point)
    @test @inferred(evaluator(point)) ≈ target(point)

    one_domain = BoxDomain([0.0], [1.0])
    one_plan = FitPlan(SmolyakGrid([2], 2, one_domain),
                       ChebyshevBasisSpec([2], 2))
    one_result = fit(x -> x[1]^2, one_plan; verbose = false)
    @test one_result([0.3]) ≈ 0.09 atol = 1e-13
end

