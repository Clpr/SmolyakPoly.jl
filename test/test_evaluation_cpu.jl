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


@testset "Prepared gradients and Hessians" begin
    domain = BoxDomain([-2.0, 1.0], [4.0, 5.0])
    grid = SmolyakGrid([4, 4], 4, domain)
    basis = ChebyshevBasisSpec([2, 2], 2)
    plan = FitPlan(grid, basis)
    target(x) = 1 + 2x[1] - 3x[2] + 0.5x[1]^2 +
                4x[1] * x[2] + 2x[2]^2
    result = fit(target, plan; verbose = false)

    expected_gradient(x) = [2 + x[1] + 4x[2], -3 + 4x[1] + 4x[2]]
    expected_hessian = [1.0 4.0; 4.0 4.0]
    point = [0.25, 2.5]
    points = [0.25 2.5; -1.0 4.0; 3.0 1.5]

    gradient_evaluator = prepare_gradient(result)
    hessian_evaluator = prepare_hessian(result)
    @test gradient_evaluator(point) ≈ expected_gradient(point) atol = 2e-12
    @test gradient_evaluator(points) ≈
          reduce(vcat, permutedims.(expected_gradient.([@view points[i, :] for i in 1:3])))
          atol = 2e-12
    @test hessian_evaluator(point) ≈ expected_hessian atol = 2e-12
    batch_hessians = hessian_evaluator(points)
    @test batch_hessians isa Vector{Matrix{Float64}}
    @test length(batch_hessians) == 3
    @test all(hessian -> hessian ≈ expected_hessian, batch_hessians)
    @test all(issymmetric, batch_hessians)

    gradient32 = prepare_gradient(result; T = Float32)
    hessian32 = prepare_hessian(result; T = Float32)
    @test gradient32(point) isa Vector{Float32}
    @test gradient32(points) isa Matrix{Float32}
    @test hessian32(point) isa Matrix{Float32}
    @test hessian32(points) isa Vector{Matrix{Float32}}
    @test gradient32(point) ≈ Float32.(expected_gradient(point)) rtol = 2f-5
    @test hessian32(point) ≈ Float32.(expected_hessian) rtol = 2f-5

    @test prepare_gradient(result; backend = :auto)(point) ≈
          expected_gradient(point) atol = 2e-12
    @test prepare_hessian(result; backend = :auto)(point) ≈
          expected_hessian atol = 2e-12
    @test_throws DomainError gradient_evaluator([5.0, 2.0])
    @test_throws DomainError hessian_evaluator([5.0, 2.0])
    @test_throws ArgumentError prepare_gradient(result; backend = :missing)
    @test_throws ArgumentError prepare_hessian(result; backend = :missing)

    @test @inferred(gradient_evaluator(point)) ≈ expected_gradient(point)
    @test @inferred(hessian_evaluator(point)) ≈ expected_hessian

    cubic_basis = ChebyshevBasisSpec([3, 3], 3)
    cubic_plan = FitPlan(SmolyakGrid([5, 5], 5, domain), cubic_basis)
    cubic(x) = x[1]^3 + x[1] * x[2]^2 + x[2]^3
    cubic_result = fit(cubic, cubic_plan; verbose = false)
    cubic_gradient = prepare_gradient(cubic_result)
    cubic_hessian = prepare_hessian(cubic_result)
    cubic_gradient_expected(x) = [3x[1]^2 + x[2]^2,
                                  2x[1] * x[2] + 3x[2]^2]
    cubic_hessian_expected(x) = [6x[1] 2x[2];
                                 2x[2] 2x[1] + 6x[2]]
    @test cubic_gradient(point) ≈ cubic_gradient_expected(point) atol = 2e-11
    @test cubic_hessian(point) ≈ cubic_hessian_expected(point) atol = 2e-11
end
