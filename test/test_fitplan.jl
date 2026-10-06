@testset "FitPlan" begin
    domain = BoxDomain([-1.0], [1.0])
    square_grid = SmolyakGrid([3], 3, domain)
    square_basis = ChebyshevBasisSpec([3], 3)
    square = FitPlan(square_grid, square_basis)
    @test square.system_type == :square
    @test size(square.design) == (4, 4)
    @test square.factorization !== nothing

    over_grid = SmolyakGrid([5], 5, domain)
    small_basis = ChebyshevBasisSpec([2], 2)
    over = FitPlan(over_grid, small_basis)
    @test over.system_type == :overdetermined
    ridge = FitPlan(over_grid, small_basis; ridge_lambda = 1e-6)
    @test ridge.ridge_lambda == 1e-6
    @test size(ridge.design) == (6, 3)
    @test occursin("factorization=cached", sprint(show, ridge))

    under_basis = ChebyshevBasisSpec([6], 6)
    @test_throws ArgumentError FitPlan(square_grid, under_basis)
    wrong_dimension = ChebyshevBasisSpec([1, 1], 1)
    @test_throws DimensionMismatch FitPlan(square_grid, wrong_dimension)
    @test_throws ArgumentError FitPlan(over_grid, small_basis; ridge_lambda = -1.0)
    @test_throws ArgumentError FitPlan(over_grid, small_basis; solver = :normal)
end

