using Random

@testset "BoxDomain" begin
    domain = BoxDomain([0, -2.0f0], [2.0, 4])
    @test domain isa BoxDomain{2,Float64}
    @test dimension(domain) == 2
    @test [1.0, 0.0] in domain
    @test !([3.0, 0.0] in domain)
    @test !([1.0] in domain)
    @test project(domain, [-1.0, 8.0]) == [0.0, 4.0]
    x = [0.5, 3.0]
    @test from_canonical(domain, to_canonical(domain, x)) ≈ x
    @test to_canonical(domain, domain.lb) == [-1.0, -1.0]
    @test from_canonical(domain, [1.0, 1.0]) == collect(domain.ub)

    one_d = BoxDomain([-1.0], [1.0])
    @test dimension(one_d) == 1
    @test rand(MersenneTwister(1), one_d) isa Vector{Float64}
    samples = rand(MersenneTwister(2), domain, 20)
    @test size(samples) == (20, 2)
    @test all(i -> @view(samples[i, :]) in domain, axes(samples, 1))
    @test size(rand(domain, 3)) == (3, 2)
    quasi = sobol(domain, 16)
    @test size(quasi) == (16, 2)
    @test all(i -> @view(quasi[i, :]) in domain, axes(quasi, 1))

    @test_throws DimensionMismatch BoxDomain([0.0], [1.0, 2.0])
    @test_throws ArgumentError BoxDomain(Float64[], Float64[])
    @test_throws ArgumentError BoxDomain([0.0], [0.0])
    @test_throws ArgumentError BoxDomain([NaN], [1.0])
    @test_throws DimensionMismatch to_canonical(domain, [1.0])
    @test occursin("BoxDomain", sprint(show, domain))
end
