@testset "SmolyakGrid" begin
    domain = BoxDomain([0.0], [1.0])
    grid = SmolyakGrid([4], 4, domain)
    @test grid isa SmolyakGrid{1}
    @test dimension(grid) == 1
    @test length(grid) == 5
    @test size(grid) == (5, 1)
    @test vec(grid.nodes[1:3, :]) == [0.5, 0.0, 1.0]
    @test length(unique(vec(grid.nodes))) == length(grid)
    @test all(i -> @view(grid.nodes[i, :]) in domain, 1:length(grid))
    @test grid.indices == [(0,), (1,), (2,), (3,), (4,)]

    domain2 = BoxDomain([-1.0, 2.0], [1.0, 4.0])
    anisotropic = SmolyakGrid([1, 3], 2, domain2)
    @test all(index -> index[1] <= 1 && index[2] <= 3 && sum(index) <= 2,
              anisotropic.indices)
    @test length(unique(Tuple.(eachrow(anisotropic.nodes)))) == length(anisotropic)
    @test all(i -> @view(anisotropic.nodes[i, :]) in domain2,
              1:length(anisotropic))
    nested_small = SmolyakGrid([2], 2, domain)
    @test grid.nodes[1:length(nested_small), :] == nested_small.nodes
    @test_throws ArgumentError SmolyakGrid([2], 2, domain; rule = :other)
    @test_throws ArgumentError SmolyakGrid([-1], 2, domain)
    @test occursin("SmolyakGrid", sprint(show, anisotropic))
end

