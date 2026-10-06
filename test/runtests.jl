using Test
using LinearAlgebra
using SmolyakPoly

@testset "SmolyakPoly CPU" begin
    include("test_domain.jl")
    include("test_grid.jl")
    include("test_basis.jl")
    include("test_fitplan.jl")
    include("test_fit.jl")
    include("test_evaluation_cpu.jl")
    include("test_serialization.jl")
end
