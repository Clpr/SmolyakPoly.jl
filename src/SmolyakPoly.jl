module SmolyakPoly

using LinearAlgebra
using Random
using JSON3
using Sobol

export BoxDomain, SmolyakGrid, ChebyshevBasisSpec, FitPlan,
       SmolyakApproximation, dimension, project, to_canonical,
       from_canonical, sobol, evaluate_nodes, fit, fit!, coefficients,
       basis_vector, basis_matrix, prepare, available_backends, save

const FORMAT_VERSION = 1
const MultiIndex{D} = NTuple{D,Int}

# ------------------------------------------------------------------------------
# Validation and index construction
# ------------------------------------------------------------------------------

"""Return `x` as a length-`D` tuple of values converted to `T`."""
function _point_tuple(::Type{T}, x, D::Int, name::AbstractString) where {T}
    length(x) == D || throw(DimensionMismatch(
        "$name must have length $D; received length $(length(x))"))
    return ntuple(d -> convert(T, x[d]), D)
end

"""
    _lower_indices(caps, budget)

Construct the deterministic lower multi-index set
`{alpha : 0 <= alpha[d] <= caps[d], sum(alpha) <= budget}`. Indices are
ordered by total degree and then lexicographically.
"""
function _lower_indices(caps::NTuple{D,Int}, budget::Int) where {D}
    budget >= 0 || throw(ArgumentError("the total budget must be nonnegative"))
    all(>=(0), caps) || throw(ArgumentError(
        "all coordinate caps must be nonnegative; received $caps"))
    result = MultiIndex{D}[]
    current = zeros(Int, D)
    function visit!(coordinate::Int, remaining::Int)
        if coordinate > D
            push!(result, ntuple(d -> current[d], Val(D)))
            return
        end
        for value in 0:min(caps[coordinate], remaining)
            current[coordinate] = value
            visit!(coordinate + 1, remaining - value)
        end
    end
    visit!(1, budget)
    sort!(result; by = alpha -> (sum(alpha), alpha))
    return result
end

"""Return a matrix whose rows contain the supplied multi-indices."""
function _index_matrix(indices::Vector{MultiIndex{D}}) where {D}
    matrix = Matrix{Int}(undef, length(indices), D)
    for k in eachindex(indices), d in 1:D
        matrix[k, d] = indices[k][d]
    end
    return matrix
end

# ------------------------------------------------------------------------------
# Box domain
# ------------------------------------------------------------------------------

"""
    BoxDomain(lb, ub)

Describe a finite Cartesian product of closed intervals. `lb` and `ub` may be
any indexable, finite real vectors of equal positive length. Their values are
promoted to a floating-point type. The returned `BoxDomain{D,T}` caches the
affine map between physical coordinates and `[-1,1]^D`.
"""
struct BoxDomain{D,T<:AbstractFloat}
    lb::NTuple{D,T}
    ub::NTuple{D,T}
    center::NTuple{D,T}
    halfwidth::NTuple{D,T}
    invhalfwidth::NTuple{D,T}
end

function BoxDomain(lb, ub)
    length(lb) == length(ub) || throw(DimensionMismatch(
        "lower and upper bounds must have equal lengths; received " *
        "$(length(lb)) and $(length(ub))"))
    D = length(lb)
    D >= 1 || throw(ArgumentError("a box domain must have at least one dimension"))
    T = promote_type(map(x -> typeof(float(x)), lb)...,
                     map(x -> typeof(float(x)), ub)...)
    T <: AbstractFloat || throw(ArgumentError("bounds must promote to a floating type"))
    lower = _point_tuple(T, lb, D, "lower bounds")
    upper = _point_tuple(T, ub, D, "upper bounds")
    for d in 1:D
        isfinite(lower[d]) || throw(ArgumentError("lower bound $d is not finite"))
        isfinite(upper[d]) || throw(ArgumentError("upper bound $d is not finite"))
        lower[d] < upper[d] || throw(ArgumentError(
            "lower bound $d must be less than its upper bound; received " *
            "$(lower[d]) and $(upper[d])"))
    end
    two = T(2)
    center = ntuple(d -> (lower[d] + upper[d]) / two, D)
    halfwidth = ntuple(d -> (upper[d] - lower[d]) / two, D)
    inverse = ntuple(d -> inv(halfwidth[d]), D)
    return BoxDomain{D,T}(lower, upper, center, halfwidth, inverse)
end

"""Load a `BoxDomain` from a versioned JSON file written by [`save`](@ref)."""
BoxDomain(path::AbstractString) = _load_domain(_read_json(path))

"""Return the mathematical dimension encoded by an object."""
dimension(::BoxDomain{D}) where {D} = D

"""
    x in domain

Return whether vector-like point `x` has the correct length and lies in every
closed coordinate interval. Non-finite and non-convertible values return false.
"""
function Base.in(x, domain::BoxDomain{D,T}) where {D,T}
    length(x) == D || return false
    try
        return all(d -> domain.lb[d] <= x[d] <= domain.ub[d], 1:D)
    catch
        return false
    end
end

"""Project vector-like point `x` coordinatewise onto `domain`."""
function project(domain::BoxDomain{D,T}, x) where {D,T}
    point = _point_tuple(T, x, D, "point")
    return T[clamp(point[d], domain.lb[d], domain.ub[d]) for d in 1:D]
end

"""Map a physical vector-like point to canonical coordinates in `[-1,1]^D`."""
function to_canonical(domain::BoxDomain{D,T}, x) where {D,T}
    point = _point_tuple(T, x, D, "point")
    return T[(point[d] - domain.center[d]) * domain.invhalfwidth[d] for d in 1:D]
end

"""Map a length-`D` canonical vector-like point to physical coordinates."""
function from_canonical(domain::BoxDomain{D,T}, xi) where {D,T}
    point = _point_tuple(T, xi, D, "canonical point")
    return T[domain.center[d] + domain.halfwidth[d] * point[d] for d in 1:D]
end

"""Draw one uniformly distributed point from `domain` using `rng`."""
function Random.rand(rng::AbstractRNG, domain::BoxDomain{D,T}) where {D,T}
    return T[domain.lb[d] + rand(rng, T) *
             (domain.ub[d] - domain.lb[d]) for d in 1:D]
end

"""Draw one uniformly distributed point using Julia's default RNG."""
Random.rand(domain::BoxDomain) = rand(Random.default_rng(), domain)

"""Draw `n` uniform points as an `n`-by-`D` dense matrix."""
function Random.rand(rng::AbstractRNG, domain::BoxDomain{D,T}, n::Integer) where {D,T}
    n >= 0 || throw(ArgumentError("sample count must be nonnegative; received $n"))
    points = Matrix{T}(undef, n, D)
    for d in 1:D, i in 1:n
        points[i, d] = domain.lb[d] + rand(rng, T) *
                       (domain.ub[d] - domain.lb[d])
    end
    return points
end

"""Draw `n` uniform point rows using Julia's default RNG."""
Random.rand(domain::BoxDomain, n::Integer) = rand(Random.default_rng(), domain, n)

"""
    sobol(domain, n)

Return the first `n` points of a deterministic Sobol low-discrepancy sequence,
scaled to `domain`, as an `n`-by-`D` matrix.
"""
function sobol(domain::BoxDomain{D,T}, n::Integer) where {D,T}
    n >= 0 || throw(ArgumentError("sample count must be nonnegative; received $n"))
    sequence = SobolSeq(D)
    points = Matrix{T}(undef, n, D)
    workspace = Vector{Float64}(undef, D)
    for i in 1:n
        Sobol.next!(sequence, workspace)
        for d in 1:D
            points[i, d] = domain.lb[d] + T(workspace[d]) *
                           (domain.ub[d] - domain.lb[d])
        end
    end
    return points
end

function Base.show(io::IO, domain::BoxDomain{D,T}) where {D,T}
    print(io, "BoxDomain{$D,$T}(")
    for d in 1:D
        d > 1 && print(io, ", ")
        print(io, "[", domain.lb[d], ", ", domain.ub[d], "]")
    end
    print(io, ")")
end

# ------------------------------------------------------------------------------
# Nested real Leja rule and sparse grid
# ------------------------------------------------------------------------------

"""Evaluate the logarithm of the Leja product at `x`."""
function _leja_log_product(x::T, nodes::Vector{T}) where {T<:AbstractFloat}
    total = zero(T)
    for node in nodes
        distance = abs(x - node)
        iszero(distance) && return T(-Inf)
        total += log(distance)
    end
    return total
end

"""
    _leja_nodes(T, n)

Generate `n` nested real Leja nodes on `[-1,1]`, beginning at zero. At every
step this routine maximizes the product of distances from earlier nodes. The
log-product is strictly concave between adjacent nodes, so its unique stationary
point is found by bisection. Equal maxima are resolved toward the smaller
coordinate. This defines deterministic ordering without a discretization grid.
"""
function _leja_nodes(::Type{T}, n::Int) where {T<:AbstractFloat}
    n >= 1 || throw(ArgumentError("at least one Leja node is required"))
    nodes = T[zero(T)]
    tolerance = eps(T) * T(64)
    while length(nodes) < n
        ordered = sort(nodes)
        candidates = T[]
        !any(==(-one(T)), nodes) && push!(candidates, -one(T))
        !any(==(one(T)), nodes) && push!(candidates, one(T))
        for j in 1:(length(ordered) - 1)
            left = ordered[j]
            right = ordered[j + 1]
            lo = nextfloat(left)
            hi = prevfloat(right)
            for _ in 1:80
                mid = (lo + hi) / T(2)
                derivative = zero(T)
                for node in nodes
                    derivative += inv(mid - node)
                end
                if derivative > zero(T)
                    lo = mid
                else
                    hi = mid
                end
            end
            push!(candidates, (lo + hi) / T(2))
        end
        best = candidates[1]
        best_value = _leja_log_product(best, nodes)
        for candidate in @view candidates[2:end]
            value = _leja_log_product(candidate, nodes)
            scale = max(one(T), abs(best_value), abs(value))
            if value > best_value + tolerance * scale ||
               (abs(value - best_value) <= tolerance * scale && candidate < best)
                best = candidate
                best_value = value
            end
        end
        push!(nodes, best)
    end
    return nodes
end

"""
    SmolyakGrid(max_levels, level_budget, domain; rule=:leja)

Materialize a nested sparse Leja grid. Levels are zero-based: multi-index
`alpha` selects the `(alpha[d]+1)`-st one-dimensional Leja node in coordinate
`d`. The admissible set is
`sum(alpha) <= level_budget` and `alpha[d] <= max_levels[d]`.
Rows of `nodes` are physical points and follow total-degree/lexicographic order.
"""
struct SmolyakGrid{D,T<:AbstractFloat,Dom<:BoxDomain{D,T}}
    domain::Dom
    max_levels::NTuple{D,Int}
    level_budget::Int
    indices::Vector{MultiIndex{D}}
    nodes::Matrix{T}
    rule::Symbol
end

function SmolyakGrid(max_levels, level_budget::Integer,
                     domain::BoxDomain{D,T}; rule::Symbol = :leja) where {D,T}
    rule === :leja || throw(ArgumentError(
        "unsupported node rule $rule; only :leja is available"))
    levels = _point_tuple(Int, max_levels, D, "max_levels")
    indices = _lower_indices(levels, Int(level_budget))
    maximum_level = maximum(levels)
    one_dimensional = _leja_nodes(T, maximum_level + 1)
    nodes = Matrix{T}(undef, length(indices), D)
    for i in eachindex(indices), d in 1:D
        xi = one_dimensional[indices[i][d] + 1]
        nodes[i, d] = domain.center[d] + domain.halfwidth[d] * xi
    end
    return SmolyakGrid{D,T,typeof(domain)}(
        domain, levels, Int(level_budget), indices, nodes, rule)
end

"""Load a `SmolyakGrid` from a versioned JSON file written by [`save`](@ref)."""
SmolyakGrid(path::AbstractString) = _load_grid(_read_json(path))

"""Return the state-space dimension of `grid`."""
dimension(::SmolyakGrid{D}) where {D} = D
Base.length(grid::SmolyakGrid) = size(grid.nodes, 1)
Base.size(grid::SmolyakGrid) = size(grid.nodes)

function Base.show(io::IO, grid::SmolyakGrid{D,T}) where {D,T}
    print(io, "SmolyakGrid{$D,$T}(rule=", grid.rule,
          ", nodes=", length(grid), ", level_budget=", grid.level_budget,
          ", max_levels=", grid.max_levels, ")")
end

# ------------------------------------------------------------------------------
# Chebyshev basis
# ------------------------------------------------------------------------------

"""
    ChebyshevBasisSpec(max_orders, order_budget)

Describe a finite tensor-product Chebyshev lower set. Polynomial multi-indices
obey `sum(alpha) <= order_budget` and `alpha[d] <= max_orders[d]`, independently
of any grid. Ordering is total degree followed by lexicographic order.
"""
struct ChebyshevBasisSpec{D}
    max_orders::NTuple{D,Int}
    order_budget::Int
    indices::Vector{MultiIndex{D}}
end

function ChebyshevBasisSpec(max_orders, order_budget::Integer)
    D = length(max_orders)
    D >= 1 || throw(ArgumentError("a basis must have at least one dimension"))
    orders = _point_tuple(Int, max_orders, D, "max_orders")
    indices = _lower_indices(orders, Int(order_budget))
    return ChebyshevBasisSpec{D}(orders, Int(order_budget), indices)
end

"""Load a basis specification from versioned JSON written by [`save`](@ref)."""
ChebyshevBasisSpec(path::AbstractString) = _load_basis(_read_json(path))

"""Return the state-space dimension of `basis`."""
dimension(::ChebyshevBasisSpec{D}) where {D} = D
Base.length(basis::ChebyshevBasisSpec) = length(basis.indices)

function Base.show(io::IO, basis::ChebyshevBasisSpec{D}) where {D}
    print(io, "ChebyshevBasisSpec{$D}(terms=", length(basis),
          ", order_budget=", basis.order_budget,
          ", max_orders=", basis.max_orders, ")")
end

"""Evaluate Chebyshev polynomial `T_order(x)` by the stable three-term recurrence."""
@inline function _chebyshev_value(order::Int, x::T) where {T}
    order == 0 && return one(T)
    order == 1 && return x
    previous = one(T)
    current = x
    for _ in 2:order
        following = muladd(T(2) * x, current, -previous)
        previous = current
        current = following
    end
    return current
end

"""Check the shape and membership of one point, throwing a domain-specific error."""
function _validate_point(domain::BoxDomain{D}, x) where {D}
    length(x) == D || throw(DimensionMismatch(
        "point must have length $D; received length $(length(x))"))
    x in domain || throw(DomainError(x,
        "point lies outside the approximation domain; use project(domain, x) " *
        "explicitly if projection is intended"))
    return nothing
end

"""Check an `N`-by-`D` point matrix and verify every row is in `domain`."""
function _validate_points(domain::BoxDomain{D}, X::AbstractMatrix) where {D}
    size(X, 2) == D || throw(DimensionMismatch(
        "batch input must have $D columns; received size $(size(X))"))
    for i in axes(X, 1), d in 1:D
        domain.lb[d] <= X[i, d] <= domain.ub[d] || throw(DomainError(
            X[i, d], "point row $i lies outside coordinate $d of the domain"))
    end
    return nothing
end

"""Evaluate one tensor-product term using physical coordinates."""
@inline function _term_value(::Type{T}, domain::BoxDomain{D}, alpha,
                             x) where {T,D}
    value = one(T)
    for d in 1:D
        xi = (T(x[d]) - T(domain.center[d])) * T(domain.invhalfwidth[d])
        value *= _chebyshev_value(alpha[d], xi)
    end
    return value
end

# ------------------------------------------------------------------------------
# Fit plan
# ------------------------------------------------------------------------------

"""
    FitPlan(grid, basis; ridge_lambda=0.0, solver=:qr)

Cache the canonical design matrix and a pivoted QR factorization for repeated
fits. Ridge regression uses QR on the augmented system
`[Phi; sqrt(lambda)I]`. Underdetermined systems are rejected. Unregularized
rank-deficient systems are also rejected with an informative error.
"""
struct FitPlan{D,T,G,B,F}
    grid::G
    basis::B
    canonical_nodes::Matrix{T}
    design::Matrix{T}
    factorization::F
    system_type::Symbol
    solver::Symbol
    ridge_lambda::T
end

function FitPlan(grid::SmolyakGrid{D}, basis::ChebyshevBasisSpec{D2};
                 ridge_lambda::Real = 0.0, solver::Symbol = :qr) where {D,D2}
    D == D2 || throw(DimensionMismatch(
        "grid dimension $D does not match basis dimension $D2"))
    solver === :qr || throw(ArgumentError(
        "unsupported solver $solver; only :qr is available"))
    isfinite(ridge_lambda) && ridge_lambda >= 0 || throw(ArgumentError(
        "ridge_lambda must be finite and nonnegative; received $ridge_lambda"))
    node_count = length(grid)
    term_count = length(basis)
    node_count >= term_count || throw(ArgumentError(
        "underdetermined fitting system: $node_count nodes for $term_count " *
        "basis terms; reduce the basis or enlarge the grid"))
    T = Float64
    canonical = Matrix{T}(undef, node_count, D)
    for i in 1:node_count, d in 1:D
        canonical[i, d] = (T(grid.nodes[i, d]) - T(grid.domain.center[d])) *
                          T(grid.domain.invhalfwidth[d])
    end
    design = _basis_matrix_canonical(T, basis, canonical)
    lambda = T(ridge_lambda)
    if iszero(lambda)
        matrix_rank = rank(design)
        matrix_rank == term_count || throw(ArgumentError(
            "fitting design is rank deficient: rank $matrix_rank for " *
            "$term_count basis terms; change the grid or use positive ridge_lambda"))
        factorization = qr(design, ColumnNorm())
    else
        augmented = [design; sqrt(lambda) * Matrix{T}(I, term_count, term_count)]
        factorization = qr(augmented, ColumnNorm())
    end
    system_type = node_count == term_count ? :square : :overdetermined
    return FitPlan{D,T,typeof(grid),typeof(basis),typeof(factorization)}(
        grid, basis, canonical, design, factorization, system_type, solver, lambda)
end

"""Build a basis matrix from canonical point rows without domain validation."""
function _basis_matrix_canonical(::Type{T}, basis::ChebyshevBasisSpec{D},
                                 canonical::AbstractMatrix) where {T,D}
    matrix = Matrix{T}(undef, size(canonical, 1), length(basis))
    for i in axes(canonical, 1), k in eachindex(basis.indices)
        value = one(T)
        alpha = basis.indices[k]
        for d in 1:D
            value *= _chebyshev_value(alpha[d], T(canonical[i, d]))
        end
        matrix[i, k] = value
    end
    return matrix
end

function Base.show(io::IO, plan::FitPlan)
    print(io, "FitPlan(nodes=", length(plan.grid), ", terms=", length(plan.basis),
          ", system=", plan.system_type, ", solver=", plan.solver,
          ", ridge_lambda=", plan.ridge_lambda, ", factorization=cached)")
end

# ------------------------------------------------------------------------------
# Node evaluation and fitted approximation
# ------------------------------------------------------------------------------

"""
    evaluate_nodes(fun, grid; multi_threading=false, verbose=true)

Call arbitrary callable `fun` once for each grid row. Each call receives a
vector-like row view. The result is a dense floating vector. With threading,
the first value is evaluated once to establish the output type and remaining
rows are distributed using `Threads.@threads`.
"""
function evaluate_nodes(fun, grid::SmolyakGrid;
                        multi_threading::Bool = false, verbose::Bool = true)
    count = length(grid)
    first_value = float(fun(@view grid.nodes[1, :]))
    T = typeof(first_value)
    T <: AbstractFloat || throw(ArgumentError(
        "node function values must convert to a floating scalar; received $T"))
    values = Vector{T}(undef, count)
    values[1] = first_value
    if multi_threading && count > 1
        Threads.@threads for i in 2:count
            values[i] = fun(@view grid.nodes[i, :])
        end
    else
        for i in 2:count
            values[i] = fun(@view grid.nodes[i, :])
        end
    end
    verbose && println("Evaluated $count sparse-grid nodes.")
    return values
end

"""
    SmolyakApproximation

A mutable, scalar-valued fitted polynomial. It stores persistent mathematical
state and diagnostics only; prepared CPU/GPU runtime buffers are separate.
"""
mutable struct SmolyakApproximation{D,T<:AbstractFloat,Dom,G,B}
    domain::Dom
    grid::G
    basis::B
    coeffs::Vector{T}
    system_type::Symbol
    solver::Symbol
    ridge_lambda::T
    node_rmse::T
    max_node_error::T
end

"""Load a fitted approximation from versioned JSON written by [`save`](@ref)."""
SmolyakApproximation(path::AbstractString) = _load_approximation(_read_json(path))

"""Return the state-space dimension of a fitted approximation."""
dimension(::SmolyakApproximation{D}) where {D} = D

"""Return the canonical coefficient vector stored by `res` without copying it."""
coefficients(res::SmolyakApproximation) = res.coeffs

"""Solve a cached fitting system for a validated floating right-hand side."""
function _solve(plan::FitPlan{D,T}, values) where {D,T}
    length(values) == length(plan.grid) || throw(DimensionMismatch(
        "expected $(length(plan.grid)) node values; received $(length(values))"))
    y = T.(values)
    all(isfinite, y) || throw(ArgumentError("all node values must be finite"))
    if iszero(plan.ridge_lambda)
        return Vector{T}(plan.factorization \ y), y
    end
    rhs = [y; zeros(T, length(plan.basis))]
    return Vector{T}(plan.factorization \ rhs), y
end

"""
    fit(plan, node_values; verbose=true)

Fit coefficients using the cached plan and return a new approximation with
empirical node RMSE and maximum absolute residual diagnostics.
"""
function fit(plan::FitPlan{D,T}, node_values; verbose::Bool = true) where {D,T}
    coeffs, y = _solve(plan, node_values)
    residual = plan.design * coeffs - y
    rmse = sqrt(sum(abs2, residual) / length(residual))
    max_error = maximum(abs, residual)
    result = SmolyakApproximation{D,T,typeof(plan.grid.domain),
        typeof(plan.grid),typeof(plan.basis)}(
        plan.grid.domain, plan.grid, plan.basis, coeffs, plan.system_type,
        plan.solver, plan.ridge_lambda, rmse, max_error)
    verbose && println("Fitted $(length(coeffs)) coefficients; node RMSE = $rmse.")
    return result
end

"""
    fit(fun, plan; multi_threading=false, verbose=true)

Convenience route equivalent to [`evaluate_nodes`](@ref) followed by
`fit(plan, values)`.
"""
function fit(fun, plan::FitPlan; multi_threading::Bool = false,
             verbose::Bool = true)
    values = evaluate_nodes(fun, plan.grid;
                            multi_threading = multi_threading, verbose = verbose)
    return fit(plan, values; verbose = verbose)
end

"""Return whether a result and plan describe exactly the same grid and basis."""
function _compatible(res::SmolyakApproximation, plan::FitPlan)
    return dimension(res) == dimension(plan.grid) &&
           res.grid.max_levels == plan.grid.max_levels &&
           res.grid.level_budget == plan.grid.level_budget &&
           res.grid.nodes == plan.grid.nodes &&
           res.basis.max_orders == plan.basis.max_orders &&
           res.basis.order_budget == plan.basis.order_budget &&
           res.basis.indices == plan.basis.indices
end

"""
    fit!(res, plan, new_node_values; verbose=true)

Refit a structurally compatible approximation in place using the cached
factorization. Coefficients and diagnostics are updated; grid and basis are not.
"""
function fit!(res::SmolyakApproximation{D,T}, plan::FitPlan,
              new_node_values; verbose::Bool = true) where {D,T}
    _compatible(res, plan) || throw(ArgumentError(
        "approximation and fit plan have incompatible grid or basis structure"))
    eltype(plan.design) === T || throw(ArgumentError(
        "approximation coefficient type $T differs from plan type $(eltype(plan.design))"))
    coeffs, y = _solve(plan, new_node_values)
    copyto!(res.coeffs, coeffs)
    residual = plan.design * coeffs - y
    res.node_rmse = sqrt(sum(abs2, residual) / length(residual))
    res.max_node_error = maximum(abs, residual)
    res.system_type = plan.system_type
    res.solver = plan.solver
    res.ridge_lambda = plan.ridge_lambda
    verbose && println("Refitted $(length(coeffs)) coefficients; node RMSE = " *
                       "$(res.node_rmse).")
    return res
end

function Base.show(io::IO, res::SmolyakApproximation{D,T}) where {D,T}
    print(io, "SmolyakApproximation{$D,$T}(nodes=", length(res.grid),
          ", coefficients=", length(res.coeffs), ", system=", res.system_type,
          ", solver=", res.solver, ", ridge_lambda=", res.ridge_lambda,
          ", node_rmse=", res.node_rmse,
          ", max_node_error=", res.max_node_error, ")")
end

# ------------------------------------------------------------------------------
# CPU evaluation and explicit basis diagnostics
# ------------------------------------------------------------------------------

"""Fused single-point coefficient accumulation without a basis-vector allocation."""
function _evaluate_point(::Type{T}, domain::BoxDomain{D}, indices,
                         coeffs, x) where {T,D}
    total = zero(T)
    for k in eachindex(indices)
        total = muladd(T(coeffs[k]), _term_value(T, domain, indices[k], x), total)
    end
    return total
end

"""Evaluate `res` at one physical point using its coefficient precision."""
function (res::SmolyakApproximation{D,T})(x::AbstractVector) where {D,T}
    _validate_point(res.domain, x)
    return _evaluate_point(T, res.domain, res.basis.indices, res.coeffs, x)
end

"""Evaluate `res` at point rows in `X`, returning one value per row."""
function (res::SmolyakApproximation{D,T})(X::AbstractMatrix) where {D,T}
    _validate_points(res.domain, X)
    output = Vector{T}(undef, size(X, 1))
    for i in axes(X, 1)
        output[i] = _evaluate_point(T, res.domain, res.basis.indices,
                                    res.coeffs, @view X[i, :])
    end
    return output
end

"""Evaluate one point explicitly in floating type `T`."""
function (res::SmolyakApproximation)(::Type{T}, x::AbstractVector) where {T<:AbstractFloat}
    _validate_point(res.domain, x)
    return _evaluate_point(T, res.domain, res.basis.indices, res.coeffs, x)
end

"""Evaluate a batch explicitly in floating type `T`."""
function (res::SmolyakApproximation)(::Type{T}, X::AbstractMatrix) where {T<:AbstractFloat}
    _validate_points(res.domain, X)
    output = Vector{T}(undef, size(X, 1))
    for i in axes(X, 1)
        output[i] = _evaluate_point(T, res.domain, res.basis.indices,
                                    res.coeffs, @view X[i, :])
    end
    return output
end

"""Construct a CPU basis vector at one physical point."""
basis_vector(res::SmolyakApproximation, x; backend::Symbol = :cpu) =
    basis_vector(eltype(res.coeffs), res, x; backend = backend)

"""Construct a basis vector in explicit precision `T`."""
function basis_vector(::Type{T}, res::SmolyakApproximation, x;
                      backend::Symbol = :cpu) where {T<:AbstractFloat}
    if backend !== :cpu
        return basis_vector(prepare(res; T = T, backend = backend), x)
    end
    _validate_point(res.domain, x)
    vector = Vector{T}(undef, length(res.basis))
    for k in eachindex(res.basis.indices)
        vector[k] = _term_value(T, res.domain, res.basis.indices[k], x)
    end
    return vector
end

"""Construct a dense CPU basis matrix for physical point rows."""
basis_matrix(res::SmolyakApproximation, X; backend::Symbol = :cpu) =
    basis_matrix(eltype(res.coeffs), res, X; backend = backend)

"""Construct an explicit-precision basis matrix for physical point rows."""
function basis_matrix(::Type{T}, res::SmolyakApproximation{D}, X::AbstractMatrix;
                      backend::Symbol = :cpu) where {T<:AbstractFloat,D}
    if backend !== :cpu
        return basis_matrix(prepare(res; T = T, backend = backend), X)
    end
    _validate_points(res.domain, X)
    matrix = Matrix{T}(undef, size(X, 1), length(res.basis))
    for i in axes(X, 1), k in eachindex(res.basis.indices)
        matrix[i, k] = _term_value(T, res.domain, res.basis.indices[k],
                                   @view X[i, :])
    end
    return matrix
end

# ------------------------------------------------------------------------------
# Prepared evaluator and backend registry
# ------------------------------------------------------------------------------

"""
    CPUPreparedEvaluator

Backend-ready immutable CPU data. Coefficients, polynomial indices, and affine
metadata are converted once by [`prepare`](@ref), then reused by calls.
"""
struct CPUPreparedEvaluator{D,T<:AbstractFloat,Dom}
    domain::Dom
    coeffs::Vector{T}
    indices::Matrix{Int}
end

"""Evaluate one point with a prepared CPU evaluator."""
function (evaluator::CPUPreparedEvaluator{D,T})(x::AbstractVector) where {D,T}
    _validate_point(evaluator.domain, x)
    total = zero(T)
    for k in axes(evaluator.indices, 1)
        term = one(T)
        for d in 1:D
            xi = (T(x[d]) - T(evaluator.domain.center[d])) *
                 T(evaluator.domain.invhalfwidth[d])
            term *= _chebyshev_value(evaluator.indices[k, d], xi)
        end
        total = muladd(evaluator.coeffs[k], term, total)
    end
    return total
end

"""Evaluate point rows with a prepared CPU evaluator."""
function (evaluator::CPUPreparedEvaluator{D,T})(X::AbstractMatrix) where {D,T}
    _validate_points(evaluator.domain, X)
    output = Vector{T}(undef, size(X, 1))
    for i in axes(X, 1)
        total = zero(T)
        for k in axes(evaluator.indices, 1)
            term = one(T)
            for d in 1:D
                xi = (T(X[i, d]) - T(evaluator.domain.center[d])) *
                     T(evaluator.domain.invhalfwidth[d])
                term *= _chebyshev_value(evaluator.indices[k, d], xi)
            end
            total = muladd(evaluator.coeffs[k], term, total)
        end
        output[i] = total
    end
    return output
end

"""Construct a basis vector using a prepared CPU evaluator."""
function basis_vector(evaluator::CPUPreparedEvaluator{D,T}, x) where {D,T}
    _validate_point(evaluator.domain, x)
    result = Vector{T}(undef, size(evaluator.indices, 1))
    for k in axes(evaluator.indices, 1)
        term = one(T)
        for d in 1:D
            xi = (T(x[d]) - T(evaluator.domain.center[d])) *
                 T(evaluator.domain.invhalfwidth[d])
            term *= _chebyshev_value(evaluator.indices[k, d], xi)
        end
        result[k] = term
    end
    return result
end

"""Construct a basis matrix using a prepared CPU evaluator."""
function basis_matrix(evaluator::CPUPreparedEvaluator{D,T}, X::AbstractMatrix) where {D,T}
    _validate_points(evaluator.domain, X)
    result = Matrix{T}(undef, size(X, 1), size(evaluator.indices, 1))
    for i in axes(X, 1), k in axes(evaluator.indices, 1)
        term = one(T)
        for d in 1:D
            xi = (T(X[i, d]) - T(evaluator.domain.center[d])) *
                 T(evaluator.domain.invhalfwidth[d])
            term *= _chebyshev_value(evaluator.indices[k, d], xi)
        end
        result[i, k] = term
    end
    return result
end

const _BACKEND_FACTORIES = Dict{Symbol,Any}()

"""Register an optional backend factory. Intended for sidecar extensions."""
function _register_backend!(name::Symbol, factory)
    name in (:cpu, :auto) && throw(ArgumentError("backend name $name is reserved"))
    _BACKEND_FACTORIES[name] = factory
    return name
end

"""Return registered executable backends, always beginning with `:cpu`."""
function available_backends()
    preferred = (:cuda, :metal)
    known = Symbol[:cpu]
    append!(known, filter(name -> haskey(_BACKEND_FACTORIES, name), preferred))
    extras = sort!(collect(setdiff(keys(_BACKEND_FACTORIES), preferred)))
    append!(known, extras)
    return Tuple(known)
end

"""
    prepare(res; T=eltype(coefficients(res)), backend=:cpu)

Create reusable backend-ready evaluator data. `:auto` conservatively selects
CPU and never changes precision. Optional sidecars register other backends.
"""
function prepare(res::SmolyakApproximation; T::Type{<:AbstractFloat} =
                 eltype(res.coeffs), backend::Symbol = :cpu)
    selected = backend === :auto ? :cpu : backend
    if selected === :cpu
        return CPUPreparedEvaluator{dimension(res),T,typeof(res.domain)}(
            res.domain, T.(res.coeffs), _index_matrix(res.basis.indices))
    end
    factory = get(_BACKEND_FACTORIES, selected, nothing)
    factory === nothing && throw(ArgumentError(
        "backend $selected is unavailable; loaded backends are " *
        "$(available_backends()). Load its optional sidecar first."))
    return factory(res, T)
end

# ------------------------------------------------------------------------------
# Versioned JSON serialization
# ------------------------------------------------------------------------------

"""Read and parse a JSON object, adding a useful filename to parse errors."""
function _read_json(path::AbstractString)
    isfile(path) || throw(ArgumentError("serialized file does not exist: $path"))
    try
        return JSON3.read(read(path, String))
    catch error
        throw(ArgumentError("could not parse serialized JSON file $path: $error"))
    end
end

"""Validate a serialized object's schema version and kind."""
function _validate_serialized(data, expected_kind::AbstractString)
    haskey(data, :format_version) || throw(ArgumentError("missing format_version"))
    data.format_version == FORMAT_VERSION || throw(ArgumentError(
        "unsupported format_version $(data.format_version); expected $FORMAT_VERSION"))
    haskey(data, :kind) && String(data.kind) == expected_kind || throw(ArgumentError(
        "serialized object kind must be $expected_kind"))
    return nothing
end

"""Return the serializable dictionary for a domain."""
function _domain_data(domain::BoxDomain)
    return Dict("format_version" => FORMAT_VERSION, "kind" => "BoxDomain",
                "numeric_type" => string(eltype(domain.lb)),
                "lower_bounds" => collect(domain.lb),
                "upper_bounds" => collect(domain.ub))
end

"""Recover the requested floating type from a supported serialized name."""
function _serialized_float(name)
    value = String(name)
    value == "Float32" && return Float32
    value == "Float64" && return Float64
    throw(ArgumentError("unsupported serialized floating type $value"))
end

"""Load a validated domain JSON object."""
function _load_domain(data)
    _validate_serialized(data, "BoxDomain")
    T = _serialized_float(data.numeric_type)
    return BoxDomain(T.(data.lower_bounds), T.(data.upper_bounds))
end

"""Return the serializable dictionary for a sparse grid."""
function _grid_data(grid::SmolyakGrid)
    return Dict("format_version" => FORMAT_VERSION, "kind" => "SmolyakGrid",
                "domain" => _domain_data(grid.domain),
                "max_levels" => collect(grid.max_levels),
                "level_budget" => grid.level_budget, "rule" => String(grid.rule),
                "indices" => [collect(index) for index in grid.indices],
                "nodes" => [collect(@view grid.nodes[i, :]) for i in axes(grid.nodes, 1)])
end

"""Load a validated grid JSON object and verify deterministic reconstruction."""
function _load_grid(data)
    _validate_serialized(data, "SmolyakGrid")
    domain = _load_domain(data.domain)
    grid = SmolyakGrid(Int.(data.max_levels), Int(data.level_budget), domain;
                       rule = Symbol(data.rule))
    stored_indices = [ntuple(d -> Int(row[d]), dimension(domain)) for row in data.indices]
    stored_nodes = reduce(vcat, permutedims.(collect.(data.nodes)))
    T = eltype(domain.lb)
    stored_matrix = T.(stored_nodes)
    grid.indices == stored_indices || throw(ArgumentError(
        "serialized grid indices do not match its reconstruction metadata"))
    grid.nodes == stored_matrix || throw(ArgumentError(
        "serialized grid nodes do not match deterministic reconstruction"))
    return grid
end

"""Return the serializable dictionary for a basis specification."""
function _basis_data(basis::ChebyshevBasisSpec)
    return Dict("format_version" => FORMAT_VERSION,
                "kind" => "ChebyshevBasisSpec",
                "max_orders" => collect(basis.max_orders),
                "order_budget" => basis.order_budget,
                "indices" => [collect(index) for index in basis.indices])
end

"""Load and validate a basis JSON object."""
function _load_basis(data)
    _validate_serialized(data, "ChebyshevBasisSpec")
    basis = ChebyshevBasisSpec(Int.(data.max_orders), Int(data.order_budget))
    stored = [ntuple(d -> Int(row[d]), dimension(basis)) for row in data.indices]
    basis.indices == stored || throw(ArgumentError(
        "serialized basis indices do not match its reconstruction metadata"))
    return basis
end

"""Return persistent approximation data without runtime/factorization state."""
function _approximation_data(res::SmolyakApproximation)
    return Dict("format_version" => FORMAT_VERSION,
                "kind" => "SmolyakApproximation",
                "grid" => _grid_data(res.grid), "basis" => _basis_data(res.basis),
                "coefficient_type" => string(eltype(res.coeffs)),
                "coefficients" => res.coeffs, "system_type" => String(res.system_type),
                "solver" => String(res.solver), "ridge_lambda" => res.ridge_lambda,
                "node_rmse" => res.node_rmse,
                "max_node_error" => res.max_node_error)
end

"""Load and validate an approximation JSON object."""
function _load_approximation(data)
    _validate_serialized(data, "SmolyakApproximation")
    grid = _load_grid(data.grid)
    basis = _load_basis(data.basis)
    dimension(grid) == dimension(basis) || throw(DimensionMismatch(
        "serialized grid and basis dimensions differ"))
    T = _serialized_float(data.coefficient_type)
    coeffs = T.(data.coefficients)
    length(coeffs) == length(basis) || throw(DimensionMismatch(
        "serialized coefficient count does not match the basis"))
    D = dimension(grid)
    return SmolyakApproximation{D,T,typeof(grid.domain),typeof(grid),typeof(basis)}(
        grid.domain, grid, basis, coeffs, Symbol(data.system_type),
        Symbol(data.solver), T(data.ridge_lambda), T(data.node_rmse),
        T(data.max_node_error))
end

"""
    save(obj, path)

Write a `BoxDomain`, `SmolyakGrid`, `ChebyshevBasisSpec`, or
`SmolyakApproximation` as human-readable, versioned JSON. Prepared evaluators
and transient factorization data are intentionally unsupported.
"""
function save(obj, path::AbstractString)
    data = if obj isa BoxDomain
        _domain_data(obj)
    elseif obj isa SmolyakGrid
        _grid_data(obj)
    elseif obj isa ChebyshevBasisSpec
        _basis_data(obj)
    elseif obj isa SmolyakApproximation
        _approximation_data(obj)
    else
        throw(ArgumentError("serialization is unsupported for $(typeof(obj))"))
    end
    open(path, "w") do io
        JSON3.pretty(io, data)
        write(io, '\n')
    end
    return path
end

end # module SmolyakPoly
