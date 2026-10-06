# ==============================================================================
# Smolyax.jl -- proposed usage example
#
# Design principles:
#
# 1. The module approximates scalar-valued functions only:
#
#        f : R^D -> R
#
#    Vector-valued functions are intentionally outside the scope.
#
# 2. The numerical pipeline is separated into five conceptually orthogonal
#    objects:
#
#        Domain
#          |
#        Grid            where is the expensive function evaluated?
#          |
#        Basis           in which function space is it approximated?
#          |
#        FitPlan         how are coefficients recovered efficiently?
#          |
#        Approximation   what fitted mathematical function do we obtain?
#          |
#        Evaluator       where / at what precision is it evaluated?
#
# 3. Grid accuracy and polynomial accuracy are completely independent.
#
#        grid level budget != polynomial order budget
#
#    Interpolation is therefore only a special case. It is NOT defined by
#    equality of these two accuracy parameters.
#
# 4. Bulk points are always represented as dense matrices:
#
#        size(X) == (#points, D)
#
#    Rows correspond to points and columns correspond to coordinates.
#
# 5. Canonical fitting is performed in Float64 unless explicitly requested
#    otherwise. Evaluation precision and hardware backend are independent.
#
# 6. Smolyax.jl works by itself on CPU.
#
#    Optional files:
#
#        SmolyaxCUDAExt.jl
#        SmolyaxMetalExt.jl
#
#    add CUDA / Apple-GPU execution, respectively.
#
# ==============================================================================


# ------------------------------------------------------------------------------
# Load the portable CPU implementation
# ------------------------------------------------------------------------------

include("Smolyax.jl")

import .Smolyax

const smx = Smolyax


# The core module must always work without CUDA.jl or Metal.jl.
#
# With only Smolyax.jl available:
#
#     smx.available_backends()
#
# should return something conceptually equivalent to:
#
#     (:cpu,)
#
# GPU support is optional and discussed later in this example.


# ==============================================================================
# 1. Define the bounded domain
# ==============================================================================

# ------------------------------------------------------------------------------
# The Domain object only describes the mathematical state space.
#
# It knows:
#
#     - dimensionality D
#     - lower bounds
#     - upper bounds
#     - affine transformations between the physical domain and [-1, 1]^D
#
# It does NOT know:
#
#     - how grid points are generated
#     - which polynomial basis will be used
#     - what hardware will be used
#
# Since this module is intended only for bounded-domain problems, no
# Gauss-Hermite / unbounded-domain abstraction is required.
# ------------------------------------------------------------------------------

domain = smx.BoxDomain(
    [1.0, -1.0, 3.0],       # lower bounds
    [2.0,  1.0, 10.0],      # upper bounds
)::smx.BoxDomain{3,Float64}


# Formatted display
display(domain)

# Expected conceptual output:
#
# Box Domain{3, Float64}:
#   x1 in [1.0,  2.0]
#   x2 in [-1.0, 1.0]
#   x3 in [3.0, 10.0]


# Dimensionality means the dimension of the mathematical state space.
#
# Do not overload Base.ndims for this purpose, because ndims conventionally
# refers to the number of array axes.

smx.dimension(domain)       # == 3


# Check whether a point belongs to the domain.
#
# A point is represented by any suitable vector-like object with length D.

[1.5, 0.0, 5.0] in domain      # true
[3.0, 0.0, 5.0] in domain      # false


# Explicitly project a point onto the box.
#
# Projection is a domain operation rather than a grid operation.

smx.project(domain, [3.0, 0.0, 12.0])

# expected:
#
# [2.0, 0.0, 10.0]


# The Domain object owns the physical <-> canonical coordinate map.
#
# For Chebyshev and many other polynomial bases, the canonical coordinates are
# in [-1, 1]^D.

x = [1.5, 0.0, 5.0]

ξ = smx.to_canonical(domain, x)

x_recovered = smx.from_canonical(domain, ξ)


# ==============================================================================
# 2. Construct the sparse Leja grid
# ==============================================================================

# ------------------------------------------------------------------------------
# The Grid answers one question only:
#
#       "At which state-space locations should the expensive model be solved?"
#
# It consists conceptually of:
#
#       Domain
#       +
#       one-dimensional Leja node rule
#       +
#       sparse hierarchical multi-index set Λ_G
#       +
#       materialized physical nodes X_G
#
# The grid does NOT contain any polynomial approximation assumptions.
# ------------------------------------------------------------------------------


grid = smx.SmolyakGrid(
    [3, 4, 5],          # maximum hierarchy level allowed in each dimension
    6,                  # total sparse-grid level budget
    domain;
    rule = :leja,
)::smx.SmolyakGrid{3}


# The public constructor is deliberately non-parametric.
#
# The dimensionality D = 3 is inferred from the supplied vectors/domain and is
# encoded in the concrete returned type.


display(grid)

# Expected conceptual output:
#
# Smolyak Grid{3} (Leja):
#   # nodes        = ...
#   level budget   = 6
#   max levels     = (3, 4, 5)
#
#   x1 in [1.0,  2.0], max level = 3
#   x2 in [-1.0, 1.0], max level = 4
#   x3 in [3.0, 10.0], max level = 5


# Number of materialized grid nodes

length(grid)


# Mathematical state-space dimension

smx.dimension(grid)             # == 3


# The logical shape of the materialized grid is:
#
#       (#nodes, D)

size(grid)


# ------------------------------------------------------------------------------
# Bulk node storage
# ------------------------------------------------------------------------------

# Nodes are stored as ONE dense matrix:
#
#       size(grid.nodes) == (#nodes, D)
#
# Each row is a state-space point.
#
# We intentionally do NOT use
#
#       Vector{Vector{Float64}}
#
# because dense matrix storage is substantially more suitable for:
#
#       - CPU cache locality
#       - SIMD
#       - GPU transfer
#       - GPU coalesced access
#       - basis evaluation
#       - serialization

grid.nodes


# For example:

grid.nodes[1, :]                 # first node, presented as a row slice

grid.nodes[:, 2]                 # all grid values of coordinate x2


# ------------------------------------------------------------------------------
# Sparse-grid hierarchical multi-indices
# ------------------------------------------------------------------------------

# Multi-indices can be stored internally as:
#
#       NTuple{D,Int}
#
# because D is encoded in the Grid type and tuples are compact and type-stable.
#
# These indices describe Λ_G -- the GRID hierarchy only.

grid.indices


# Conceptually:
#
#     grid.indices == Λ_G
#
# There is no requirement that Λ_G equal the polynomial index set Λ_P.


# ==============================================================================
# 3. Random and quasi-random points in the domain
# ==============================================================================

import Random


# ------------------------------------------------------------------------------
# Ordinary random sampling
# ------------------------------------------------------------------------------

rng = Random.default_rng()

x_random = rand(rng, domain)

# result:
#
# Vector{Float64} of length D


X_random = rand(rng, domain, 100)

# result:
#
# Matrix{Float64}
#
# size(X_random) == (100, D)


# ------------------------------------------------------------------------------
# Sobol quasi-Monte-Carlo points
# ------------------------------------------------------------------------------

# Sobol sequences are deliberately NOT exposed through Base.rand because they
# are low-discrepancy deterministic/quasi-random sequences, not iid draws.
#
# Sobol.jl may be used internally.

X_sobol = smx.sobol(
    domain,
    100,
)

# size(X_sobol) == (100, D)


# ==============================================================================
# 4. Define the polynomial approximation space
# ==============================================================================

# ------------------------------------------------------------------------------
# The Basis answers a completely different question:
#
#       "In which finite-dimensional function space should f be approximated?"
#
# It knows nothing about how the grid was generated.
#
# In this example we use a sparse Chebyshev polynomial space:
#
#       f_hat(x)
#           =
#       sum_{α in Λ_P} c_α T_α(ξ(x))
#
# where:
#
#       ξ(x) in [-1,1]^D
#
# is supplied by the Domain object.
#
# Importantly:
#
#       Λ_P != Λ_G
#
# is completely valid.
# ------------------------------------------------------------------------------


bspec = smx.ChebyshevBasisSpec(
    [3, 4, 5],          # maximum polynomial order / hierarchy per dimension
    5,                  # total polynomial order budget
)::smx.ChebyshevBasisSpec{3}


display(bspec)

# Expected conceptual output:
#
# Chebyshev Basis{3}:
#   order budget    = 5
#   max orders      = (3, 4, 5)
#   # basis terms   = ...


smx.dimension(bspec)             # == 3


# Polynomial multi-index set Λ_P

bspec.indices


# Again:
#
#       grid.indices   == Λ_G
#       bspec.indices  == Λ_P
#
# These are independent mathematical objects.


# For example, this is completely valid:
#
#       grid  level budget = 6
#       basis order budget = 5
#
# which means:
#
#       solve the expensive model on a relatively rich grid,
#       but fit a smoother / lower-dimensional polynomial representation.


# ==============================================================================
# 5. Build a reusable FitPlan
# ==============================================================================

# ------------------------------------------------------------------------------
# This is an important performance-oriented layer.
#
# For fixed:
#
#       grid
#       basis
#       fitting method
#       ridge parameter
#
# the design matrix
#
#       Φ[n,k] = φ_k(x_n)
#
# does not change across repeated fits.
#
# Therefore a FitPlan may cache:
#
#       - canonicalized grid coordinates
#       - basis design matrix Φ
#       - QR factorization
#       - ridge-augmented factorization, if needed
#       - reusable CPU workspaces
#
# The expensive structural work is then performed only once.
#
# This is particularly useful inside value-function iteration / policy
# iteration, where node values change repeatedly but grid and basis do not.
# ------------------------------------------------------------------------------


plan = smx.FitPlan(
    grid,
    bspec;
    ridge_lambda = 0.0,
    solver = :qr,
)


display(plan)

# Possible output:
#
# Smolyak FitPlan{3}:
#   # nodes           = 231
#   # basis terms     = 126
#   system            = overdetermined
#   solver            = QR
#   ridge lambda      = 0.0
#   cached design     = yes
#   cached factorization = yes


# ------------------------------------------------------------------------------
# Interpretation of interpolation vs approximation
# ------------------------------------------------------------------------------

# Interpolation is NOT defined by:
#
#       grid level budget == basis order budget
#
# Instead it depends on the actual collocation system:
#
#       Φ c = y
#
# If:
#
#       #nodes == #basis terms
#
# and Φ has full rank, the system is square and unisolvent, and the fit can be
# classified as interpolation.
#
# If:
#
#       #nodes > #basis terms
#
# the system is overdetermined and the coefficients are obtained by
# least-squares approximation.
#
# If:
#
#       #nodes < #basis terms
#
# the system is underdetermined and should normally trigger an error unless the
# user explicitly requests an underdetermined/regularized fitting strategy.


# ==============================================================================
# 6. Evaluate the expensive function at the grid nodes
# ==============================================================================

# ------------------------------------------------------------------------------
# Suppose the expensive function is:
#
#       f : R^3 -> R
#
# The callable may be:
#
#       - an ordinary Julia function
#       - a closure
#       - a callable struct
#
# We deliberately do NOT require ::Function or a hypothetical ::Callable type.
# Julia should specialize on the concrete callable type automatically.
# ------------------------------------------------------------------------------


fun = x -> sin(x[1]) + x[2]^2 + log(x[3])


# Evaluate the function at all grid nodes.
#
# This operation is conceptually separate from polynomial fitting because an
# arbitrary economic-model solver is NOT assumed to be GPU-compatible.
#
# CPU multithreading is useful here because node solves are typically
# embarrassingly parallel.

node_values = smx.evaluate_nodes(
    fun,
    grid;
    multi_threading = true,
    verbose = true,
)

# result:
#
# Vector{Float64}
#
# length(node_values) == length(grid)


# Internally, each point supplied to `fun` is vector-like and always has length
# D.
#
# Even when D == 1, a node remains a one-element vector-like object:
#
#       [x]
#
# rather than a scalar x.
#
# This keeps the API dimensionally consistent.


# ==============================================================================
# 7. Fit the approximation
# ==============================================================================

# ------------------------------------------------------------------------------
# Performance-oriented route:
#
#       expensive model evaluation
#                |
#                v
#          node_values
#                |
#                v
#            FitPlan
#                |
#                v
#         coefficients c
# ------------------------------------------------------------------------------


res = smx.fit(
    plan,
    node_values;
    verbose = true,
)::smx.SmolyakApproximation{3}


display(res)

# Expected conceptual output:
#
# Smolyak Approximation{3, Float64}:
#   grid level budget       = 6
#   basis order budget      = 5
#   # nodes                 = 231
#   # coefficients          = 126
#
#   fit type                = Approximation
#   system                  = overdetermined
#   solver                  = QR
#   ridge lambda            = 0.0
#
#   node RMSE               = ...
#   maximum node error      = ...


smx.dimension(res)             # == 3


# Canonical coefficients retain fitting precision.

approx_coefs = smx.coefficients(res)

# expected:
#
# Vector{Float64}


# ------------------------------------------------------------------------------
# Convenience route
# ------------------------------------------------------------------------------

# For one-off use, the following convenience call may combine node evaluation
# and fitting:
#
#     res = smx.fit(
#         fun,
#         plan;
#         multi_threading = true,
#         verbose = true,
#     )
#
# Internally this is conceptually equivalent to:
#
#     y   = smx.evaluate_nodes(fun, grid; ...)
#     res = smx.fit(plan, y; ...)
#
# Keeping the lower-level route exposed is important for expensive quantitative
# models where the user may already have node values available.


# ==============================================================================
# 8. Direct CPU evaluation
# ==============================================================================

# ------------------------------------------------------------------------------
# The approximation object itself is callable.
#
# Calling the raw approximation defaults to:
#
#       backend   = :cpu
#       precision = coefficient precision
#
# Therefore a Float64-fitted approximation returns Float64 by default.
#
# Hardware choice NEVER silently changes arithmetic precision.
# ------------------------------------------------------------------------------


x_test = rand(rng, domain)

X_test = rand(rng, domain, 100)


# Single point

y_test = res(x_test)

# expected:
#
# Float64


# Multiple points

Y_test = res(X_test)

# expected:
#
# Vector{Float64}
#
# length(Y_test) == 100


# Explicit evaluation precision is also supported.

y32 = res(Float32, x_test)

Y32 = res(Float32, X_test)

# expected:
#
# Float32
# Vector{Float32}


# Attempting to evaluate outside the domain throws an error by default.
#
# Smolyax intentionally makes no extrapolation assumption.

# res([100.0, 0.0, 5.0])
#
# ERROR: point lies outside approximation domain


# If projection is intentionally desired, it is explicit:

x_projected = smx.project(domain, [100.0, 0.0, 5.0])

res(x_projected)


# ==============================================================================
# 9. Optional CUDA support
# ==============================================================================

# ------------------------------------------------------------------------------
# SmolyaxCUDAExt.jl is optional.
#
# The core Smolyax.jl file must NOT depend on CUDA.jl.
#
# On an NVIDIA machine:
#
#     import CUDA
#     include("SmolyaxCUDAExt.jl")
#
# The extension file registers CUDA-specific functionality with Smolyax.
#
# If the file is absent, everything above still works unchanged on CPU.
# ------------------------------------------------------------------------------


# Example, only on an NVIDIA machine:
#
# import CUDA
# include("SmolyaxCUDAExt.jl")
#
# smx.available_backends()
#
# possible result:
#
#     (:cpu, :cuda)


# ==============================================================================
# 10. Optional Apple Metal support
# ==============================================================================

# ------------------------------------------------------------------------------
# On Apple Silicon:
#
#     import Metal
#     include("SmolyaxMetalExt.jl")
#
# Again, the extension is optional.
# ------------------------------------------------------------------------------


# Example, only on an Apple Silicon Mac:
#
# import Metal
# include("SmolyaxMetalExt.jl")
#
# smx.available_backends()
#
# possible result:
#
#     (:cpu, :metal)


# ==============================================================================
# 11. Prepared evaluators
# ==============================================================================

# ------------------------------------------------------------------------------
# This is the preferred high-performance API for repeated evaluation.
#
# Calling:
#
#       res(X; backend=:cuda)
#
# as a convenience method is possible, but may require repeatedly:
#
#       - moving coefficients to the device
#       - moving polynomial index metadata
#       - preparing device-specific buffers
#
# For repeated evaluation we instead explicitly PREPARE an evaluator once.
#
# The evaluator owns backend-specific copies of all hot-path data.
# ------------------------------------------------------------------------------


cpu_eval = smx.prepare(
    res;
    T = Float64,
    backend = :cpu,
)


# Repeated calls now contain no preparation overhead.

y1 = cpu_eval(x_test)

Y1 = cpu_eval(X_test)


# ------------------------------------------------------------------------------
# CUDA example
# ------------------------------------------------------------------------------

# After loading CUDA extension:
#
# cuda_eval = smx.prepare(
#     res;
#     T = Float32,
#     backend = :cuda,
# )
#
#
# The evaluator now stores device-ready:
#
#     Float32 coefficients
#     polynomial multi-indices
#     domain-transform metadata
#     recurrence metadata
#     reusable GPU buffers where appropriate
#
#
# Then repeated calls:
#
#     Y1 = cuda_eval(X1)
#     Y2 = cuda_eval(X2)
#     Y3 = cuda_eval(X3)
#
# avoid repeatedly preparing the approximation.


# ------------------------------------------------------------------------------
# Metal example
# ------------------------------------------------------------------------------

# After loading Metal extension:
#
# metal_eval = smx.prepare(
#     res;
#     T = Float32,
#     backend = :metal,
# )
#
# Y = metal_eval(X_test)


# ------------------------------------------------------------------------------
# Precision and hardware are deliberately independent.
# ------------------------------------------------------------------------------

# These are conceptually different choices:
#
#     smx.prepare(res; T=Float64, backend=:cpu)
#     smx.prepare(res; T=Float32, backend=:cpu)
#     smx.prepare(res; T=Float64, backend=:cuda)
#     smx.prepare(res; T=Float32, backend=:cuda)
#     smx.prepare(res; T=Float32, backend=:metal)
#
# Unsupported hardware/precision combinations should produce a clear error
# rather than silently changing precision.


# ==============================================================================
# 12. Optional :auto backend
# ==============================================================================

# ------------------------------------------------------------------------------
# :auto may be offered as a convenience, but should not mean:
#
#       GPU exists -> always use GPU
#
# because GPU launch and transfer overhead can dominate for small batches.
#
# A future heuristic may inspect:
#
#       # evaluation points
#       # basis terms
#       state dimension
#       available backends
#
# before choosing a device.
#
# Importantly, :auto chooses HARDWARE ONLY. It never changes precision.
# ------------------------------------------------------------------------------


auto_eval = smx.prepare(
    res;
    T = Float32,
    backend = :auto,
)


# If no GPU extension has been loaded:
#
#       :auto -> :cpu
#
# If a GPU extension is available, the implementation may choose a GPU only
# when the expected batch size makes doing so worthwhile.


# ==============================================================================
# 13. Basis-vector / basis-matrix interface
# ==============================================================================

# ------------------------------------------------------------------------------
# Smolyax should also expose the conventional basis-coefficient language:
#
#       f_hat(X) = B(X) c
#
# This interface is useful for:
#
#       diagnostics
#       testing
#       external regressions
#       inspecting basis terms
#
# It is NOT the preferred high-performance evaluation implementation.
#
# Direct evaluation should use fused recurrence + accumulation and should NOT
# materialize a huge basis matrix internally.
# ------------------------------------------------------------------------------


# Single point

bvec_test = smx.basis_vector(
    res,
    x_test,
)

# expected:
#
# Vector{Float64}
#
# length(bvec_test) == length(smx.coefficients(res))


# Batch

bmat_test = smx.basis_matrix(
    res,
    X_test,
)

# expected:
#
# Matrix{Float64}
#
# size(bmat_test) ==
#
#     (#points, #basis terms)


# Explicit lower precision

bmat32 = smx.basis_matrix(
    Float32,
    res,
    X_test,
)


# ------------------------------------------------------------------------------
# GPU basis matrices
# ------------------------------------------------------------------------------

# If explicitly requested on CUDA:
#
#     B_cuda = smx.basis_matrix(
#         Float32,
#         res,
#         X_test;
#         backend = :cuda,
#     )
#
# the returned object should remain on the GPU, e.g. a CuMatrix.
#
# It should NOT silently copy the result back to an ordinary CPU Matrix.
#
#
# Likewise:
#
#     B_metal = smx.basis_matrix(
#         Float32,
#         res,
#         X_test;
#         backend = :metal,
#     )
#
# should remain a Metal-backed matrix.


# ==============================================================================
# 14. Validate basis-coefficient equivalence
# ==============================================================================

import Test


c = smx.coefficients(res)

b = smx.basis_vector(res, x_test)

B = smx.basis_matrix(res, X_test)


Test.@test isapprox(
    dot(b, c),
    res(x_test);
    rtol = 1e-12,
    atol = 1e-12,
)


Test.@test all(
    isapprox.(
        B * c,
        res(X_test);
        rtol = 1e-11,
        atol = 1e-12,
    )
)


# Exact equality is deliberately NOT required because:
#
#       - direct evaluation and matrix multiplication may sum in different order
#       - GPU reductions may use different reduction trees
#       - fused multiply-add behavior may differ
#       - Float32 and Float64 have different rounding behavior


# ==============================================================================
# 15. Re-fit the same grid/basis efficiently
# ==============================================================================

# ------------------------------------------------------------------------------
# This use case is particularly important in quantitative dynamic models.
#
# Suppose:
#
#       grid is unchanged
#       basis is unchanged
#
# but a new iteration generates new function values:
#
#       y^(1)
#       y^(2)
#       y^(3)
#       ...
#
# The FitPlan already contains the reusable design matrix / factorization.
#
# Therefore re-fitting should reuse it rather than reconstructing Φ and QR.
# ------------------------------------------------------------------------------


new_node_values = [
    sin(grid.nodes[i, 1]) +
    grid.nodes[i, 2]^2 +
    0.5 * log(grid.nodes[i, 3])
    for i in axes(grid.nodes, 1)
]


smx.fit!(
    res,
    plan,
    new_node_values;
    verbose = true,
)


# After fit!:
#
#       res.coefs
#       res.rmse
#       res.max_error
#       other fit diagnostics
#
# are updated in-place.
#
# But:
#
#       grid
#       basis
#       Φ
#       QR factorization structure
#
# are reused.


# ------------------------------------------------------------------------------
# Prepared evaluators become stale after coefficients change.
# ------------------------------------------------------------------------------

# To avoid ambiguous behavior, the safest rule is:
#
#     modifying res invalidates previously prepared Evaluator objects.
#
# The user explicitly prepares a new one:
#
#     cuda_eval = smx.prepare(res; T=Float32, backend=:cuda)
#
# This avoids hidden host -> device synchronization inside fit!.


# ==============================================================================
# 16. Serialization
# ==============================================================================

# ------------------------------------------------------------------------------
# Persistent mathematical objects may be serialized.
#
# Runtime caches should NOT be serialized:
#
#       FitPlan QR workspaces
#       GPU arrays
#       compiled kernels
#       temporary buffers
#       PreparedEvaluator
#
# Those are reconstructed when needed.
# ------------------------------------------------------------------------------


smx.save(domain, "domain.json")

domain2 = smx.BoxDomain("domain.json")


smx.save(grid, "grid.json")

grid2 = smx.SmolyakGrid("grid.json")


smx.save(bspec, "basis.json")

bspec2 = smx.ChebyshevBasisSpec("basis.json")


smx.save(res, "approximation.json")

res2 = smx.SmolyakApproximation("approximation.json")


# Every serialized object should contain a format version such as:
#
#       "format_version": 1
#
# so that future structural changes can be migrated explicitly.


# ==============================================================================
# 17. A compact end-to-end workflow
# ==============================================================================

# After understanding the individual components, a normal quantitative workflow
# can be very short:


domain = smx.BoxDomain(
    [1.0, -1.0, 3.0],
    [2.0,  1.0, 10.0],
)

grid = smx.SmolyakGrid(
    [3, 4, 5],
    6,
    domain;
    rule = :leja,
)

basis = smx.ChebyshevBasisSpec(
    [3, 4, 5],
    5,
)

plan = smx.FitPlan(
    grid,
    basis;
    solver = :qr,
    ridge_lambda = 0.0,
)

fun = x -> sin(x[1]) + x[2]^2 + log(x[3])

res = smx.fit(
    fun,
    plan;
    multi_threading = true,
)


# Ordinary CPU use:

X = smx.sobol(domain, 10_000)

Y = res(X)


# Repeated high-throughput GPU use, if an optional extension is available:
#
#     evaluator = smx.prepare(
#         res;
#         T = Float32,
#         backend = :cuda,      # or :metal
#     )
#
#     Y = evaluator(X)


# Repeated model iteration:
#
#     for iteration in 1:maxiter
#
#         y_new = solve_model_at_nodes(grid, ...)
#
#         smx.fit!(
#             res,
#             plan,
#             y_new,
#         )
#
#         ...
#     end
#
# Here the expensive basis construction and QR structure are reused across
# iterations.