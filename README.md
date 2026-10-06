# SmolyakPoly.jl

SmolyakPoly.jl approximates scalar-valued functions on bounded, rectangular
domains with sparse, nested Leja grids and multivariate Chebyshev polynomials.
Its distinguishing design choice is that the sampling grid and polynomial space
are independent: users may evaluate an expensive model on a relatively rich
sparse grid while fitting a smaller, smoother polynomial space. The package
also separates reusable fitting work from fitted coefficients and separates the
mathematical approximation from CPU, CUDA, or Metal execution. Dense
row-oriented batch storage, cached QR factorizations, fused prediction, explicit
precision selection, versioned JSON serialization, and optional GPU extensions
make the same workflow suitable for one-off approximations and repeated model
iterations.

The current scope is deliberately focused:

- functions have the form `f: ℝᴰ → ℝ`;
- domains are finite boxes;
- sparse-grid nodes use a deterministic, nested real Leja sequence;
- approximation spaces use first-kind Chebyshev polynomials;
- CPU support is always available, while CUDA and Metal are optional;
- points in a batch are rows of a dense `N × D` matrix.

## Mathematics

Let the physical domain be the box

```math
\mathcal X = \prod_{d=1}^{D}[a_d,b_d].
```

`BoxDomain` owns the affine transformation from a physical point
`x ∈ 𝒳` to a canonical point `ξ(x) ∈ [-1,1]^D`:

```math
\xi_d(x)=\frac{x_d-(a_d+b_d)/2}{(b_d-a_d)/2}.
```

### Sparse Leja grid

For coordinate-level caps `L = (L₁,…,Lᴰ)` and a total level budget `q_G`,
SmolyakPoly.jl constructs the lower index set

```math
\Lambda_G = \left\{\alpha\in\mathbb N_0^D:
\alpha_d\le L_d,\ \sum_{d=1}^{D}\alpha_d\le q_G\right\}.
```

Levels are zero-based. If `z₀,z₁,…` is the deterministic nested Leja sequence
on `[-1,1]`, an index `α ∈ Λ_G` selects the canonical point
`(z_{α₁},…,z_{αᴰ})`; the domain transformation maps that point into `𝒳`.
Coordinate caps permit anisotropy, while the total budget prevents construction
of the full tensor grid.

### Chebyshev approximation space

The polynomial space has its own coordinate-order caps
`P = (P₁,…,Pᴰ)` and total order budget `q_P`:

```math
\Lambda_P = \left\{\beta\in\mathbb N_0^D:
\beta_d\le P_d,\ \sum_{d=1}^{D}\beta_d\le q_P\right\}.
```

For first-kind Chebyshev polynomials,

```math
T_0(t)=1,\qquad T_1(t)=t,\qquad
T_{k+1}(t)=2tT_k(t)-T_{k-1}(t),
```

the fitted function is

```math
\widehat f(x)=\sum_{\beta\in\Lambda_P}
c_\beta\prod_{d=1}^{D}T_{\beta_d}(\xi_d(x)).
```

At grid nodes `xₙ`, define the design matrix

```math
\Phi_{n,k}=\prod_{d=1}^{D}T_{\beta_{k,d}}(\xi_d(x_n)).
```

The coefficients solve `Φc ≈ y`. Ordinary fits use a pivoted QR
factorization. With `ridge_lambda = λ > 0`, the package applies QR to the
augmented least-squares system

```math
\begin{bmatrix}\Phi\\\sqrt{\lambda}I\end{bmatrix}c
\approx
\begin{bmatrix}y\\0\end{bmatrix},
```

without forming normal equations or an explicit inverse.

### Difference from standard Smolyak interpolation

In a standard Smolyak interpolation construction, the sparse node set and
polynomial space are normally coupled through one admissible multi-index set.
This produces a square interpolation problem when the construction is
unisolvent. SmolyakPoly.jl deliberately allows

```math
\Lambda_G \ne \Lambda_P.
```

The consequences are practical:

- `length(grid) == length(basis)` gives a square system; it is interpolation
  only when the actual design matrix has full rank;
- `length(grid) > length(basis)` gives an overdetermined least-squares
  approximation, which is useful when sampling can be richer than the desired
  polynomial representation;
- `length(grid) < length(basis)` is underdetermined and is rejected by the
  current implementation.

Grid and basis budgets therefore control different errors. A larger grid can
provide more information about the target function, while a smaller basis can
regularize or smooth the fitted representation. Equality of nominal budgets is
not used to decide whether a fit is interpolation.

## Pipeline

The package is organized as a sequence of mathematical and runtime objects:

```text
BoxDomain
    ↓
SmolyakGrid
    ↓
ChebyshevBasisSpec
    ↓
FitPlan
    ↓
SmolyakApproximation
    ↓
prepared evaluator
```

Each object answers one question:

- `BoxDomain`: where is the function defined?
- `SmolyakGrid`: where should the expensive function be evaluated?
- `ChebyshevBasisSpec`: in which finite-dimensional space should it be
  represented?
- `FitPlan`: how should coefficients be recovered, and which structural work
  can be reused?
- `SmolyakApproximation`: what fitted mathematical function was obtained?
- `prepare`: on which backend and at which precision will that function be
  evaluated repeatedly?

This separation is intentional. In particular, arbitrary user functions are
evaluated at grid nodes on the CPU; fitting is a linear-algebra operation; and
GPU support accelerates evaluation of the resulting polynomial without making
the original function GPU-compatible.


### 1. Load the package and define a domain

Activate this project, instantiate its dependencies, and load the package:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using SmolyakPoly
using Random
using LinearAlgebra
```

Define a three-dimensional bounded domain:

```julia
domain = BoxDomain(
    [1.0, -1.0,  3.0],  # lower bounds
    [2.0,  1.0, 10.0],  # upper bounds
)

dimension(domain)               # 3
[1.5, 0.0, 5.0] in domain      # true
[3.0, 0.0, 5.0] in domain      # false
```

Bounds are promoted to a floating-point type, must be finite, and must satisfy
`lower[d] < upper[d]`. A point is always vector-like and has length `D`; even
for `D == 1`, use `[x]`, not the scalar `x`.

The domain also owns projection and canonical-coordinate transformations:

```julia
x = [1.5, 0.0, 5.0]
xi = to_canonical(domain, x)
x_recovered = from_canonical(domain, xi)

project(domain, [3.0, 0.0, 12.0])  # [2.0, 0.0, 10.0]
```

Projection is explicit. Approximation evaluation does not silently extrapolate
or clamp points outside the domain.

### 2. Construct the sparse sampling grid

```julia
grid = SmolyakGrid(
    [3, 4, 5],  # maximum zero-based level in each coordinate
    6,          # total grid-level budget
    domain;
    rule = :leja,
)

dimension(grid)  # 3
length(grid)     # number of materialized nodes
size(grid)       # (number of nodes, 3)
```

`grid.indices` is `Λ_G`, and `grid.nodes` is a dense physical-coordinate
matrix. The storage convention is important throughout the package:

```text
size(X) == (number of points, dimension)
```

Rows are points and columns are coordinates. This convention applies to grid
nodes, random samples, Sobol samples, batch prediction, and explicit basis
matrices. It avoids `Vector{Vector}` storage and provides predictable memory
layout for CPU and accelerator execution.

The Leja sequence is nested and deterministically ordered, so enlarging a
one-dimensional level retains the earlier nodes. The lower-set construction
produces unique tensor points and preserves node order during serialization.

Random and low-discrepancy points use the same row-oriented convention:

```julia
rng = Random.default_rng()

x_random = rand(rng, domain)       # Vector of length 3
X_random = rand(rng, domain, 100)  # 100 × 3 Matrix
X_sobol = sobol(domain, 10_000)    # 10_000 × 3 Matrix
```

### 3. Choose the polynomial space independently

```julia
basis = ChebyshevBasisSpec(
    [3, 4, 5],  # maximum polynomial order in each coordinate
    5,          # total polynomial-order budget
)

dimension(basis)  # 3
length(basis)     # number of coefficients
basis.indices     # Λ_P
```

The grid budget is `6`, whereas the basis budget is `5`. This is valid and is
the central distinction of the package: `grid.indices` and `basis.indices`
describe different mathematical objects. Here the model is sampled on a richer
grid and represented by a smaller polynomial space.

Choose the grid according to where model solves are informative and affordable.
Choose the basis according to the complexity and smoothness that the stored
approximation should retain. Inspect `length(grid)` and `length(basis)` rather
than comparing their nominal budgets.

### 4. Build a reusable `FitPlan`

```julia
plan = FitPlan(
    grid,
    basis;
    solver = :qr,
    ridge_lambda = 0.0,
)
```

For fixed grid, basis, solver, and ridge parameter, the following do not change:

- canonicalized grid coordinates;
- design matrix `Φ`;
- system classification;
- QR factorization, including the augmented factorization for ridge fitting.

`FitPlan` computes and caches this structural work once. This matters when a
simulation, value-function iteration, policy iteration, or parameter loop
produces many new node-value vectors on the same grid and basis. Without a plan,
each fit would rebuild the design matrix and factorization even though only the
right-hand side changed.

The constructor rejects mismatched dimensions, invalid ridge parameters,
underdetermined systems, and unregularized rank-deficient systems. Displaying a
plan reports the node and term counts, whether the system is square or
overdetermined, the solver, and the ridge parameter:

```julia
display(plan)
```

Use ridge fitting when shrinkage is wanted:

```julia
ridge_plan = FitPlan(grid, basis; ridge_lambda = 1e-8)
```

The ridge parameter changes the fitting problem, so it is part of the plan
rather than an argument supplied separately to every fit.

### 5. Evaluate the expensive function and fit

The target can be any callable object returning one scalar:

```julia
fun = x -> sin(x[1]) + x[2]^2 + log(x[3])
```

For explicit control, first evaluate it at the nodes and then fit:

```julia
node_values = evaluate_nodes(
    fun,
    grid;
    multi_threading = true,
    verbose = true,
)

res = fit(plan, node_values; verbose = true)
```

`evaluate_nodes` calls `fun` with one vector-like point per grid row.
Multithreading uses Julia's thread pool and is useful when individual model
solves are expensive. Start Julia with an appropriate thread count, for example
`julia --threads=auto --project=.`. The callable itself and any state it uses
must be safe for concurrent calls.

For one-off work, the convenience route is equivalent:

```julia
res = fit(
    fun,
    plan;
    multi_threading = true,
    verbose = true,
)
```

The resulting `SmolyakApproximation` contains the domain, grid and basis
metadata, coefficients, fitting choices, node RMSE, and maximum absolute node
error:

```julia
dimension(res)
coefficients(res)
display(res)
```

Node RMSE is an empirical diagnostic, not a declaration that the fit is exact.
Even for a square system, interpolation depends on the actual rank and numerical
conditioning of `Φ`.

### 6. Evaluate the fitted approximation

The fitted object is callable on one point or a batch:

```julia
x_test = [1.5, 0.25, 6.0]
X_test = sobol(domain, 1_000)

y = res(x_test)  # scalar
Y = res(X_test)  # Vector with 1_000 entries
```

Default evaluation uses the CPU and the stored coefficient precision. Select a
different arithmetic precision explicitly:

```julia
y32 = res(Float32, x_test)
Y32 = res(Float32, X_test)
```

This converts evaluation arithmetic; it does not mutate the stored canonical
coefficients. Backend choice never silently changes precision.

Out-of-domain points throw an informative error:

```julia
# res([100.0, 0.0, 5.0])  # DomainError

x_projected = project(domain, [100.0, 0.0, 5.0])
res(x_projected)
```

### 7. Inspect the basis when needed

Ordinary prediction evaluates Chebyshev recurrences and accumulates
coefficients directly; it does not materialize a full basis matrix. Explicit
basis APIs are available for diagnostics and interoperability:

```julia
b = basis_vector(res, x_test)
B = basis_matrix(res, X_test)
c = coefficients(res)

isapprox(dot(b, c), res(x_test); rtol = 1e-12, atol = 1e-12)
isapprox(B * c, res(X_test); rtol = 1e-11, atol = 1e-12)
```

Use numerical tolerances rather than exact equality: fused prediction and
matrix multiplication may accumulate terms in different orders, especially at
different precisions or on accelerators.

The explicit-precision variants are:

```julia
b32 = basis_vector(Float32, res, x_test)
B32 = basis_matrix(Float32, res, X_test)
```

### 8. Prepare an evaluator for repeated prediction

A `SmolyakApproximation` is the persistent mathematical result. A prepared
evaluator is transient execution state for a particular backend and precision:

```julia
cpu_eval = prepare(res; T = Float64, backend = :cpu)

y1 = cpu_eval(x_test)
Y1 = cpu_eval(X_test)
Y2 = cpu_eval(X_sobol)
```

Preparation converts and stores coefficients, polynomial indices, and domain
transformation metadata once. On a GPU it also performs the one-time device
uploads. Reusing the evaluator avoids reconstructing metadata or transferring
unchanged coefficients before each prediction call.

Available backends can be inspected with:

```julia
available_backends()  # (:cpu,) before an optional GPU extension is loaded
```

CUDA is activated automatically by Julia's extension mechanism after CUDA.jl is
installed and imported:

```julia
import CUDA

cuda_eval = prepare(res; T = Float32, backend = :cuda)
Y_cuda = cuda_eval(X_test)
```

Likewise, on a compatible Apple Silicon system:

```julia
import Metal

metal_eval = prepare(res; T = Float32, backend = :metal)
Y_metal = metal_eval(X_test)
```

Metal evaluation currently requires explicit `Float32`; unsupported precision
requests fail instead of being silently downgraded. Batch results from GPU
evaluators and GPU basis-matrix requests remain on the selected device. Copy
them to the host explicitly when needed.

`backend = :auto` is intentionally conservative at present:

```julia
auto_eval = prepare(res; T = Float32, backend = :auto)
```

It currently selects CPU. This avoids arbitrary workload thresholds and keeps
precision unchanged; future benchmark-supported policies can use the same API.

Prepared evaluators are not serialized. They are cheap runtime views of a
persistent approximation and should be reconstructed in each process.

### 9. Refit without rebuilding structural work

When new values become available on the same grid, reuse the plan:

```julia
new_node_values = [
    sin(grid.nodes[i, 1]) +
    grid.nodes[i, 2]^2 +
    0.5 * log(grid.nodes[i, 3])
    for i in axes(grid.nodes, 1)
]

fit!(res, plan, new_node_values; verbose = true)
```

`fit!` updates coefficients and diagnostics while preserving the grid, basis,
design matrix, and cached factorization. It rejects structurally incompatible
plans.

A previously prepared evaluator contains a snapshot of the old coefficients.
Prepare a new evaluator after `fit!`:

```julia
cpu_eval = prepare(res; T = Float64, backend = :cpu)
```

This explicit refresh avoids hidden synchronization or device transfers.

### 10. Save persistent mathematical objects

Domains, grids, basis specifications, and fitted approximations support
versioned JSON serialization:

```julia
save(domain, "domain.json")
save(grid, "grid.json")
save(basis, "basis.json")
save(res, "approximation.json")

domain_loaded = BoxDomain("domain.json")
grid_loaded = SmolyakGrid("grid.json")
basis_loaded = ChebyshevBasisSpec("basis.json")
res_loaded = SmolyakApproximation("approximation.json")
```

Files contain `"format_version": 1`. Grid node ordering and coefficient
ordering are preserved. `FitPlan` factorizations, GPU arrays, kernels,
workspaces, and prepared evaluators are runtime details and are intentionally
not serialized.

### Compact end-to-end example

```julia
using SmolyakPoly

domain = BoxDomain(
    [1.0, -1.0,  3.0],
    [2.0,  1.0, 10.0],
)

grid = SmolyakGrid([3, 4, 5], 6, domain)
basis = ChebyshevBasisSpec([3, 4, 5], 5)
plan = FitPlan(grid, basis)

fun = x -> sin(x[1]) + x[2]^2 + log(x[3])
res = fit(fun, plan; multi_threading = true)

X = sobol(domain, 10_000)
Y = res(X)

evaluator = prepare(res; T = Float64, backend = :cpu)
Y_repeated = evaluator(X)
```

## References

- Kenneth L. Judd, Lilia Maliar, Serguei Maliar, and Rafael Valero (2014),
  “Smolyak Method for Solving Dynamic Economic Models: Lagrange Interpolation,
  Anisotropic Grid and Adaptive Domain,” *Journal of Economic Dynamics and
  Control*, 44, 92–123.
  [doi:10.1016/j.jedc.2014.03.003](https://doi.org/10.1016/j.jedc.2014.03.003)
- Josephine Westermann and Joshua Chen,
  [`smolyax`](https://github.com/JoWestermann/smolyax), a high-performance JAX
  implementation of the Smolyak interpolation operator.
- R. J. Dennis,
  [`SmolyakApprox.jl`](https://github.com/RJDennis/SmolyakApprox.jl), a Julia
  package for Smolyak approximation with polynomial and piecewise-linear
  interpolation facilities.

These projects are useful points of comparison and sources of broader Smolyak
method context. SmolyakPoly.jl has a narrower scalar, bounded-domain scope and
emphasizes independent grid/basis specifications, reusable QR-based fitting,
and prepared backend-specific evaluation.

## License

SmolyakPoly.jl is released under the [MIT License](LICENSE).

## AI usage disclaimer

The author used OpenAI Codex as an assistive development tool to help refine the
public API and pipeline design and to help draft portions of the source code and
documentation. The author has personally and carefully reviewed the resulting
source files and has run the test suite on the author's own devices. Codex's
assistance does not replace that review, and no claim is made that automated
assistance guarantees correctness. The author remains responsible for the
package and welcomes clear, constructive reports of any issues that may be
discovered in future use.
