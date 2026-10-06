module SmolyakPolyCUDAExt

using SmolyakPoly
import CUDA
import SmolyakPoly: basis_vector, basis_matrix

# ------------------------------------------------------------------------------
# CUDA prepared storage
# ------------------------------------------------------------------------------

"""
    SmolyakPolyCUDAEvaluator

Prepared CUDA data. All numeric and index arrays are uploaded once by
`SmolyakPoly.prepare`; repeated batch calls reuse them and return `CuArray`
results. Float32 and Float64 are accepted when the active device supports them.
"""
struct SmolyakPolyCUDAEvaluator{D,T,Dom,C,I,V}
    domain::Dom
    coeffs::C
    indices::I
    center::V
    invhalfwidth::V
end

"""Device-compatible scalar Chebyshev recurrence."""
@inline function _cuda_chebyshev(order::Int32, x)
    order == 0 && return one(x)
    order == 1 && return x
    previous = one(x)
    current = x
    for _ in Int32(2):order
        following = muladd((one(x) + one(x)) * x, current, -previous)
        previous = current
        current = following
    end
    return current
end

"""CUDA kernel for fused polynomial evaluation, one thread per point row."""
function _cuda_evaluate_kernel!(output, X, coeffs, indices, center,
                                invhalfwidth, term_count::Int32,
                                dimension_count::Int32)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if i <= size(X, 1)
        total = zero(eltype(output))
        for k in Int32(1):term_count
            term = one(eltype(output))
            for d in Int32(1):dimension_count
                xi = (X[i, d] - center[d]) * invhalfwidth[d]
                term *= _cuda_chebyshev(indices[k, d], xi)
            end
            total = muladd(coeffs[k], term, total)
        end
        output[i] = total
    end
    return
end

"""CUDA kernel for explicit basis-matrix construction."""
function _cuda_basis_kernel!(output, X, indices, center, invhalfwidth,
                             dimension_count::Int32)
    linear = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if linear <= length(output)
        i = mod(linear - 1, size(output, 1)) + 1
        k = div(linear - 1, size(output, 1)) + 1
        term = one(eltype(output))
        for d in Int32(1):dimension_count
            xi = (X[i, d] - center[d]) * invhalfwidth[d]
            term *= _cuda_chebyshev(indices[k, d], xi)
        end
        output[i, k] = term
    end
    return
end

# ------------------------------------------------------------------------------
# CUDA preparation and calls
# ------------------------------------------------------------------------------

"""Create a CUDA evaluator or report unavailable hardware/precision clearly."""
function _prepare_cuda(res, ::Type{T}) where {T<:AbstractFloat}
    CUDA.functional() || throw(ArgumentError(
        "CUDA backend was loaded, but CUDA.functional() is false"))
    T in (Float32, Float64) || throw(ArgumentError(
        "CUDA preparation supports Float32 and Float64; received $T"))
    D = SmolyakPoly.dimension(res)
    indices = Int32.(SmolyakPoly._index_matrix(res.basis.indices))
    center = T.(collect(res.domain.center))
    inverse = T.(collect(res.domain.invhalfwidth))
    return SmolyakPolyCUDAEvaluator{D,T,typeof(res.domain),
        typeof(CUDA.CuArray(T.(res.coeffs))),typeof(CUDA.CuArray(indices)),
        typeof(CUDA.CuArray(center))}(
        res.domain, CUDA.CuArray(T.(res.coeffs)), CUDA.CuArray(indices),
        CUDA.CuArray(center), CUDA.CuArray(inverse))
end

"""Convert a host/device batch to a precision-matched CUDA matrix."""
function _cuda_batch(evaluator::SmolyakPolyCUDAEvaluator{D,T}, X::AbstractMatrix) where {D,T}
    if X isa CUDA.CuArray
        size(X, 2) == D || throw(DimensionMismatch(
            "CUDA batch must have $D columns; received size $(size(X))"))
        device = T.(X)
        lower = CUDA.CuArray(T.(collect(evaluator.domain.lb)))
        upper = CUDA.CuArray(T.(collect(evaluator.domain.ub)))
        valid = CUDA.all((device .>= reshape(lower, 1, D)) .&
                         (device .<= reshape(upper, 1, D)))
        valid || throw(DomainError(X, "CUDA batch contains points outside the domain"))
        return device
    end
    SmolyakPoly._validate_points(evaluator.domain, X)
    return CUDA.CuArray(T.(X))
end

"""Evaluate a point batch and leave the result on the CUDA device."""
function (evaluator::SmolyakPolyCUDAEvaluator{D,T})(X::AbstractMatrix) where {D,T}
    device = _cuda_batch(evaluator, X)
    output = CUDA.CuArray{T}(undef, size(device, 1))
    threads = 256
    blocks = cld(length(output), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_evaluate_kernel!(
        output, device, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(length(evaluator.coeffs)), Int32(D))
    return output
end

"""Evaluate one point on CUDA and return the scalar to the caller."""
function (evaluator::SmolyakPolyCUDAEvaluator{D,T})(x::AbstractVector) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    batch = reshape(T.(collect(x)), 1, D)
    return Array(evaluator(batch))[1]
end

"""Construct a CUDA basis matrix and keep it on device."""
function basis_matrix(evaluator::SmolyakPolyCUDAEvaluator{D,T},
                      X::AbstractMatrix) where {D,T}
    device = _cuda_batch(evaluator, X)
    output = CUDA.CuArray{T}(undef, size(device, 1), size(evaluator.indices, 1))
    threads = 256
    blocks = cld(length(output), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_basis_kernel!(
        output, device, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(D))
    return output
end

"""Construct one CUDA basis vector and keep it on device."""
function basis_vector(evaluator::SmolyakPolyCUDAEvaluator{D,T}, x) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    matrix = basis_matrix(evaluator, reshape(T.(collect(x)), 1, D))
    return vec(matrix)
end

SmolyakPoly._register_backend!(:cuda, _prepare_cuda)

end # module SmolyakPolyCUDAExt
