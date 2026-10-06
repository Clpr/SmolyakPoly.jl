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

"""Prepared CUDA data for physical-coordinate gradients."""
struct SmolyakPolyCUDAGradientEvaluator{D,T,Dom,C,I,V}
    domain::Dom
    coeffs::C
    indices::I
    center::V
    invhalfwidth::V
end

"""Prepared CUDA data for physical-coordinate Hessians."""
struct SmolyakPolyCUDAHessianEvaluator{D,T,Dom,C,I,V}
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

"""Device-compatible Chebyshev value and first two derivatives."""
@inline function _cuda_chebyshev_derivatives(order::Int32, x)
    order == 0 && return (one(x), zero(x), zero(x))
    order == 1 && return (x, one(x), zero(x))
    previous_value = one(x)
    value = x
    previous_first = zero(x)
    first = one(x)
    previous_second = zero(x)
    second = zero(x)
    for _ in Int32(2):order
        following_value = muladd((one(x) + one(x)) * x, value, -previous_value)
        following_first = (one(x) + one(x)) * value +
                          (one(x) + one(x)) * x * first - previous_first
        following_second = (one(x) + one(x) + one(x) + one(x)) * first +
                           (one(x) + one(x)) * x * second - previous_second
        previous_value = value
        value = following_value
        previous_first = first
        first = following_first
        previous_second = second
        second = following_second
    end
    return (value, first, second)
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

"""CUDA kernel for gradients, one thread per point-coordinate pair."""
function _cuda_gradient_kernel!(output, X, coeffs, indices, center,
                                invhalfwidth, term_count::Int32,
                                dimension_count::Int32)
    linear = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if linear <= length(output)
        i = mod(linear - 1, size(output, 1)) + 1
        j = div(linear - 1, size(output, 1)) + 1
        total = zero(eltype(output))
        for k in Int32(1):term_count
            term = coeffs[k]
            for d in Int32(1):dimension_count
                xi = (X[i, d] - center[d]) * invhalfwidth[d]
                value, first, _ = _cuda_chebyshev_derivatives(indices[k, d], xi)
                term *= d == j ? first * invhalfwidth[d] : value
            end
            total += term
        end
        output[i, j] = total
    end
    return
end

"""CUDA kernel for Hessians, one thread per point-coordinate pair."""
function _cuda_hessian_kernel!(output, X, coeffs, indices, center,
                               invhalfwidth, term_count::Int32,
                               dimension_count::Int32)
    linear = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if linear <= length(output)
        point_count = size(output, 1)
        i = mod(linear - 1, point_count) + 1
        remainder = div(linear - 1, point_count)
        j = mod(remainder, dimension_count) + 1
        l = div(remainder, dimension_count) + 1
        total = zero(eltype(output))
        for k in Int32(1):term_count
            term = coeffs[k]
            for d in Int32(1):dimension_count
                xi = (X[i, d] - center[d]) * invhalfwidth[d]
                value, first, second = _cuda_chebyshev_derivatives(
                    indices[k, d], xi)
                if d == j && d == l
                    term *= second * invhalfwidth[d] * invhalfwidth[d]
                elseif d == j || d == l
                    term *= first * invhalfwidth[d]
                else
                    term *= value
                end
            end
            total += term
        end
        output[i, j, l] = total
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
    coeffs_device = CUDA.CuArray(T.(res.coeffs))
    indices_device = CUDA.CuArray(indices)
    center_device = CUDA.CuArray(center)
    inverse_device = CUDA.CuArray(inverse)
    return SmolyakPolyCUDAEvaluator{D,T,typeof(res.domain),
        typeof(coeffs_device),typeof(indices_device),typeof(center_device)}(
        res.domain, coeffs_device, indices_device, center_device, inverse_device)
end

"""Create a CUDA gradient evaluator with one-time device uploads."""
function _prepare_cuda_gradient(res, ::Type{T}) where {T<:AbstractFloat}
    base = _prepare_cuda(res, T)
    D = SmolyakPoly.dimension(res)
    return SmolyakPolyCUDAGradientEvaluator{D,T,typeof(base.domain),
        typeof(base.coeffs),typeof(base.indices),typeof(base.center)}(
        base.domain, base.coeffs, base.indices, base.center, base.invhalfwidth)
end

"""Create a CUDA Hessian evaluator with one-time device uploads."""
function _prepare_cuda_hessian(res, ::Type{T}) where {T<:AbstractFloat}
    base = _prepare_cuda(res, T)
    D = SmolyakPoly.dimension(res)
    return SmolyakPolyCUDAHessianEvaluator{D,T,typeof(base.domain),
        typeof(base.coeffs),typeof(base.indices),typeof(base.center)}(
        base.domain, base.coeffs, base.indices, base.center, base.invhalfwidth)
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


_cuda_batch(evaluator::SmolyakPolyCUDAGradientEvaluator{D,T},
            X::AbstractMatrix) where {D,T} =
    _cuda_batch(SmolyakPolyCUDAEvaluator{D,T,typeof(evaluator.domain),
        typeof(evaluator.coeffs),typeof(evaluator.indices),typeof(evaluator.center)}(
        evaluator.domain, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth), X)

_cuda_batch(evaluator::SmolyakPolyCUDAHessianEvaluator{D,T},
            X::AbstractMatrix) where {D,T} =
    _cuda_batch(SmolyakPolyCUDAEvaluator{D,T,typeof(evaluator.domain),
        typeof(evaluator.coeffs),typeof(evaluator.indices),typeof(evaluator.center)}(
        evaluator.domain, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth), X)

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

# ------------------------------------------------------------------------------
# CUDA gradient and Hessian calls
# ------------------------------------------------------------------------------

"""Evaluate CUDA gradients and return a host `N × D Matrix{T}`."""
function (evaluator::SmolyakPolyCUDAGradientEvaluator{D,T})(
        X::AbstractMatrix) where {D,T}
    device = _cuda_batch(evaluator, X)
    output = CUDA.CuArray{T}(undef, size(device, 1), D)
    threads = 256
    blocks = cld(length(output), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_gradient_kernel!(
        output, device, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(length(evaluator.coeffs)), Int32(D))
    return Array(output)
end

"""Evaluate one CUDA gradient and return a host `Vector{T}`."""
function (evaluator::SmolyakPolyCUDAGradientEvaluator{D,T})(
        x::AbstractVector) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    return vec(evaluator(reshape(T.(collect(x)), 1, D)))
end

"""Evaluate CUDA Hessians and return one host matrix per point row."""
function (evaluator::SmolyakPolyCUDAHessianEvaluator{D,T})(
        X::AbstractMatrix) where {D,T}
    device = _cuda_batch(evaluator, X)
    output = CUDA.CuArray{T}(undef, size(device, 1), D, D)
    threads = 256
    blocks = cld(length(output), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_hessian_kernel!(
        output, device, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(length(evaluator.coeffs)), Int32(D))
    host = Array(output)
    return [Matrix(@view host[i, :, :]) for i in axes(host, 1)]
end

"""Evaluate one CUDA Hessian and return a host `D × D Matrix{T}`."""
function (evaluator::SmolyakPolyCUDAHessianEvaluator{D,T})(
        x::AbstractVector) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    return first(evaluator(reshape(T.(collect(x)), 1, D)))
end

SmolyakPoly._register_backend!(:cuda, _prepare_cuda)
SmolyakPoly._register_gradient_backend!(:cuda, _prepare_cuda_gradient)
SmolyakPoly._register_hessian_backend!(:cuda, _prepare_cuda_hessian)

end # module SmolyakPolyCUDAExt
