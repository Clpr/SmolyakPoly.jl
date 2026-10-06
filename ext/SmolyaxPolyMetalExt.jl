module SmolyaxPolyMetalExt

using SmolyakPoly
import Metal
import SmolyakPoly: basis_vector, basis_matrix

# ------------------------------------------------------------------------------
# Metal prepared storage
# ------------------------------------------------------------------------------

"""
    SmolyaxPolyMetalEvaluator

Prepared Apple Metal evaluator. Metal supports explicit Float32 evaluation;
unsupported precision requests are rejected rather than silently converted.
"""
struct SmolyaxPolyMetalEvaluator{D,T,Dom,C,I,V}
    domain::Dom
    coeffs::C
    indices::I
    center::V
    invhalfwidth::V
end

"""Prepared Metal data for physical-coordinate gradients."""
struct SmolyaxPolyMetalGradientEvaluator{D,T,Dom,C,I,V}
    domain::Dom
    coeffs::C
    indices::I
    center::V
    invhalfwidth::V
end

"""Prepared Metal data for physical-coordinate Hessians."""
struct SmolyaxPolyMetalHessianEvaluator{D,T,Dom,C,I,V}
    domain::Dom
    coeffs::C
    indices::I
    center::V
    invhalfwidth::V
end

"""Device-compatible scalar Chebyshev recurrence for Metal kernels."""
@inline function _metal_chebyshev(order::Int32, x)
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
@inline function _metal_chebyshev_derivatives(order::Int32, x)
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

"""Metal kernel for fused approximation evaluation."""
function _metal_evaluate_kernel!(output, X, coeffs, indices, center,
                                 invhalfwidth, term_count::Int32,
                                 dimension_count::Int32)
    i = Metal.thread_position_in_grid_1d()
    if i <= size(X, 1)
        total = zero(eltype(output))
        for k in Int32(1):term_count
            term = one(eltype(output))
            for d in Int32(1):dimension_count
                xi = (X[i, d] - center[d]) * invhalfwidth[d]
                term *= _metal_chebyshev(indices[k, d], xi)
            end
            total = muladd(coeffs[k], term, total)
        end
        output[i] = total
    end
    return
end

"""Metal kernel for an explicit basis matrix."""
function _metal_basis_kernel!(output, X, indices, center, invhalfwidth,
                              dimension_count::Int32)
    linear = Metal.thread_position_in_grid_1d()
    if linear <= length(output)
        i = mod(linear - 1, size(output, 1)) + 1
        k = div(linear - 1, size(output, 1)) + 1
        term = one(eltype(output))
        for d in Int32(1):dimension_count
            xi = (X[i, d] - center[d]) * invhalfwidth[d]
            term *= _metal_chebyshev(indices[k, d], xi)
        end
        output[i, k] = term
    end
    return
end

"""Metal kernel for gradients, one thread per point-coordinate pair."""
function _metal_gradient_kernel!(output, X, coeffs, indices, center,
                                 invhalfwidth, term_count::Int32,
                                 dimension_count::Int32)
    linear = Metal.thread_position_in_grid_1d()
    if linear <= length(output)
        i = mod(linear - 1, size(output, 1)) + 1
        j = div(linear - 1, size(output, 1)) + 1
        total = zero(eltype(output))
        for k in Int32(1):term_count
            term = coeffs[k]
            for d in Int32(1):dimension_count
                xi = (X[i, d] - center[d]) * invhalfwidth[d]
                value, first, _ = _metal_chebyshev_derivatives(indices[k, d], xi)
                term *= d == j ? first * invhalfwidth[d] : value
            end
            total += term
        end
        output[i, j] = total
    end
    return
end

"""Metal kernel for Hessians, one thread per point-coordinate pair."""
function _metal_hessian_kernel!(output, X, coeffs, indices, center,
                                invhalfwidth, term_count::Int32,
                                dimension_count::Int32)
    linear = Metal.thread_position_in_grid_1d()
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
                value, first, second = _metal_chebyshev_derivatives(
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
# Metal preparation and calls
# ------------------------------------------------------------------------------

"""Create a Float32 Metal evaluator with one-time device uploads."""
function _prepare_metal(res, ::Type{T}) where {T<:AbstractFloat}
    T === Float32 || throw(ArgumentError(
        "Metal preparation supports Float32 only; requested $T. " *
        "No implicit precision downgrade is performed."))
    D = SmolyakPoly.dimension(res)
    indices = Int32.(SmolyakPoly._index_matrix(res.basis.indices))
    center = T.(collect(res.domain.center))
    inverse = T.(collect(res.domain.invhalfwidth))
    coeffs_device = Metal.MtlArray(T.(res.coeffs))
    indices_device = Metal.MtlArray(indices)
    center_device = Metal.MtlArray(center)
    return SmolyaxPolyMetalEvaluator{D,T,typeof(res.domain),
        typeof(coeffs_device),typeof(indices_device),typeof(center_device)}(
        res.domain, coeffs_device, indices_device, center_device,
        Metal.MtlArray(inverse))
end

"""Create a Float32 Metal gradient evaluator."""
function _prepare_metal_gradient(res, ::Type{T}) where {T<:AbstractFloat}
    base = _prepare_metal(res, T)
    D = SmolyakPoly.dimension(res)
    return SmolyaxPolyMetalGradientEvaluator{D,T,typeof(base.domain),
        typeof(base.coeffs),typeof(base.indices),typeof(base.center)}(
        base.domain, base.coeffs, base.indices, base.center, base.invhalfwidth)
end

"""Create a Float32 Metal Hessian evaluator."""
function _prepare_metal_hessian(res, ::Type{T}) where {T<:AbstractFloat}
    base = _prepare_metal(res, T)
    D = SmolyakPoly.dimension(res)
    return SmolyaxPolyMetalHessianEvaluator{D,T,typeof(base.domain),
        typeof(base.coeffs),typeof(base.indices),typeof(base.center)}(
        base.domain, base.coeffs, base.indices, base.center, base.invhalfwidth)
end

"""Convert and validate a host/device Metal point batch."""
function _metal_batch(evaluator::SmolyaxPolyMetalEvaluator{D,T},
                      X::AbstractMatrix) where {D,T}
    if X isa Metal.MtlArray
        size(X, 2) == D || throw(DimensionMismatch(
            "Metal batch must have $D columns; received size $(size(X))"))
        # A host validation round-trip keeps boundary semantics explicit and is
        # outside the fused prediction kernel. Prepared metadata stays resident.
        host = Array(X)
        SmolyakPoly._validate_points(evaluator.domain, host)
        return T.(X)
    end
    SmolyakPoly._validate_points(evaluator.domain, X)
    return Metal.MtlArray(T.(X))
end


_metal_batch(evaluator::SmolyaxPolyMetalGradientEvaluator{D,T},
             X::AbstractMatrix) where {D,T} =
    _metal_batch(SmolyaxPolyMetalEvaluator{D,T,typeof(evaluator.domain),
        typeof(evaluator.coeffs),typeof(evaluator.indices),typeof(evaluator.center)}(
        evaluator.domain, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth), X)

_metal_batch(evaluator::SmolyaxPolyMetalHessianEvaluator{D,T},
             X::AbstractMatrix) where {D,T} =
    _metal_batch(SmolyaxPolyMetalEvaluator{D,T,typeof(evaluator.domain),
        typeof(evaluator.coeffs),typeof(evaluator.indices),typeof(evaluator.center)}(
        evaluator.domain, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth), X)

"""Evaluate a point batch and leave the output on the Metal device."""
function (evaluator::SmolyaxPolyMetalEvaluator{D,T})(X::AbstractMatrix) where {D,T}
    device = _metal_batch(evaluator, X)
    output = Metal.MtlArray{T}(undef, size(device, 1))
    threads = min(256, max(1, length(output)))
    groups = cld(length(output), threads)
    Metal.@metal threads=threads groups=groups _metal_evaluate_kernel!(
        output, device, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(length(evaluator.coeffs)), Int32(D))
    return output
end

"""Evaluate one point on Metal and return a host scalar."""
function (evaluator::SmolyaxPolyMetalEvaluator{D,T})(x::AbstractVector) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    return Array(evaluator(reshape(T.(collect(x)), 1, D)))[1]
end

"""Construct an explicit basis matrix and leave it on Metal."""
function basis_matrix(evaluator::SmolyaxPolyMetalEvaluator{D,T},
                      X::AbstractMatrix) where {D,T}
    device = _metal_batch(evaluator, X)
    output = Metal.MtlArray{T}(undef, size(device, 1), size(evaluator.indices, 1))
    threads = min(256, max(1, length(output)))
    groups = cld(length(output), threads)
    Metal.@metal threads=threads groups=groups _metal_basis_kernel!(
        output, device, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(D))
    return output
end

"""Construct one basis vector and leave it on Metal."""
function basis_vector(evaluator::SmolyaxPolyMetalEvaluator{D,T}, x) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    return vec(basis_matrix(evaluator, reshape(T.(collect(x)), 1, D)))
end

# ------------------------------------------------------------------------------
# Metal gradient and Hessian calls
# ------------------------------------------------------------------------------

"""Evaluate Metal gradients and return a host `N × D Matrix{Float32}`."""
function (evaluator::SmolyaxPolyMetalGradientEvaluator{D,T})(
        X::AbstractMatrix) where {D,T}
    device = _metal_batch(evaluator, X)
    output = Metal.MtlArray{T}(undef, size(device, 1), D)
    threads = min(256, max(1, length(output)))
    groups = cld(length(output), threads)
    Metal.@metal threads=threads groups=groups _metal_gradient_kernel!(
        output, device, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(length(evaluator.coeffs)), Int32(D))
    return Array(output)
end

"""Evaluate one Metal gradient and return a host `Vector{Float32}`."""
function (evaluator::SmolyaxPolyMetalGradientEvaluator{D,T})(
        x::AbstractVector) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    return vec(evaluator(reshape(T.(collect(x)), 1, D)))
end

"""Evaluate Metal Hessians and return one host matrix per point row."""
function (evaluator::SmolyaxPolyMetalHessianEvaluator{D,T})(
        X::AbstractMatrix) where {D,T}
    device = _metal_batch(evaluator, X)
    output = Metal.MtlArray{T}(undef, size(device, 1), D, D)
    threads = min(256, max(1, length(output)))
    groups = cld(length(output), threads)
    Metal.@metal threads=threads groups=groups _metal_hessian_kernel!(
        output, device, evaluator.coeffs, evaluator.indices, evaluator.center,
        evaluator.invhalfwidth, Int32(length(evaluator.coeffs)), Int32(D))
    host = Array(output)
    return [Matrix(@view host[i, :, :]) for i in axes(host, 1)]
end

"""Evaluate one Metal Hessian and return a host `D × D Matrix{Float32}`."""
function (evaluator::SmolyaxPolyMetalHessianEvaluator{D,T})(
        x::AbstractVector) where {D,T}
    SmolyakPoly._validate_point(evaluator.domain, x)
    return first(evaluator(reshape(T.(collect(x)), 1, D)))
end

SmolyakPoly._register_backend!(:metal, _prepare_metal)
SmolyakPoly._register_gradient_backend!(:metal, _prepare_metal_gradient)
SmolyakPoly._register_hessian_backend!(:metal, _prepare_metal_hessian)

end # module SmolyaxPolyMetalExt
