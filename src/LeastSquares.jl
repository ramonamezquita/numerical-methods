using Random

import Base.@kwdef

abstract type Optim end

@kwdef struct SGD <: Optim
	lr::Float64 = 0.001
	shuffle::Bool = true
end


@kwdef struct MiniBatchGD <: Optim
	lr::Float64 = 0.001
	bathsize::Integer = 32
end


@kwdef struct BatchGD <: Optim
	lr::Float64 = 0.001
end



@doc raw"""
	MSE(θ::Vector{Float64}; X::Matrix{Float64}, y::Vector{Float64})::Float64

Mean Squared Error function for a linear model. It is defined as:

```math
L(w) = \frac{1}{2N} \sum_{i=1}^N (y_n - x_n^Tw)^2 = \frac{1}{2N} e^T e
```

"""
function MSE(θ::Vector{Float64}; X::Matrix{Float64}, y::Vector{Float64})::Float64
	n = size(X, 1)
	ŷ = X * θ
	e = y - ŷ
	(1 / 2 * n) * (transpose(e) * e)
end


@doc raw"""
	∇MSE(θ::Vector{Float64}; X::Matrix{Float64}, y::Vector{Float64})::Float64

Gradient of Mean Squared Error function for a linear model. It is defined as:

```math
∇L(w) = - \frac{1}{N} X^T e
```

"""
function ∇MSE(θ::Vector{Float64}; X::Matrix{Float64}, y::Vector{Float64})::Vector{Float64}
	n = size(X, 1)
	ŷ = X * θ
	e = y - ŷ
	(-1 / n) * transpose(X) * e
end



"""
	iter(optim::Optim, X::Matrix{Float64})

`iter` methods provide different ways of iterating over X and y.
"""
function iter(optim::Optim, X::Matrix{Float64}, y::Vector{Float64})
	# ... [implementation sold separately] ...
end


"""
	iter(optim::SGD, X::Matrix{Float64})

Iterates over X on row at a time.
"""
function iter(optim::SGD, X::Matrix{Float64}, y::Vector{Float64})
	if optim.shuffle
		X = X[shuffle(1:end), :]
	end

	n = size(X, 1)
	[X[[i], :] for i ∈ 1:n]
end


"""
	iter(optim::BatchGD, X::Matrix{Float64})

Iterates over X in a single pass.
"""
function iter(optim::BatchGD, X::Matrix{Float64}, y::Vector{Float64})
	[(X, y)]
end



"""
	step(θ::Vector{Float64}, optim::Optim)::Vector{Float64}

Update θ using gradient descent.
"""
function step(
	θ::Vector{Float64}
	;
	optim::Optim,
	X::Matrix{Float64},
	y::Vector{Float64},
)::Vector{Float64}

	for Xy in iter(optim, X, y)
		X, y = Xy
		grad = ∇MSE(θ; X = X, y = y)
		θ = θ - optim.lr * grad

	end

	return θ

end


"""
Optimizes `loss` functon using the specified `optim`.
"""
function leastsquares(
	X,
	y
	;
	optim::Optim = SGD(),
	θ::Union{Vector{Float64}, Nothing} = nothing,
	maxiter::Integer = 100,
	gtol::Float64 = 0.001,
)
	if θ === nothing
		θ = zeros(size(X, 2))
	end

	nit = 1

	while nit <= maxiter

		θ = step(θ, optim = optim, X = X, y = y)

		nit += 1
	end
	return θ


end


