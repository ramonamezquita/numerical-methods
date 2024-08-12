using LinearAlgebra

"""
	steepestdescent(
	f::Function,
	g::Function,
	x0::Vector{Float64},
	maxiter::Integer = 100,
	lr::Float64 = 0.01,
	gtol::Float64 = 0.001,
	printevery::Integer = 50,
)

Perform steepest descent algorithm.

Given a continuously diffentiable (loss) function `f : Rn → R`, 
steepest descent is an iterative procedure to find a local minimum 
of `f` by moving in the opposite direction of the gradient `g` at 
every iteration `k`.

"""
function steepestdescent(
	f::Function,
	g::Function,
	x0::Vector{Float64};
	maxiter::Integer = 100,
	lr::Float64 = 0.01,
	gtol::Float64 = 0.001,
	printevery::Integer = 50,
)
	x = x0
	nit = 1
	gradient = g(x)

	while nit <= maxiter && norm(gradient) > gtol

		if nit % printevery == 0
			println("[Step $(nit)] loss = $(f(x))")
		end

		x = x - lr * gradient
		gradient = g(x)
		nit += 1

	end

	return x

end
