using LinearAlgebra
using ArgParse
using Plots



"""
Represents a nth-differentiable function.

# Fields
- `fn::Function`: The actual function. Singature: `fn(x::Number)::Number`.

- `diffGen::Function`: Derivative generator. Singature: `diffGen(n::Integer)::Function`.
	Returns the nth-derivative of the original function `fn`. The returned function `g` 
	should output the nth-derivative at `x` and has signature `g(x::Number)::Number`.
"""
struct Differentiable
	fn::Function
	diffGen::Function
end


"""
Collection of `Differentiable` instances.

# Exports
- getinstance
"""
module Differentiables

import ..Differentiable

export getinstance

function cosine_diff_gen(n::Integer)::Function

	function g(x::Number)::Number
		cos(x + n * (pi / 2))
	end

end

cosine = Differentiable(cos, cosine_diff_gen)


function getinstance(name::String)::Union{Differentiable, Nothing}
	d = Dict("cosine" => Differentiables.cosine)
	get(d, name, nothing)
end

end


"""
	make_taylorpoly(diff::Differentiable, N::Integer, a::Number)::Function

Return the Nth Taylor polynomial around `a`.

# Arguments
- `fname::string`: Name of the function to approximate. Available: {"cosine"}
- `N::Integer`: Order of the Taylor polynomial.
- `a::Number`: Taylor polynomial is centered around this value.
"""
function make_taylorpoly(fname::String, N::Integer, a::Number)::Function

	diff = Differentiables.getinstance(fname)

	"""
		ncoeff(n::Integer)::Number

	Return the nth Taylor coefficient.
	"""
	function ncoeff(n::Integer)::Number
		if n == 0
			return diff.fn(a)
		end

		diff.diffGen(n)(a) / factorial(n)
	end

	coefficients = [ncoeff(n) for n ∈ 0:N]

	"""
		taylorpoly(x::Number)::Number

	Taylor polynomial of order `n` centered around `a`.
	"""
	function taylorpoly(x::Number)::Number
		powers = [(x - a)^n for n ∈ 0:N]
		dot(coefficients, powers)
	end

	return taylorpoly

end
