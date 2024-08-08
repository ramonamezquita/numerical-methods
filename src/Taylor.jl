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
- `fname::string`: Name of the function to approximate.
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


function parse_commandline()
	s = ArgParseSettings(description = "Plots Taylor approximation.")

	@add_arg_table s begin
		"--f"
		help = "Name of the function to approximate."
		arg_type = String
		required = true
		"--order"
		help = "Order of the Taylor polynomial."
		arg_type = Int
		default = 3
		"--center"
		help = "Taylor polynomial is centered around this value."
		arg_type = Float32
		default = 0
		"--plotrange"
		arg_type = Int
		default = 10
	end

	return parse_args(s)
end



function main()
	parsed_args = parse_commandline()

	# Retrieve user args.
	f = parsed_args["f"]
	order = parsed_args["order"]
	center = parsed_args["center"]

	# Taylor polynomial.
	fn = Differentiables.getinstance(parsed_args["f"]).fn
	taylorpoly = make_taylorpoly(f, order, center)

	# Plot fn vs approximation.
	plotrange = parsed_args["plotrange"]
	x = range(center - plotrange, center + plotrange, length = 100)
	y1 = fn.(x)
	y2 = taylorpoly.(x)
	plt = plot(x, [y1 y2], title = "Taylor approximation for $(f) with N=$(order)", label = ["Original" "Taylor"], linewidth = 2)

	display(plt)
	readline()

end


main()
