module OptimizersAlgorithms
include("math_functions.jl")

using .MLMathFunctions

export Gradient_descent
# Gradient Descent function
function Gradient_descent(J::Function,α::Float64,max_it::Int,Θ₀::Vector,δ = -1) # α es tasa de aprendizaje y p la parada del algoritmo
    Θⱼ = Θ₀
    it = 0
    c = length(Θ₀)
    gradient = fill(Inf,c)
    
    while it < max_it && (abs(sum(gradient-zeros(c))))/c > δ
        gradient = MLMathFunctions.∇f(J,Θⱼ)
        Θⱼ = Θⱼ - α*gradient
        it += 1

    end 
    return Θⱼ
end




end 