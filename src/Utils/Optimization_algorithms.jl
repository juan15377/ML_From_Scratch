module OptimizersAlgorithms
include("math_functions.jl")

using .MLMathFunctions

export Gradient_descent_ml
# Gradient Descent function
function gradient_descent_ml(
    J::Function,           # Función de costo
    Θ₀::Vector{Float64},   # Parámetros iniciales
    grad_J::Function;    # Gradiente de la función de costo
    α::Float64 = 0.01,     # Tasa de aprendizaje inicial
    max_it::Int = 1000,    # Máximo número de iteraciones
    δ::Float64 = 1e-6,     # Tolerancia para el gradiente
    regularization::Float64 = 0.0  # Parámetro de regularización (L2)
    )

    Θ = copy(Θ₀)
    v = zeros(length(Θ))  # Para tasas de aprendizaje adaptativas
    β1 = 0.9  # Momento (Adam)
    β2 = 0.999  # Escala adaptativa (Adam)
    ε = 1e-8  # Para evitar divisiones por cero
    m = zeros(length(Θ))
    it = 0
    
    for it in 1:max_it
        # Calcular gradiente con regularización L2
        grad = grad_J(Θ) .+ regularization .* Θ
        
        # Algoritmo Adam (mezcla de momento y RMSprop)
        m = β1 .* m .+ (1 .- β1) .* grad
        v = β2 .* v .+ (1 .- β2) .* (grad .^ 2)
        m_hat = m ./ (1 .- β1^it)
        v_hat = v ./ (1 .- β2^it)
        
        # Actualización de parámetros
        Θ_new = Θ - α .* m_hat ./ (sqrt.(v_hat) .+ ε)
        
        # Condición de parada basada en el gradiente
        if norm(grad) < δ
            println("Convergencia alcanzada en la iteración $it.")
            break
        end
        
        Θ = Θ_new
    end
    
    return Θ
end

export Gradient_descent
function Gradient_descent(∇f::Function,α::Float64,max_it::Int,Θ₀::Vector,δ = -1) # α es tasa de aprendizaje y p la parada del algoritmo
    Θⱼ = Θ₀
    it = 0
    c = length(Θ₀)
    gradient = fill(Inf,c)
    
    while it < max_it && (abs(sum(gradient-zeros(c))))/c > δ
        gradient = ∇f(Θⱼ)
        Θⱼ = Θⱼ - α*gradient
        it += 1
    end 
    return Θⱼ
end


end 