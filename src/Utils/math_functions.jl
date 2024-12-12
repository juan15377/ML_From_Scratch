
module MLMathFunctions

export ∇f 
@inline function ∇f(f::Function, p::Vector{<:Number})::Vector{Float64}
    n = length(p)
    ϵ = 1e-6  # Ajustar ϵ para un mejor compromiso entre precisión y estabilidad
    ∇ = zeros(Float64, n)
    
    for j in 1:n
        p_plus = copy(p)
        p_minus = copy(p)
        p_plus[j] += ϵ
        p_minus[j] -= ϵ
        ∇[j] = (f(p_plus) - f(p_minus)) / (2 * ϵ)
    end
    
    return ∇
end


export sigmoide
function sigmoide(x)
    return 1 / (1 + exp(-x))
end

end 