module OptimizerLinearRegression

include("../../../regressions/linear_regression/Linear_regression.jl")

using CairoMakie
using ..MyLinearRegression
export optim!
using ..ParametersRegression

# Gradient calculation function
@inline function ∇f(f::Function, p::Vector{<:Number})::Vector{Float64}
    n = length(p)
    ϵ = 1e-6  # Ajustar ϵ para un mejor compromiso entre precisión y estabilidad
    ∇ = zeros(Float64, n)
    
    for j in 1:n
        p_plus = copy(p)
        p_minus = copy(p)
        p_plus[j] += ϵ
        p_minus[j] -= ϵ
        ∇[j] = (f(p_plus...) - f(p_minus...)) / (2 * ϵ)
    end
    
    return ∇
end


# vamos a definir la funcion de costo para un modelo de regression 
export J
function J(model::MyLinearRegressionModel, paremeters::ParametersRegression)

    X_matriz = Matrix(model.data.features)
    cost = (X_matriz * paremeters.parameters - model.data.features).^2
    return cost

end 



# Gradient Descent function
function Gradient_descent(J::Function,α::Float64,max_it::Int,Θ₀::Vector,δ = -1) # α es tasa de aprendizaje y p la parada del algoritmo
    Θⱼ = Θ₀
    it = 0
    c = length(Θ₀)
    gradient = fill(Inf,c)
    
    while it < max_it && (abs(sum(gradient-zeros(c))))/c > δ
        gradient = ∇f(J,Θⱼ)
        Θⱼ = Θⱼ - α*gradient
        it += 1

    end 
    return Θⱼ
end



function stochastic_gradient_descent(J::Function, α::Float64, max_it::Int, Θ₀::Vector{Float64}, dataset, δ::Float64 = -1.0)::Vector{Float64}
    Θ = copy(Θ₀)
    (f, c) = size(dataset.x)
    it = 0
    
    while it < max_it && (norm(∇f(Θ -> J(Θ, dataset), Θ)) / c > δ)
        n = rand(1:f)
        X = reshape(dataset.x[n, :], 1, c)
        Y = dataset.y[n]
        data = Dataset(X, [Y])
        gradient = ∇f(Θ -> J(Θ, data), Θ)
        Θ -= α * gradient
        it += 1
    end
    return Θ
end


CairoMakie.activate!()
function stochastic_vs_gradient_descence(dataset,J,num_it)

    Θ_adj = stochastic_gradient_descent(J,.05,1000,Float64[0,0],dataset)

    Θ_stochatic = Float64[-100,170]
    Θ_standar = Float64[-100,170]

    points_stochatic = []
    points_standar = []
    for i in 1:num_it
        Θ_stochatic = stochastic_gradient_descent(J,.1,1,Θ_stochatic,dataset)
        push!(points_stochatic,Θ_stochatic)

        Θ_standar =  Gradient_descent(Θ->J(Θ,dataset),.0000001,1,Θ_standar)
        push!(points_standar,Θ_standar)
    end 

    print(Θ_standar)

    θ₀,θ₁ = Θ_adj

    θ₀_Range = range(θ₀ - 100, θ₀ + 100, length=100)
    
    θ₁_Range = range(θ₁ - 100, θ₁ + 100, length=100)
    
    z = [J([x,y],dataset) for x in θ₀_Range , y in θ₁_Range]
    
    fig = Figure(size=(800, 400))
    ax1 = Axis(fig[1, 1], aspect=1, xlabel="θ₀", ylabel="θ₁")
    ax2 = Axis(fig[1, 2], aspect=1, xlabel="θ₀", ylabel="θ₁")

    contour!(ax1,θ₀_Range, θ₁_Range, z; colormap=:plasma, levels=50, linewidth=1.5)
    contour!(ax2,θ₀_Range, θ₁_Range, z; colormap=:plasma, levels=50, linewidth=1.5)

    plot!(ax1,[Float64(θ₀)],[Float64(θ₁)])
    plot!(ax2,[Float64(θ₀)],[Float64(θ₁)])

    #return points_standar

    lines!(ax1,(x->x[1]).(points_standar),(x -> x[2]).(points_standar),color = :green)
    lines!(ax2,(x->x[1]).(points_stochatic),(x -> x[2]).(points_stochatic),color = :green)


    return fig

end 

# ! Grafica de ajuste 

CairoMakie.activate!()
function grafica_regresion_lineal_simple(dataset,J)
    fig = Figure()
    ax = Axis(fig[1,1])
    plot!(ax,dataset.x[:,2],dataset.y)

    Θⱼ = Float64[1,5]
    for i in 1:20
        Θⱼ = stochastic_gradient_descent(J,.05,30,Θⱼ,dataset,0.001)
        # graficamos el hiperplano
        lines!(ax,dataset.x[:,2],dataset.x * Θⱼ,color = (:red,.3),linewidth = 2)
    end 

    fig
end 



#
#f() = grafica_regresion_lineal_simple(dataset_1,J)
#
#with_theme(f,theme_dark())
#


function grafica_J(J,dataset)

    optimal_Θ = Gradient_descent(Θ -> J(Θ,dataset),0.00001,1000,Float64[0, 0])

    GLMakie.activate!()
    fig = Figure()
    ax = Axis(fig[1,1])

    X = Y =LinRange(-100 + optimal_Θ[1],optimal_Θ[2] + 100,100)
    Z = [J([x,y],dataset) for x in X, y in Y]

    surface(X,Y,Z/1000000)

end 


function optim!(model::MyLinearRegressionModel)

end 


end




include("../../../regressions/linear_regression/Linear_regression.jl")

using DataFrames

using .LinearRegression

using .OptimizerLinearRegression.OptimizerLinearRegression

using .Data_Structs

features = DataFrame(
    x1 = [1, 2, 3, 4, 5],
    x2 = ["A", "B", "B", "B", "A"],
)

targets = [1, 2, 3, 4, 5]

print(typeof(data))


data = MLData(features, targets)

MyLinearRegressionModel

MyLinearRegressionModel

modelo = LinearRegressionModel(data)

# Definir f para que acepte un vector de parámetros
g(p) = p[1] + p[2]# Suma de todos los elementos del vector

# Llamar a la función de derivada numérica
∇f(g, [1.0, 2.0, 3.0])  # Usa números de punto flotante (Float64)


using DataFrames, LinearAlgebra

# Crear un DataFrame de ejemplo
df = DataFrame(a = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

# Vector con el que se hará el producto punto
v = [0.1, 0.2, 0.3]

# Convertir el DataFrame a una matriz
matrix_df = Matrix(df)

# Calcular el producto punto para cada fila
result = matrix_df * v

# Mostrar el resultado
println(result)

using CairoMakie