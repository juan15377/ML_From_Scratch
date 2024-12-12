include("../../../../src/models/regressions/logistic_regression/logistic_regression.jl")
include("../../../Utils/math_functions.jl")




using Revise
using .MyLogisticRegression
using .Data_Structs


## ? Ejemplo sencillo de la regression logistica 


using DataFrames, CairoMakie

# Generar datos para dos cúmulos
cluster1_x1 = randn(50) .+ 1.0   # Primer cúmulo, desplazado por 1
cluster1_x2 = randn(50) .+ 1.0   # Primer cúmulo, desplazado por 1

cluster2_x1 = randn(50) .+ 5.0   # Segundo cúmulo, desplazado por 5
cluster2_x2 = randn(50) .+ 5.0   # Segundo cúmulo, desplazado por 5

# Etiquetas para los dos cúmulos (0 para el primero, 1 para el segundo)
y1 = zeros(50)   # Cúmulo 1 -> 0
y2 = ones(50)    # Cúmulo 2 -> 1

# Crear DataFrame con las características y etiquetas
data = DataFrame(
    c = ones(100),
    x1 = vcat(cluster1_x1, cluster2_x1),
    x2 = vcat(cluster1_x2, cluster2_x2),
)

y = vcat(y1, y2)

colors = map(x -> x == 1 ? :red : :blue, y)

model = LogisticRegressionModel(data, y)

optim!(model, α = 0.0001, max_it = 10000)

f(x) = (-model.parameters[1] - model.parameters[2]*x)/model.parameters[3]


fig = Figure()
axis = Axis(fig[1,1])
plot!(axis, data.x1, data.x2, color = colors)
lines!(axis, data.x1, f.(data.x1))
fig
# Mostrar los datos
println(data)




1 .- [1, 2, 3]

include("../../../Utils/Optimization_algorithms.jl")

using .MLMathFunctions

using DataFrames
X = DataFrame(
    constante = ones(Float64, 5),
    feature1 = [2.0, 1.0, 3.5, 2.0, 1.5],
    feature2 = [3.0, 2.0, 0.5, 1.0, 2.5],
)

# Vector de etiquetas (0 y 1)
y = [1, 0, 1, 0, 1]



using CSV


data =  CSV.read("src/datasets/german.csv", DataFrame)

X = copy(data)
X[:, 1] = ones(Int64, length(data[:, 1]))

y = data[:,1]

model = LogisticRegressionModel(X, y)


optim!(model; α = 5.0, max_it = 10000)


model.parameters


confusion_matrix(model)


∇f(p -> J(model, p), model.parameters)
∇f(p -> J(model, p),  coef(model_GLM))

using GLM


using DataFrames, GLM

# Definir la fórmula correctamente
form = @formula(Creditability ~ Account_Balance + Duration_of_Credit_monthly +
    Payment_Status_of_Previous_Credit + Purpose + Credit_Amount +
    Value_Savings_Stocks + Length_of_current_employment + Instalment_per_cent +
    Sex_Marital_Status + Guarantors + Duration_in_Current_address +
    Most_valuable_available_asset + Age_years + Concurrent_Credits +
    Type_of_apartment + No_of_Credits_at_this_Bank + Occupation +
    No_of_dependents + Telephone + Foreign_Worker)

# Ajustar el modelo logístico
model_GLM = glm(form, data, Binomial(), LogitLink())

# Ver los resultados
println(coef(model))  # Coeficientes del modelo

predict

confusion_matrix(data.Creditability, Int64.(GLM.predict(model_GLM, data) .>=.5))