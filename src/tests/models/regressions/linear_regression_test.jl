include("../../../../src/models/regressions/linear_regression/Linear_regression.jl")
include("../../../Utils/math_functions.jl")


using .MyLinearRegression
using .Data_Structs

using DataFrames, Random

methods(predict)
optim!

# Generar datos sintéticos
Random.seed!(123)  # Fijar semilla para reproducibilidad
n = 1000000  # Número de observaciones

x1 = randn(n)  # Variable independiente 1 (distribución normal)
x2 = randn(n)  # Variable independiente 2 (distribución normal)
y = 3.5 .+ 2.0 .* x1 .+ -1.5 .* x2 .+ randn(n)  # Relación lineal con ruido

features = DataFrame(
    x_0 = ones(Float64, n),
    x1 = x1,
    x2 = x2,
)

targets = y

methods(LinearRegressionModel)


MLData

data = MLData(features, targets)

model = LinearRegressionModel(data)

model.data.features



@time optim!(model; α = 0.1, max_int = 100)

parameters = model.parameters.parameters

# comparation with GLM
using GLM 

@time model_GLM = lm(@formula(y ~ x1 + x2), DataFrame(
    y = y,
    x1 = x1,
    x2 = x2,
))


