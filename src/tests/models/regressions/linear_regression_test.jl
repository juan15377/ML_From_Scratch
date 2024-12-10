include("../../../../src/models/regressions/linear_regression/Linear_regression.jl")
include("../../../Utils/math_functions.jl")


using .MyLinearRegression
using .Data_Structs

using DataFrames, Random

# Generar datos sintéticos
Random.seed!(123)  # Fijar semilla para reproducibilidad
n = 10000  # Número de observaciones

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

optim!(model; α = 0.000001, max_int = 10000)

parameters = model.parameters.parameters

# comparation with GLM
using GLM 

model_GLM = lm(@formula(y ~ x1 + x2), DataFrame(
    y = y,
    x1 = x1,
    x2 = x2,
))
