#
include("../../../Utils/Parameters/regression/parameters_regression.jl")
include("../../../model_type.jl")
include("../../../Utils/Data_structs/data_structs.jl")

module MyLinearRegression

using ..TypeModel
using ..Data_Structs  # Importa MLData desde Data_Structs
using ..ParameterRegression
using DataFrames

mutable struct MyLinearRegressionModel <: MyRegressionModel
    data::Data_Structs.MLData
    parameters::Union{ParametersRegression, Nothing}

    # Constructor interno
    function MyLinearRegressionModel(ml_data::Data_Structs.MLData)
        new(ml_data, nothing)
    end
end

export LinearRegressionModel
# Constructor externo para crear un modelo
function LinearRegressionModel(ml_data::MLData)
    model = MyLinearRegressionModel(ml_data)
    return model
end

export J
# Función de costo (J) para el modelo de regresión lineal
# Recibe un modelo y parámetros, devuelve el costo
function J(model::MyLinearRegressionModel, paremeters::ParametersRegression)

    X_matriz = Matrix(model.data.features)
    cost = sum((X_matriz * paremeters.parameters - model.data.targets).^2)
    return cost

end 


function gradient(Θ::Vector, model::MyLinearRegressionModel)
    m, n = size(model.data.features)
    X = Matrix(model.data.features)
    ∇f = 1/m * X' * (X * Θ  - model.data.targets)
    return ∇f
end 

include("../../../Utils/math_functions.jl")
include("../../../Utils/Optimization_algorithms.jl")

using .MLMathFunctions
using .OptimizersAlgorithms

export optim!
function optim!(model::MyLinearRegressionModel; method=:Gradient_descent, α=0.0001, max_int = 10000)
    if method == :Gradient_descent
        num_params = length(model.data.features[1,:])
        initial_params = ones(Float64, num_params)
        fit_parameters = Gradient_descent(p -> gradient(p, model), α, max_int, initial_params)
        model.parameters = ParametersRegression(fit_parameters)
    end 
end 


export predict
function predict(model::MyLinearRegressionModel, new_data::DataFrame)
    if model.parameters === nothing
        error("model must be fitted")
    end 

    data_preccesing = proccessing_features!(new_data)
    return Matrix(data_preccesing) * model.parameters.parameters
end

end

