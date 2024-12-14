
include("../../../Utils/Data_structs/data_structs.jl")
module MyLogisticRegression

using ..Data_Structs
using DataFrames

export MyLogisticRegression
mutable struct MyLogisticRegressionModel
    data::MLData
    parameters::Vector{<:Number}


    function MyLogisticRegressionModel(features::DataFrame, targets::Vector)
        
        #return unique(targets)
        if Set(unique(targets)) != Set([0, 1])
            if length(unique(targets)) != 2
                error("Targets should be binary.")
            end 
            values = unique(targets)
            map_values = Dict( values[1] => 0, values[1] => 1)
            targets = map(x -> map_values[x], targets)
        end

        ml_data = MLData(features, targets)
        new(ml_data)
    end


end 

include("../../../Utils/math_functions.jl")
include("../../../Utils/Optimization_algorithms.jl")

using .MLMathFunctions
using .OptimizersAlgorithms

export LogisticRegressionModel
function LogisticRegressionModel(features::DataFrame, targets::Vector)
    return MyLogisticRegressionModel(features, targets)
end 



hipotesis_function(x_i::Vector{<:Number}, params::Vector{<:Number}) = sigmoide(sum(x_i .* params))

# the cost function is based the log likelihood function
export J

function J(model::MyLogisticRegressionModel, params::Vector{<:Number})
    X = Matrix(model.data.features)
    y = model.data.targets   
    
    # Calcular h (predicciones)
    h = sigmoide.(X * params)
    
    # Asegurarse de que h esté en el rango válido
    h = clamp.(h, eps(Float64), 1 - eps(Float64))

    # Número de ejemplos
    m = size(X, 1)
    
    # Calcular el costo logístico
    cost = -1 / m * (y' * log.(h) + (1 .- y)' * log.(1 .- h))
    
    return cost[1]  # Devolver el costo como un escalar
end

export optim!
function optim!(model; α = 0.00001, max_it = 10000, method = :Gradient_descent)

    if method == :Gradient_descent
        params = zeros(length(model.data.features[1,:]))
        fit_parameters = Gradient_descent(p -> J(model, p), α, max_it, params)
        model.parameters = fit_parameters
    end
end 

export predict
function predict(model::MyLogisticRegressionModel, new_data::DataFrame)
    new_data_proccessing = processing_features!(new_data)
    h = Matrix(new_data_proccessing) * model.parameters
    return sigmoide.(h)
end

export confusion_matrix
function confusion_matrix(model::MyLogisticRegressionModel)
    y_real = Int64.(model.data.targets)
    y_pred = Int64.(predict(model, model.data.features) .>=.5)
    
    # Validación de tamaños
    if length(y_real) != length(y_pred)
        error("Los vectores de entrada deben tener la misma longitud.")
    end
    
    # Inicialización de la matriz de confusión
    cm = zeros(Int, 2, 2)  # Matriz de 2x2
    
    # Rellenar la matriz de confusión
    for (real, pred) in zip(y_real, y_pred)
        cm[real + 1, pred + 1] += 1
    end
    
    # Imprimir la matriz de confusión
    println("Matriz de confusión:")
    println("           Predicción")
    println("            0     1")
    println("Real   0   ", cm[1, 1], "     ", cm[1, 2])
    println("       1   ", cm[2, 1], "     ", cm[2, 2])
    
    return cm
end

export confusion_matrix
function confusion_matrix(y_real::Vector{Int}, y_pred::Vector{Int})
    # Validación de tamaños
    if length(y_real) != length(y_pred)
        error("Los vectores de entrada deben tener la misma longitud.")
    end
    
    # Inicialización de la matriz de confusión
    cm = zeros(Int, 2, 2)  # Matriz de 2x2
    
    # Rellenar la matriz de confusión
    for (real, pred) in zip(y_real, y_pred)
        cm[real + 1, pred + 1] += 1
    end
    
    # Imprimir la matriz de confusión
    println("Matriz de confusión:")
    println("           Predicción")
    println("            0     1")
    println("Real  0   ", cm[1, 1], "     ", cm[1, 2])
    println("       1   ", cm[2, 1], "     ", cm[2, 2])
    
    return cm
end



end 
