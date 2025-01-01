include("../../../Utils/Data_structs/data_structs.jl")

module MySoftMaxRegression

using ..Data_Structs
using DataFrames

export SoftMaxRegressionModel, predict, confusion_matrix, optim!
mutable struct MySoftMaxRegressionModel
    data::MLData
    parameters::Matrix{Float64}
    # data.targets is a matrixOneHot
    function MySoftMaxRegressionModel(data::MLData)
        parameters = zeros(size(data.features, 2), 1)
        new(data, parameters)
    end 
end 

export SoftMaxRegressionModel
function SoftMaxRegressionModel(data::MLData)
    @assert typeof(data.targets) == MatrixOneHot "type of targets must be MatrixOneHot"
    m, n = size(data.features)
    k, n = size(data.targets.matrix)
    return MySoftMaxRegressionModel(data)
end
# si tengo una matriz de caracteristicas de la siguiente manera 
# 
#  X in M(mxn)
# X = [x11, x12 x13 ..... x1m
#     x21, x22 x23 ..... x2m
#    ...
#     xn1, xn2 xn3 ..... xnm]
#
#  y in M(mXK)
# matriz de targets en matrix One Hot 
# y = [0 0 0 1 0 0 0 ;
#     0 0 1 0 0 0 0 ;
#     0 1 0 0 0 0 0 ;
#     ...
#     0 0 0 0 1 0 0] -- K class and m registers 
#
# matrix de parametros 
# parameters in M(kxn)
# parameters = [w11, w12 w13 ..... w1n ;
#              w21, w22 w23 ..... w2n ;
#             ... 
#              wk1, wk2 wn3 ..... wkn]
#
# el objectivo es poder calcular las probabilidades de cada una de las 
# clases dado un conjunto de parametros 
# 
# X is 
#export probablity_class
function probablity_class(W::Matrix{Float64}, X::Matrix{<:Number})
    z_matrix = X * W
    z_matrix = exp.(z_matrix)
    probability_matrix = z_matrix ./ sum(z_matrix, dims = 2)
    return probability_matrix
end 


#export cross_entropy
function cross_entropy(model::MySoftMaxRegressionModel, W::Matrix{Float64})
    X = Matrix(model.data.features)
    targets = model.data.targets.matrix
    m = length(X[:,1])
    predicted_probability_class = probablity_class(W, X)
    log_predicted_probability_class = log.(predicted_probability_class)

    return - 1/m * sum(sum(log_predicted_probability_class .* targets, dims = 2))

end

#export gradient_softmaxregression
function gradient_softmaxregression(model::MySoftMaxRegressionModel, W::Matrix{Float64})
    X = Matrix(model.data.features)
    Y = model.data.targets.matrix # is matrixOneHot
    Y_predicted = probablity_class(W, X)
    m, n = size(X)
    return 1/m * X' * (Y_predicted - Y)

end     

#export optim!
function optim!(model::MySoftMaxRegressionModel)
    m,k = size(model.data.targets.matrix)
    m,n = size(model.data.features)
    learning_rate = 0.01
    max_iterations = 10000
    initial_params = zeros(Float64, n, k)
    W = initial_params

    for iteration in 1:max_iterations
        gradient = gradient_softmaxregression(model, W)
        W -= learning_rate * gradient
    end

    model.parameters = W

end


function predict(model::MySoftMaxRegressionModel, new_data::DataFrames.DataFrame)
    # Procesar los datos nuevos
    new_data_processed = processing_features!(new_data)
    W = model.parameters  # Matriz de pesos del modelo
    X = Matrix(new_data_processed)
    classes = model.data.targets.values  # Vector de nombres de las clases

    # Calcular las probabilidades predichas
    predicted_probability_class = probablity_class(W, X)  # m × k matriz de probabilidades

    # Identificar el índice de la clase con mayor probabilidad para cada muestra

    f(bool, str) = bool ? str : ""

    predicted_class_indices = map(row -> row .== maximum(row), eachrow(predicted_probability_class))

    predicted_class = map( row -> reduce( * , map(f, row, classes)), predicted_class_indices)
    
    return predicted_class
end

function confusion_matrix(model::MySoftMaxRegressionModel)
    
    categories = model.data.targets.values

    true_matrix_one_hot = model.data.targets.matrix

    f(row) = [categories[index] for (element, index) in zip(row, eachindex(row))  if element == true | element == 1][1]

    true_categories = map(row -> f(row), eachrow(true_matrix_one_hot))

    predict_categories = predict(model, model.data.features)

    num_class = length(categories)

    cm = fill(0, num_class, num_class)


    row = 1
    column = 1
    for class_true in categories
        column = 1
        for class_predict in categories
            num = (true_categories .== class_true) .& (predict_categories .== class_predict)
            cm[row, column] = sum(num) # sum(num) is the value in row, an column in confussion matrix
            column += 1 
        end 
        row += 1
    end 

    return cm, categories
end



end
