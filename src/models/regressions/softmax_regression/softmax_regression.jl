include("../../../Utils/Data_structs/data_structs.jl")


using .Data_Structs

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
    return predicted_class_indices

    return predicted_one_hot
end



# generate artificial data
begin 
    using Random
    using DataFrames
    using CairoMakie
    
    # Función para generar datos de una clase
    function generate_class_data(center::Tuple{Float64, Float64}, num_points::Int, spread::Float64)
        x1 = center[1] .+ spread .* randn(num_points)
        x2 = center[2] .+ spread .* randn(num_points)
        return hcat(x1, x2)
    end
    
    # Generar datos para 4 clases
    num_points_per_class = 50
    spread = 0.5
    
    Random.seed!(42)  # Para reproducibilidad
    
    class1 = generate_class_data((2.0, 2.0), num_points_per_class, spread)
    class2 = generate_class_data((-2.0, 2.0), num_points_per_class, spread)
    class3 = generate_class_data((-2.0, -2.0), num_points_per_class, spread)
    class4 = generate_class_data((2.0, -2.0), num_points_per_class, spread)
    
    # Etiquetas para cada clase
    labels = vcat(
        fill(1, num_points_per_class),
        fill(2, num_points_per_class),
        fill(3, num_points_per_class),
        fill(4, num_points_per_class)
    )
    
    # Combinar datos y etiquetas en un DataFrame
    data = vcat(class1, class2, class3, class4)
    features = DataFrame(one = ones(Float64, length(data[:,1])), x1=data[:, 1], x2=data[:, 2])
    y = labels
    # Visualizar los datos

end     



string(:blue)



features
labels = string.(labels)

# Graficar los datos
begin
fig = Figure()
axis = Axis(fig[1, 1])
map_colors = Dict(["1" => :red, "2" => :green, "3" => :blue, "4" => :magenta])
colors = [map_colors[i] for i in labels]
scatter!(axis, features.x1, features.x2, color = colors)
fig
end


using .MySoftMaxRegression
using .Data_Structs

data = MLData(features, labels)

model.data.targets.values


predict(model, model.data.features)

result = map(row -> row .== maximum(row), eachrow(matrix_predicciones))

data.targets

model = SoftMaxRegressionModel(data)

W
gradient_softmaxregression(model, W)

sum(labels .== predict(model, model.data.features) )


W = Matrix([
    0 0 0 ;
    0.4 0.5 0.6 ;
    0.7 0.8 0.9 ;
    0.2 0.3 0.4 ;
]')



cross_entropy(model, W)



W
optim!(model)
gradient_softmaxregression(model, model.parameters)



data.features

new_data = [1 1 1;]

X = Matrix(model.data.features)

Y_predicted = probablity_class(W, X)

Y = model.data.targets.matrix

Y_predicted  -  Y

X'
Y

3 * 4
@time cross_entropy(model, W)

using .MySoftMaxRegression


  # Crear una matriz de ejemplo
matriz = [2 4 6; 8 10 12; 14 16 18]

# Crear un vector para la división
vector = [2, 4, 6]

# Dividir cada fila de la matriz por el elemento correspondiente del vector
resultado = matriz ./ vector_suma_filas

println(resultado)


# Crear una matriz de ejemplo
matriz = [1 2 3; 4 5 6; 7 8 9]

# Sumar los valores de cada fila
vector_suma_filas = sum(matriz, dims=2)

# Convertir el resultado en un vector unidimensional
vector_suma_filas = vec(vector_suma_filas)

println(vector_suma_filas)



A = rand(1000, 1000)  # Matriz 3x4

max_row

# Usando map
@time result = map(row -> row .== maximum(row), eachrow(A))