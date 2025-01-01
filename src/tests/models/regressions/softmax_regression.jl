include("../../../models/regressions/softmax_regression/softmax_regression.jl")

using .MySoftMaxRegression
using CairoMakie
using Random
using DataFrames

# Ejemplo de matrices one-hot


# generate artificial data
begin 
    # Función para generar datos de una clase
    function generate_class_data(center::Tuple{Float64, Float64}, num_points::Int, spread::Float64)
        x1 = center[1] .+ spread .* randn(num_points)
        x2 = center[2] .+ spread .* randn(num_points)
        return hcat(x1, x2)
    end
    
    # Generar datos para 4 clases
    num_points_per_class = 50
    spread = 0.5
    
    
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

features
targets = string.(labels)

# Graficar los datos
begin
fig = Figure()
axis = Axis(fig[1, 1])
map_colors = Dict(["1" => :red, "2" => :green, "3" => :blue, "4" => :magenta])
colors = [map_colors[i] for i in labels]
scatter!(axis, features.x1, features.x2, color = colors)
fig
end

using .Data_Structs

features 
targets

data = MLData(features, targets)

model = SoftMaxRegressionModel(data)

optim!(model)

@time predict(model, model.data.features)

cm, categories = confusion_matrix(model)

cm

