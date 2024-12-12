
module Data_Structs
using DataFrames
using CategoricalArrays
using SparseArrays
using StatsBase
export MLData

# Función que procesa las características (features) de un DataFrame
export processing_features!
function processing_features!(df::DataFrame)
    new_columns = DataFrame()  # Crear un DataFrame temporal para las nuevas columnas
    columns_delete = []  # Lista de columnas a eliminar
    
    for (col, name_col) in zip(eachcol(df), names(df))
        if eltype(col) <: Number
            continue
        else
            push!(columns_delete, name_col)
            # Crear nuevas columnas para las categorías
            categorical_values = unique(col)
            for categorical_value in categorical_values
                # Crear el nombre de la nueva columna
                name_new_col = Symbol(name_col, "_", categorical_value)
                # Mapear 1 y 0 para la columna categórica
                new_columns[!, name_new_col] = Int.(col .== categorical_value)
            end
        end
    end
    
    select!(df, Not(columns_delete))  # Eliminar las columnas originales
    df = hcat(df, new_columns)  # Combinar el DataFrame original con las nuevas columnas
    return df
end


# Estructura para almacenar la matriz One-Hot y sus valores
struct MatrixOneHot
    matrix::SparseMatrixCSC{Int64, Int64} 
    values::Vector{String}

    function MatrixOneHot(values::Vector{String})
        unique_values = unique(values)
        value_to_col = Dict(value => i for (i, value) in enumerate(unique_values))
        rows = 1:length(values)
        cols = [value_to_col[value] for value in values]
        matrix = sparse(rows, cols, ones(Int64, length(rows)))
        return new(matrix, unique_values)
    end
end

# Función para procesar los targets numéricos
function proccessing_targets(targets::Vector{<: Number})
    return targets
end

# Función para procesar los targets categóricos
proccessing_targets(targets::Vector{String}) = MatrixOneHot(targets)

# Estructura de datos para almacenamiento de características y objetivos de un modelo ML
struct MLData
    features_names::Vector{String}
    features::DataFrame
    targets::Union{MatrixOneHot, Vector{<: Number}}
    test_features::Union{DataFrame, Nothing}
    test_target::Union{Vector{Union{Number, AbstractString}}, Nothing}

    # Constructor para MLData con características y objetivos de entrenamiento y prueba
    function MLData(features::DataFrame, 
                    targets::AbstractVector{<:Union{Number, AbstractString}}, 
                    test_features::Union{DataFrame, Nothing}, 
                    test_target::Union{AbstractVector{<:Union{Number, AbstractString}}, Nothing})
        features =  processing_features!(features)  # Procesar características
        targets = proccessing_targets(targets)  # Procesar objetivos
        features_names = names(features)  # Obtener los nombres de las características
        return new(features_names, features, targets, test_features, test_target)
    end

    # Constructor para MLData con características y objetivos sin separación de prueba
    function MLData(features::DataFrame,
                    targets::AbstractVector{<:Union{Number, String}}) 
        return MLData(features, targets, features, targets)
    end

    # Constructor para MLData con separación de prueba según un porcentaje dado
    function MLData(features::DataFrame,
                    targets::AbstractVector{<:Union{Number, String}};
                    porcent_testing::Float64 = 1.0)
        m = length(targets)
        num_testing = Int(round(porcent_testing * m))  # Calcular número de datos de prueba
        indices_testing = sample(collect(1:m), num_testing)  # Seleccionar índices de prueba
        test_features = features[indices_testing,:]
        test_target = targets[indices_testing]
        return MLData(features, targets, test_features, test_target)
    end

end

end

