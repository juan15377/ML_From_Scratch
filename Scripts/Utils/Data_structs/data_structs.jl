module Data_Structs
using DataFrames
using CategoricalArrays
using SparseArrays
using StatsBase

export proccessing_features!

function proccessing_features(df::DataFrame)
    new_columns = DataFrame()  # Crear un DataFrame temporal para las nuevas columnas
    columns_delete = []
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
                new_columns[!, name_new_col] = map(x -> x == categorical_value ? 1 : 0, col)
            end
        end
    end
    select!(df, Not(columns_delete))
    # Combinar el DataFrame original con las nuevas columnas
    df = hcat(df, new_columns, copycols=false)
    return df

end
struct MatrixOneHot
    matrix::SparseMatrixCSC{Int64, Int64} 
    values::Vector{String}


    function MatrixOneHot(values::Vector{String})
        unique_values = unique(values)
        row = 1
        rows = []
        cols = []
        for value in values
            col = findfirst(x -> x == value, unique_values)
            push!(rows, row)
            push!(cols, col)
            row += 1
        end 
        matrix = sparse(rows, cols, ones(Int64, length(rows)))
        return new(matrix, unique_values)
    end 
end 

function proccessing_targets(targets::Vector{<: Number})
    return targets
end 

proccessing_targets(targets::Vector{String}) = MatrixOneHot(targets)

export MLData
struct MLData
    features::DataFrame
    targets::Union{MatrixOneHot, Vector{<: Number}}
    test_features::Union{DataFrame, Nothing}
    test_target::Union{Vector{Union{Number, AbstractString}}, Nothing}

    function MLData(features::DataFrame, 
                    targets::AbstractVector{<:Union{Number, AbstractString}}, 
                    test_features::Union{DataFrame, Nothing}, 
                    test_target::Union{AbstractVector{<:Union{Number, AbstractString}}, Nothing}
                    )

        features = proccessing_features(features)
        targets = proccessing_targets(targets)
        return new(features, targets, test_features, test_target)
    end 

    function MLData(features::DataFrame,
                    targets::AbstractVector{<:Union{Number, String}}
                    )
        return MLData(features, targets, features, targets)
    
    end 

    function MLData(features::DataFrame,
                    targets::AbstractVector{<:Union{Number, String}};
                    porcent_testing::Float64 = 1.0
                    )
        # num of fats
        m = length(targets)
        # número de valores de prueba
        num_testing = Int(round(porcent_testing * m))

        # Crear los conjuntos de entrenamiento y prueba
        indices_testing = sample(collect(1:m), num_testing)
        test_features = features[indices_testing,:]
        test_target = targets[indices_testing]

        return MLData(features, targets, test_features, test_target)

        # Crear el conjunto de entrenamiento
    end 

end

end



