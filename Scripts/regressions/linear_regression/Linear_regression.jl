

module LinearRegression
include("../../Utils/Data_structs/data_structs.jl")
include("../regressions.jl")
include("../../Utils/Parameters/regression/parameters_regression.jl")

using .Data_Structs
using .Regressions
using .ParameterRegression
using DataFrames

export MyLinearRegressionModel
import .ParameterRegression.ParametersRegression
export ParametersRegression
mutable struct MyLinearRegressionModel <: MyRegressionModel
    data::MLData
    parameters::Union{ParametersRegression, Nothing}

    function LinearRegressionModel(ml_data::MLData 
                             )
        new(ml_data, NaN)
    end 

end

end 
