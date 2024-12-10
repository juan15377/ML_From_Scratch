module TypeModelRegressions
include("../model_type.jl")
using .TypeModel
export MyRegressionModel
abstract type MyRegressionModel <: Model end

end 
