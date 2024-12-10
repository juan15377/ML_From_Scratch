# se incluyen todos lo ficheros de uso comun el modelos de regresssion


module RegressionsModels

using ..TypeModel

abstract type MyRegressionModel <: Model end 


#include("linear_regression/Linear_regression.jl")
#include("logistic_regression/logistic_regression.jl")


end 
# de aqui debe arrancar 