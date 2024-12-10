include("../../../Scripts/Utils/Data_structs/data_structs.jl")

using .Data_Structs
using DataFrames

features = DataFrame(
    x1 = [1, 2, 3],
    x2 = ["A", "B", "A"]
)

targets = [1, 2, 3]

data = MLData(features, targets)



#falta una funcion que me de una conmbinacion lineal de las features y un vector de parametros

features_predict = DataFrame(
    x1 = [10],
    x2 = [11]
)


MyRegressionModel()