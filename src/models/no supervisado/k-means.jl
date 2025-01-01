include("../../Utils/Data_structs/data_structs.jl")

mutable struct MyKMeansModel
    data::MLData
    parameters::Dict
    k::Int
    function MyKMeansModel(data::MLData, k::Int)
        parameters = zeros(size(data.features, 2), k)
        new(data, parameters, k)
    end 
end