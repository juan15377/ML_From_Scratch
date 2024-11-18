
module Data_Structs

export MLData

struct MLData
    features::Matrix{Float64}
    target::Vector{Float64}
    test_features::Union{Matrix{Float64},Nothing}
    test_target::Union{Vector{Float64},Nothing}
end 


end 