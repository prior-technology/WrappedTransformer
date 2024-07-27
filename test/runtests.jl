using SymbolicTransformer
using Test

include("loadmodel.jl")

include("TestLayerNormalization.jl")
include("TestVectorTransformer.jl")
include("TestAttention.jl")
include("TestWrappedTransformer.jl")
include("TestExpand.jl")
     