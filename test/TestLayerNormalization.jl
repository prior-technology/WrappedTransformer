
include("../data/probe_token.jl")
include("../data/pre_norm.jl")


@testset "Layer Normalization" begin
    using SymbolicTransformer
    using WrappedTransformer
    using Transformers
    using Transformers.Layers
    using LinearAlgebra
    N=8
    
    alpha = Test.Random.randn(Float32, N)
    beta = Test.Random.randn(Float32, N)
    epsilon = 1e-5
    ln = Transformers.Layers.LayerNorm(alpha, beta, epsilon)
    xs = [Test.Random.randn(Float32, N) for _ in 1:10]
    y = Test.Random.randn(Float32, N)
    x_total = sum(xs)

    #when I expand the LayerNorm
    expanded_ln = expand(ln, xs)
    expected = y ⋅ ln(x_total)
    actual = sum(map(ex -> y ⋅ ex, expanded_ln)) + (y ⋅ beta)
    #then 
    @test expected≈actual broken=true

end