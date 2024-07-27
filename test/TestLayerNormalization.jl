
include("../data/probe_token.jl")
include("../data/pre_norm.jl")

@testset "SymbolicTransformer.jl" begin
    bias = 0.8328

    final_residual = LN(pre_norm)

    logit = sum(.*(probe_token, final_residual)) + bias
    # should return  11.4077
    @test logit ≈ 11.4077 atol=1e-3

    
end

@testset "Layer Normalization" begin
    using SymbolicTransformer
    using SymbolicTransformer.WrappedTransformer
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

    @testset "center" begin

    end
    
end