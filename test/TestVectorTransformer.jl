using SymbolicTransformer

@testset "VectorTransformer.jl" begin
    config = Pythia70ModelConfig()
    #check that getter is a function 
    @test @isdefined config
end

@testset "inverse_frequencies" begin
    freqs = SymbolicTransformer.inverse_frequencies(100,8)
    @test length(freqs) == 5
    @test (1.0, 0.1, 0.01) == (freqs[1], freqs[3], freqs[5])
end

@testset "rotate_half" begin
    m = [1 2 3 4 5; 10 20 30 40 50]
    r = SymbolicTransformer.rotate_half(m)
    @test r == [-3 -4 -5 1 2; -30 -40 -50 10 20]
end

@testset "frequencies" begin
    m = SymbolicTransformer.frequencies(100, 16, 40)
    
    @test size(m) == (9,40)
    @test m[1,1] == 1.0
    @test m[9,40] == 0.4
    
end

function TestModelConfig()
    seq_len = 120
    rotary_pct=0.25
    rotary_emb_base = 100
    d_model=64
    n_heads=8
    d_head=d_model/n_heads
    return ModelConfig(seq_len, rotary_pct, rotary_emb_base, d_model, n_heads, d_head)
end
@testset "apply_rotary" begin
    m = [1 2 3 4 5 6 7 8; 10 20 30 40 50 60 70 80]
    config = TestModelConfig()
    r = SymbolicTransformer.apply_rotary(config, m)

    @test size(r) == size(m)
    @test r[1,1] ≈ (cos(1) - (2 * sin(1)))
    @test r[1,4] == 4   
    
end

@testset "apply_rotary" begin
    m = [1 2 3 4 5 6 7 8; 10 20 30 40 50 60 70 80]
    config = TestModelConfig()
    r = SymbolicTransformer.apply_rotary(config, m)

    @test size(r) == size(m)
    @test r[1,1] ≈ (cos(1) - (2 * sin(1)))
    @test r[1,4] == 4   
    
end