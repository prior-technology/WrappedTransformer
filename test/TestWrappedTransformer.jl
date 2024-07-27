using SymbolicTransformer
using Test
using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders
using SymbolicTransformer.WrappedTransformer
using TextEncodeBase
using LinearAlgebra


function test_embed()
    (model, encoder) = TestData.get_both()
    
    T = prompt(model, encoder, "Hello, world!")    
    @test T.prompt == "Hello, world!"    
    residuals = embed(T, " word")
    r=residuals[1]
    @test r.label == " word"
    @test typeof(r.vector) == Vector{Float32}
    @test r.expression == :(embed(" word"))
end

function test_unembed()
    (model, encoder) = TestData.get_both()

    #given
    T = prompt(model, encoder, "Hello,")    
    tokens = encode(encoder, " world").token
    token_ids = first(reinterpret(Int32, tokens))    
    output_vector = T.unembed_layer.layer.embed.embeddings[:,token_ids[1]]
    
    #when
    residuals = unembed(T, " world")
    r=first(residuals)

    #then
    @test r.vector == adjoint(output_vector)
    @test r.label == " world"
    @test r.expression == :(unembed(" world"))

end

function test_logits()
    (model, encoder) = TestData.get_both()

    #given an output residual which matches a specific vector of the unembedding layer
    T = prompt(model, encoder, "Hello")
    residuals = unembed(T, "Hello")
    r=transpose(first(residuals))

    #When I calculate the logits for that residual
    predictions = predict(T,r)

    #Then the logit for that token should be >> than the next closest    
    tokenid = argmax(map(p -> logit(p), predictions))
    @test predictions[tokenid].label == "Hello"
end

function test_inference()
    (model, encoder) = TestData.get_both()

    #given a transformer prompted with a sequence of numbers
    T = prompt(model, encoder, "1, 2, 3, 4")     

    #when the transformer operates on a residual of a token which continues the sequence
    residuals = embed(T, ",")
    r=residuals[1]
    y = T * r
    @test typeof(y) == Residual
    predictions = predict(T,y)
    p = first(predictions)

    #then the transformer should predict the next number in the sequence    
    @test p.label == " 5"
    @test probability(p) > 0.25
    @test p.expression == :(unembed(" 5") ⋅ (T * embed(",")))

    #and the logit should match the equivalent when using transformers.jl directly
    tjlInput = encode(encoder, "1, 2, 3, 4,")
    tjlOutput = model(tjlInput)
    @test logit(p) ≈ tjlOutput.logit[p.token_id,end,1] #token_id from vocab, end of sequence, batch 1

    #and the logit should match the result of it's own expression
    inner_product = first((unembed(" 5") ⋅ (T * embed(","))))
    @test logit(p) ≈ inner_product.vector[1]
end

function test_apply_transformer()
    model = TestData.get_model()
    #given a prompted transformer and residuals
    T = prompt(model, TestData.get_encoder(), "1, 2, 3, 4")
    residuals = prompt_residuals(T)
    #when I apply the transformer to the residuals
    result = WrappedTransformer.apply(T, residuals.hidden_state)
    #then the result should be the same as using the model directly
    expected = model((; token=T.tokens))
    @test result.hidden_state == expected.hidden_state

end
function calculate_expected_delta(b, nt)
    nt_a = Layers.apply_on_namedtuple(b.attention_norm, nt)
    nt_f = Layers.apply_on_namedtuple(b.feedforward_norm, nt)
    a = Layers.apply_on_namedtuple(b.attention, nt_a)
    f = Layers.apply_on_namedtuple(b.feedforward, nt_f)
    return a.hidden_state + f.hidden_state
end
function test_apply_block()
    #given a PromptedTransformerBlock and residuals
    (model, encoder) = TestData.get_both()
    T = prompt(model, encoder, "1, 2, 3, 4")
    prefix_residuals = prompt_residuals(T)
    transformerBlock = model.model.decoder.layers[1][1]
    promptedBlock = PromptedTransformerBlock(transformerBlock, prefix_residuals, :(test))
    
    #When I apply the block to the residuals
    result = WrappedTransformer.apply(promptedBlock, prefix_residuals.hidden_state)

    #Then the result should be the amount of change in hidden state expected from the block
    expected = calculate_expected_delta(transformerBlock, prefix_residuals)
    @test result.hidden_state ≈ expected
end

function test_apply()
    test_apply_transformer()
    test_apply_block()
    
end


function test_prefix_block()
    #given a block from a prompted transformer and residuals
    (model, encoder) = TestData.get_both()
    T = prompt(model, encoder, "1, 2, 3, 4")
    transformerBlock = model.model.decoder.layers[1][1]
    input_residuals = prompt_residuals(T)

    #when I prefix the block with the residuals
    (new_residuals, prompted_transformer_block) = WrappedTransformer.prefix_block(transformerBlock, input_residuals, :(a ⋅ b))

    #then the resulting block should hold the input residuals
    @test prompted_transformer_block.prompt_residuals == input_residuals
    #and the resulting residuals should be the result of applying the block to the input
    expected_residuals = transformerBlock(input_residuals)
    @test new_residuals.hidden_state == expected_residuals.hidden_state
end

function test_split_prediction()
    #given a Prediction
    #uses example https://en.wikipedia.org/w/index.php?title=Softmax_function&oldid=1218697397#Example
    logits = [1, 2, 3, 4, 1, 2, 1]
    max_logit = 4
    shifted_logits = logits .- max_logit
    token_id=1
    nc=WrappedTransformer.normalisation_constant(shifted_logits)
    residual = Residual([1, 3, 3, -1, -2, -3], :(test_residual), "test_residual")
    unembed = Residual([0.1, 1, 0.1, 0.1, 0.1, 0.1], :(test_unembed), "test_unembed")
    #probability= WrappedTransformer.normalise_logit(logit,max_logit,nc)
    expression=:(test)
    label="Test"
    token_id=1
    target = Prediction(unembed, residual, nc, max_logit, expression, label, 1)
    
    #when I supply a vector of Residual terms
    terms = [
        Residual([1, 2, 3, 0, 0, 0], :(test1), "Test1")
        Residual([0, 1, 0, -1, -2, 3], :(test2), "Test2")
    ]
    (results, error) = split(target, terms)


    #then I get a vector of PredictionParts showing contribution percentage, along with an error term
    @test length(results) == 2
    @test results[1].contribution ≈ (logit(target) * (2/3))
    @test results[2].contribution ≈ (logit(target) * (1/3))
    @test error ≈ 0.0
    
end

@testset "embed" test_embed()
@testset "unembed" test_unembed()
@testset "logits" test_logits()
@testset "inference" test_inference()
@testset "prefix_block" test_prefix_block()
@testset "apply" test_apply()