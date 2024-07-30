module WrappedTransformer
using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using Transformers.HuggingFace
using SymbolicTransformer
using LinearAlgebra
import Base.show
using VectorTransformer

export PromptedTransformer,PromptedTransformerBlock, Residual, Prediction, prompt, embed, unembed, predict, dot, prompt_residuals, extract_blocks, expand, logit, probability, block_outputs

"Wraps a transformer and encoder with a prompt"
struct PromptedTransformer <: SymbolicTransformer.Operation
    "Huggingface pretrained model"
    model 
    "TextEncoder corresponding with model"
    encoder
    "Embedding layer"
    embed_layer
    "Output layer which maps residual vectors to logits"
    unembed_layer
    "Original string of the prompt"
    prompt :: AbstractString
    "result of Transformers.TextEncoders.encode - nvocab x ntokens OneHotArray"
    tokens
    "Simple expression representing this Transformer"
    expression 
end

global current_transformer::PromptedTransformer

function show(io::IO, T::PromptedTransformer)
    show(io, MIME("text/plain"), T)
end
function show(io::IO, ::MIME"text/plain", T::PromptedTransformer)
    
    if (get(io, :compact, false) == true)
        print(io, "PromptedTransformer(\"$(T.prompt)\")")
    else
        #Display the model type, encoder type and prompt
        #typeof(T.model) is quite complex, simplify it
        model_type = split(string(typeof(T.model)), "{")[1]
        encoder_type = split(string(typeof(T.encoder)), "{")[1]
        print(io, "PromptedTransformer($model_type, $encoder_type, \"$(T.prompt)\")")        
    end
end

struct PromptedTransformerBlock <: SymbolicTransformer.Operation
    "One block of a Transformers.jl Huggingface transformer"
    block
    prompt_residuals
    expression
end

function show(io::IO, ::MIME"text/plain", T::PromptedTransformerBlock)    
    print(io, "$(T.expression)")        
end

"Represents a vector in the transformer's residual space"
struct Residual <:  SymbolicTransformer.Residual
    "vector in the residual space"
    vector 
    "Expression showing the source of this residual"
    expression
    "Label for printing"
    label
end
function show(io::IO, ::MIME"text/plain", r::Residual)
    if (get(io, :compact, false) == true)
        print(io, r.expression)
    else
        print(io, "Residual(\"$(r.label)\", $(r.expression))")
    end
end

"Encapsulates the normalised weight output for a particular token by a language model"
struct Prediction <: SymbolicTransformer.Prediction
    unembed
    residual
    normalization_constant
    max_logit
    expression
    label
    token_id
end

struct PredictionTerm <: SymbolicTransformer.Prediction
    unembed
    residual
    scale
    normalization_constant
    max_logit
    prediction_logit
    expression
end

function probability(p::SymbolicTransformer.Prediction)
    return normalise_logit(logit(p), p.max_logit, p.normalization_constant)
end

function probability(p::PredictionTerm)
    return (logit(p)/p.prediction_logit) * normalise_logit(p.prediction_logit, p.max_logit, p.normalization_constant)
end

function logit(p::PredictionTerm)    
    return p.scale * ( Transpose(p.unembed.vector) ⋅ p.residual.vector )
end

function logit(p::SymbolicTransformer.Prediction)
    return Transpose(p.unembed.vector) ⋅ p.residual.vector
end

function show(io::IO, ::MIME"text/plain", p::SymbolicTransformer.Prediction)
    prob = round(100*probability(p),digits=2)
    if (get(io, :compact, false) == true)
        print(io, "Prediction($prob% $(p.label)")
    else
        print(io, "Prediction($prob% \"$(p.label)\", $(p.expression)")
    end
end

function show(io::IO, ::MIME"text/plain", p::PredictionTerm)
    prob = round(100*probability(p),digits=2)
    if (get(io, :compact, false) == true)
        print(io, "Prediction($prob%)")
    else
        l = round(logit(p),digits=2)
        print(io, "Prediction($prob% l=$l $(p.expression))")
    end
end


"tokenizes the utterance, and returns an operation"
function prompt(causal_lm_model::Transformers.HuggingFace.HGFGPTNeoXForCausalLM,
        encoder,
        utterance)
    model = causal_lm_model.model
    unembed = causal_lm_model.cls
    embed = model.embed
    
    tokens = encode(encoder, utterance).token

    global current_transformer = PromptedTransformer(model, encoder, embed, unembed, utterance, tokens, :(T))
    return current_transformer
end

function bra(s::AbstractString)
    return "⟨ $s |"
end
function ket(s::AbstractString)
    return "| $s ⟩"
end

"tokenizes the utterance, and returns a Vector of Residuals representing the embedding vectors"
function embed(transformer, utterance)    
    tokens = encode(transformer.encoder, utterance).token
    labels = decode(transformer.encoder,tokens)
    vectors = transformer.embed_layer((; token=tokens))
    expressions = map(x -> :(embed($x)), labels)
    residuals = map(x -> 
        Residual(vectors.hidden_state[:,x],
            expressions[x], 
            labels[x]), 
        1:length(labels))
    return residuals
end

function embed(T, token_id::Integer)  
    token_string = decode(T.encoder, token_id)
    return Residual(T.embed_layer.token.embeddings[:,token_id], :(embed($token_string)), token_string)
end

function embed(utterance)
    return embed(current_transformer, utterance)
end

"tokenizes the utterance, and returns a Vector of Residuals which map output residuals to logits"
function unembed(transformer, utterance::AbstractString)    
    tokens = encode(transformer.encoder, utterance).token
    labels =  decode(transformer.encoder,tokens)
    tokenids = reinterpret(Int32, tokens)
    output_vectors = transformer.unembed_layer.layer.embed.embeddings[:,tokenids]
    
    expressions = map(x -> :(unembed($x)), labels)
    residuals = map(x -> 
        Residual(adjoint(output_vectors[:,x]),
            expressions[x], 
            labels[x]), 
        1:length(labels))
    return residuals
end

function unembed(utterance::AbstractString)
    return unembed(current_transformer, utterance)
end

function unembed(transformer, token_id::Integer)
    token_string = decode(transformer.encoder, token_id)

    return Residual(transformer.unembed_layer.layer.embed.embeddings[:,token_id], :(unembed($token_string)), token_string) 
end


function Base.:(+)(r1:: Residual, r2:: Residual)
    return Residual(r1.vector + r2.vector, :($(r1.expression) + $(r2.expression)), """$(r1.label) + $(r2.label)""")
end

"""
    Returns residual vectors associated with the prompt in an Operation
"""
function prompt_residuals
end

function prompt_residuals(T::PromptedTransformer)
        #We pass in an arbitrary residual vector, so bypass the embedding layer
    input = (; token=T.tokens)
    return T.embed_layer(input)
end

function prompt_residuals(B::PromptedTransformerBlock)
    return B.prompt_residuals
end

"""
    Applies the model to the hidden state. 
    
    This function was added so Base.:(*) doesn't depend on the type of operation
"""
function apply
end
function apply(T::PromptedTransformer, hidden_state)
    T.model.decoder((; hidden_state=hidden_state))
end

function apply(B::PromptedTransformerBlock, hidden_state)
    #adjust to return contribution from this block without adding to the input
    #@functor ParallelPreNorm2TransformerBlock in https://github.com/chengchingwen/Transformers.jl/blob/91a3fe00bad5bb9ebff35b61356c3d52ad3efba3/src/huggingface/implementation/gpt_neox/load.jl#L25C5-L25C69
    #returns hidden_state = a.hidden_state + f.hidden_state + nt.hidden_state
    #we just want (a.hidden_state + f.hidden_state) so deduct nt.hidden_state before returning
    #TODO: write our own version of ParallelPreNorm2TransformerBlock to encapsulate this
    result = B.block((; hidden_state=hidden_state))
    block_contribution = result.hidden_state - hidden_state
    return (; hidden_state=block_contribution)
end

function append_hidden_state(hidden_state, r::Residual)
    return hcat(hidden_state, r.vector)
end
function append_hidden_state(hidden_state, target_residuals:: AbstractVector{Residual})
    
    new_residual_matrix = hcat([r.vector for r in target_residuals]...)
    hcat(hidden_state, new_residual_matrix)
end

function label(T::PromptedTransformer)
    return T.prompt
end
function label(T::PromptedTransformerBlock)    
    return T.expression
end
function label(r:: Residual)
    return r.label
end
"applies the model to the token"
function Base.:(*)(T::SymbolicTransformer.Operation, r:: Residual)
    #To transform a new token at the end of a batch of tokens, we would push! the index of the 
    #new token onto tokens.onehots, which applies a corresponding change to the tokens OneHotArray

    residuals = prompt_residuals(T)
    hidden_state = append_hidden_state(residuals.hidden_state, r)
    y = apply(T,hidden_state)
    #take the residual in the last position
    return Residual(y.hidden_state[:,end], :($(T.expression) * $(r.expression)), string(label(T), label(r)))
end
function Base.:(*)(Op::SymbolicTransformer.Operation, target_residuals :: AbstractVector{Residual})

    residuals = prompt_residuals(Op)
    hidden_state = append_hidden_state(residuals.hidden_state, target_residuals)
    y = apply(Op,hidden_state)
    
    #return output residuals in positions corresponding with the target residuals    
    result_vectors = y.hidden_state[:,end-length(target_residuals)+1:end]
    return [Residual(result_vectors[:,i], :($(Op.expression) * $(target_residuals[i].expression)), string(label(Op), label(target_residuals[i]))) for i in eachindex(target_residuals)]
end

function LinearAlgebra.dot(r1:: Residual, r2:: Residual)
    return Residual(LinearAlgebra.dot(r1.vector,r2.vector), :($(r1.expression) ⋅ $(r2.expression)), """< "$(r1.label)" | "$(r2.label)" >""")
end

function LinearAlgebra.transpose(r:: Residual)
    return Residual(transpose(r.vector), :(transpose($(r.expression))), """ transpose($(r.label)) """)
end

function LinearAlgebra.adjoint(r:: Residual)
    return Residual(adjoint(r.vector), :(($(r.expression))'), """ ($(r.label))' """)
end

function LinearAlgebra.dot(v1:: Vector{Residual}, v2:: Vector{Residual})
    return dot.(v1, v2)
end

"Returns the sum of the exponentials of the logits"
function normalisation_constant(logits)
    return sum(exp.(logits))
end

"softmax normalisation from a logit value to a probability"
function normalise_logit(logit, shift_logit, normalization_constant)
     exp(logit-shift_logit) / normalization_constant
end

"Accepts a residual which represents output from the last position in the last block of a transformer, and returns 
predictions for the next token. The returned predictions encapsulate the logit, normalized probability, and an expression 
which traces the tokens involved in the prediction"
function predict(T::PromptedTransformer,r:: Residual)
    (_, logits) = T.unembed_layer((; hidden_state=r.vector))
    maxl = maximum(logits)
    shift_logits = logits .- maxl
    nc = normalisation_constant(shift_logits)
    
    result = [
        begin
            #probability = normalise_logit(logit, maxl, nc)
            unembed_residual = unembed(T, token_id)        
            expression = :($(unembed_residual.expression) ⋅ $(r.expression))
            label = unembed_residual.label
            Prediction(unembed_residual, r, nc, maxl,  expression, label, token_id)
        end
        for (token_id, logit) in enumerate(logits)
    ]
    #reorder by decreasing logit value
    return sort!(result; by = x -> logit(x), rev=true, dims=1)
    
end
function wrap(ln::Transformers.Layers.LayerNorm)
    return :(LN)

end

function promptBlock(block::Transformers.Layers.AbstractTransformerBlock, residuals::AbstractVector{Residual})

    #return PromptedTransformerBlock(block, residuals,:($block * $residuals))
end
function wrap(transformer_blocks::Transformers.Layers.Transformer, input_residuals::AbstractVector{Residual})
    #the operations within transformer operator are composed
    #so return an expression with each operation seperated by the composition operator ∘
    return []
end

"""
Return a PromptedTransformerBlock which includes prefix_residuals with the result of applying those residuals to the block

Note that prefix_residuals is expected to be a NamedTuple with "hidden_state" referring to a matrix
"""
function prefix_block(block::Transformers.Layers.AbstractTransformerBlock, prefix_residuals, expression)
   
    promptedBlock = PromptedTransformerBlock(block, prefix_residuals, expression)
    residuals = block(prefix_residuals) 
    return (residuals, promptedBlock)
end

"Takes an iterable of Transformer blocks and an initial residual. Returns PromptedTransformerBlocks
where each includes residuals from applying the last prefix to the last transformer"
function apply_blocks(blocks, prefix_residuals)
    result = []
    for (i,block) in enumerate(blocks)
        (prefix_residuals, promptedBlock) = prefix_block(block, prefix_residuals, :(extract_blocks(T)[$i]))
        push!(result, promptedBlock)
    end
    return result
end

function extract_blocks(model::Transformers.HuggingFace.HGFGPTNeoXModel, prefix_residuals)
    ln = model.decoder.layers[2]
    transformer = model.decoder.layers[1]
    return (ln = ln, blocks = apply_blocks(transformer.blocks, prefix_residuals))
end

function extract_blocks(model::Transformers.HuggingFace.HGFGPTNeoXForCausalLM, prefix_residuals)    
    return extract_blocks(model.model, prefix_residuals)
end
"Returns a tuple of the LayerNorm and an array of PromptedTransformerBlock from a PromptedTransformer"
function extract_blocks(T::PromptedTransformer)    
    residuals = prompt_residuals(T)
    return extract_blocks(T.model, residuals)
end

"implement center for Residual type"
function VectorTransformer.center(r::Residual)
    Residual(VectorTransformer.center(r.vector), :(center($(r.expression))), """ center($(r.label)) """)
end

norm_square(vector) = LinearAlgebra.norm(vector, 1)
norm_square(r::Residual) = LinearAlgebra.norm(r.vector, 1)

"Apply for"
function rewrite(x::Residual, ln::Transformers.LayerNorm, normedTerms)
    
    
    return (scale, seperateTerms)        
end

"Extract the left and right hand terms from a prediction expression"
function prediction_terms(prediction::Prediction)
    args = prediction.expression.args
    return (args[2], args[3])
end

function affine(LN::Transformers.Layers.LayerNorm, x::Vector{Float32})
    return x .* LN.α .+ LN.β
end

function affine(LN::Transformers.Layers.LayerNorm, x::Residual)
    return Residual(affine(LN, x.vector), :(a($x)), """a($(x.label))""")
end
function gain(LN::Transformers.Layers.LayerNorm, x::AbstractVector)
    return x .* LN.α
end
function gain(LN::Transformers.Layers.LayerNorm, x::Residual)
    return Residual(gain(LN, x.vector), :(α $x), """gain($(x.label))""")
end

function bias(LN::Transformers.Layers.LayerNorm)
    return LN.β
end

function expand(ln, xs)
    
    #<y, LN (a + b)> =  \frac{\sqrt{N}}{\sqrt{|c(a+b)|^2 + N \epsilon} } (<y,c(a)> + <y, c(b)>) 
    #$$ <x , LN(a+b)> = <x, \lambda c(a) \odot \gamma> + <x, \lambda c(b) \odot \gamma> + <x, \beta>$$
    N = length(xs[1])
    λ = sqrt(N) / sqrt(norm_square(center(sum(xs))) + N * ln.ϵ)
    
    return map(x -> λ .* gain(ln, center(x)), xs)
end

#TODO: implement for vector of residuals by appending to the prompt
#function block_outputs(T::PromptedTransformer, inputs::Vector{Residual})

function block_outputs(T::PromptedTransformer, input::Residual)
    
    (ln, blocks) = extract_blocks(T)
    
    blockOutputs = [input]
    for block in blocks
        blockOutput = block * sum(blockOutputs)
        push!(blockOutputs, blockOutput)
    end
    return (ln,blockOutputs)
end
"Replace a prediction with the contribution to the prediction from each block of the transformer"
function expand(T::PromptedTransformer, prediction::Prediction, input::Residual)
    
    (ln, blocks) = extract_blocks(T)
    
    blockOutputs = [input]
    for (i,block) in enumerate(blocks)
        blockOutput = block * sum(blockOutputs)
        expression = :($(block.expression) * sum(blockOutputs[range(1,$i)]))
        label = """B$i("$(input.label)")"""
        blockOutput = Residual(blockOutput.vector, expression, label)
        push!(blockOutputs, blockOutput)
    end
    
    #<x, LN (a + b)> =  \frac{\sqrt{N}}{\sqrt{|c(a+b)|^2 + N \epsilon} } (<x,c(a)> + <x, c(b)>) 
    N = length(input.vector)
    (lhs, rhs) = prediction_terms(prediction)
    scale = sqrt(N) / sqrt(norm_square(center(sum(blockOutputs))) + N * ln.ϵ)
    centeredBlockOutputs = map(residual -> center(residual), blockOutputs)
    transformedBlockOutputs = map(residual -> gain(ln, residual), centeredBlockOutputs)
    push!(transformedBlockOutputs, Residual(ln.β, :(β), "β"))
    return [

        PredictionTerm(
            prediction.unembed, 
            residual, 
            scale, 
            prediction.normalization_constant, 
            prediction.max_logit, 
            logit(prediction),
            if (i==1) 
                :($lhs ⋅ center($(input.expression)))
            elseif (i==length(transformedBlockOutputs))
                :($lhs ⋅ T.ln.β)
            else
                :($lhs ⋅ expand(T, $rhs)[$i])
            end
        ) 
        for (i,residual) in enumerate(transformedBlockOutputs)
    ]
end

end