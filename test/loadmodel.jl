module TestData

using Memoization
#import HuggingFaceApi: set_endpoint
import Transformers.HuggingFace: load_config,load_model,load_tokenizer

const model = "EleutherAI/pythia-14m"

@memoize function get_config()
#    set_endpoint("file:///C:\\Users\\StephenPrior\\.julia\\dev\\SymbolicTransformer\\data\\EleutherAI\\pythia-14m\\resolve\\main")
    
    return load_config(model)
end
@memoize function get_encoder()
    config = load_config(model)
    return load_tokenizer(model, config=config)
end
@memoize function get_model()
    config = load_config(model)
    return load_model(config.model_type, model, "forcausallm"; config=config)
end

@memoize function get_both()
    return (get_model(), get_encoder())
end

end
