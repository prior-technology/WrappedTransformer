# WrappedTransformer

## Summary

Extracts Transformers.jl and other dependencies from [SymbolicTransformer](https://github.com/prior-technology/SymbolicTransformer/)

## Dependencies

Depends on an unpublished version of SymbolicTransformer

## Installation

From the Julia REPL running in the base of the repo folder, press `]` to enter the package manager, then run the following commands:
```repl
(@v1.11) pkg> activate .
  Activating project at `~/repos/WrappedTransformer`

(WrappedTransformer) pkg> add https://github.com/prior-technology/SymbolicTransformer#refactor
    Updating git-repo `https://github.com/prior-technology/SymbolicTransformer`
...
Precompiling project...
  196 dependencies successfully precompiled in 105 seconds. 36 already precompiled.
```

## Running Tests

Following installation instructions above run the following commands in the package manager REPL:
```repl
(WrappedTransformer) pkg> test
     Testing WrappedTransformer
...
     Testing Running tests...
WARNING: Method definition _dummy_backedge() in module Memoization at /root/.julia/packages/Memoization/ON3Za/src/Memoization.jl:49 overwritten at /root/.julia/packages/Memoization/ON3Za/src/Memoization.jl:65.
┌ Warning: fuse_unk is unsupported, the tokenization result might be slightly different in some cases.
└ @ Transformers.HuggingFace ~/.julia/packages/Transformers/qH1VW/src/huggingface/tokenizer/utils.jl:42
┌ Warning: padsym <pad> not in vocabulary, this might cause problem.
└ @ Transformers.TextEncoders ~/.julia/packages/Transformers/qH1VW/src/textencoders/gpt_textencoder.jl:153
Test Summary: | Pass  Total   Time
embed         |    4      4  46.7s
Test Summary: | Pass  Total  Time
unembed       |    3      3  0.5s
Test Summary: | Pass  Total  Time
logits        |    1      1  5.7s
Test Summary: | Pass  Total   Time
inference     |    6      6  55.1s
Test Summary: | Pass  Total  Time
prefix_block  |    2      2  0.1s
Test Summary: | Pass  Total  Time
apply         |    2      2  9.6s
Test Summary: | Pass  Broken  Total  Time
expand        |    6       1      7  6.1s
     Testing WrappedTransformer tests passed

```

## Usage

Eventually this will be usable from SymbolicTransformer where more detailed instructions are available. For now refer to the tests for examples of usage.

