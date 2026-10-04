# [TEMP] Getting Ready for the Lecture

This series assumes some background in Python, in how transformer language models
work, and in what limits a GPU's speed. You don't need to be an expert in any of
them. Start with the vocabulary below so we all use the same words, then use the
links to fill any gaps. Items marked *optional* go deeper than the lectures require.

## Vocabulary

### The model

Token
: The unit of text an LLM reads and writes: a word, part of a word, or a
  punctuation mark. "Unbelievable!" might be three tokens: `Un`, `believable`,
  `!`. In English, a token is about ¾ of a word on average. All lengths and
  speeds in this course are counted in tokens.

Tokenizer
: The fixed procedure that converts text into a sequence of token IDs (integers)
  and back. The set of all tokens a model knows is its *vocabulary*;
  Qwen2.5 has ~152,000.

Large language model (LLM)
: A neural network that, given a sequence of tokens, predicts a probability for
  every possible next token. Today's LLMs are *transformers* (see below).

Parameters (weights)
: The learned numbers inside the model, mostly arranged in large matrices. Model
  sizes are quoted in parameters: "32B" means 32 billion. Stored in 16-bit
  precision (*fp16* or *bf16*), each parameter takes 2 bytes.

Layer
: Transformers are a stack of identical layers (Qwen2.5-32B has 64). Each layer
  has an **attention** block, which lets each token look at earlier tokens, and an
  **MLP** (feed-forward) block, which transforms each token on its own.

### Running the model

Inference
: Using a trained model to produce outputs, as opposed to *training* it.
  This course is entirely about inference.

Forward pass
: One run of input tokens through all the layers. It produces a prediction for
  the next token.

Prompt and output
: The prompt is the input text (the user's message plus any instructions or
  context); the output is the text the model generates in response.

Autoregressive generation
: Generating one token at a time: predict a token, append it to the input, and
  run the model again. A 100-token answer takes 100 steps.

KV cache
: The attention **keys** and **values** of every token so far, kept in GPU memory
  so they are not recomputed at every step.

Prefill and decode
: The two phases of handling a request. *Prefill* processes the whole prompt in
  one forward pass and fills the KV cache. *Decode* then generates the output one
  token per step. Lecture 1 is about the difference.

Request
: One prompt sent to the serving system, together with the response it gets back.

Batch
: A group of requests processed together in the same forward pass, so the
  weights are loaded once for all of them.

### Hardware

GPU
: The accelerator that runs the model, for example an NVIDIA A100 or H100.

FLOP and FLOP/s
: A FLOP is one floating-point operation (an add or a multiply). FLOP/s is how
  many a chip can do per second. An A100 peaks at 312 *tera*FLOP/s (312 × 10¹²).

HBM and memory bandwidth
: HBM (high-bandwidth memory) is the GPU's main memory, which holds the weights
  and the KV cache. Its *bandwidth*, in bytes per second, is how fast data can
  move between HBM and the compute units.

Compute-bound and memory-bound
: A computation is *compute-bound* if the math takes longer than loading its
  data, and *memory-bound* if loading the data takes longer.

Tensor parallelism (TP)
: Splitting every weight matrix across several GPUs so they share the work (and
  the memory) of each forward pass. "TP2" means two GPUs.

### Serving and metrics

Serving system
: The software that receives requests, schedules them on GPUs, and streams back
  outputs, e.g. vLLM, SGLang, or TensorRT-LLM.

Latency and throughput
: *Latency* is how long one request takes. *Throughput* is how much work the
  system finishes per second, in requests/s or tokens/s.

TTFT, TPOT, E2E
: *Time to first token*, *time per output token*, and *end-to-end* latency
  (arrival to last token). Defined precisely in lecture 1.

QPS
: Queries (requests) per second arriving at the system: the load.

p50 and p99
: Percentiles. p50 is the median; p99 is the value that 99% of requests are at or below,
  which measures the slowest ("tail") requests.

SLO
: Service-level objective: a latency target the system promises to meet, such as
  "p99 TTFT under 500 ms".

## Python and notebooks

You will read and lightly edit Python in Jupyter notebooks, and look at results
in pandas tables.

- [The Python Tutorial](https://docs.python.org/3/tutorial/): sections 3–5 are enough.
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html): selecting columns and rows, `describe()`, plotting.
- [Welcome to Colab](https://colab.research.google.com/notebooks/intro.ipynb): running cells, saving a copy.

## Transformers and LLMs

You should know what a transformer layer contains (attention and an MLP), what
the weights are, and that an LLM generates text **one token at a time**, feeding
each new token back in.

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar): the architecture, visually.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar): decoder-only models and token-by-token generation.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Polo Club, Georgia Tech): an interactive visualization of GPT-2 running live in your browser. Type a prompt and watch it flow through every layer.
- [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt) and
  [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention) (3Blue1Brown): video explanations.
- *Optional:* [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Andrej Karpathy).

## The KV cache

During generation, the model stores each token's attention **keys and values**
so it does not recompute them for every new token. This KV cache is central to
everything in this series: it lets decode process one token per step, and it
competes with the weights for GPU memory.

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), section on self-attention during generation.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (kipply): KV-cache size and the cost of a generation step.

## What limits a GPU's speed

A computation is limited either by how fast the GPU can **do math** (FLOP/s) or by
how fast it can **move data** from memory (bytes/s). Which one depends on how many
operations it does per byte it loads, its *arithmetic intensity*. This is the
roofline model, and lecture 1 is built on it.

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He): compute-, memory- and overhead-bound, without math.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) (*How To Scale Your Model*): the roofline model with worked examples.
- *Optional:* [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) (*How To Scale Your Model*).

Numbers worth remembering for the GPU used in the labs, the
[NVIDIA A100-80GB](https://www.nvidia.com/en-us/data-center/a100/):

| | A100-80GB (SXM) |
|---|---|
| Peak 16-bit math (dense) | 312 TFLOP/s |
| Memory (HBM) bandwidth | ~2.0 TB/s |
| Memory capacity | 80 GB |

## Are you ready?

You're ready for lecture 1 if you can answer these, at least roughly:

1. How many bytes does a 30-billion-parameter model take in 16-bit precision?
2. When an LLM produces a 100-token answer, how many times does it run the model?
3. What does the KV cache store, and why does it grow as the conversation gets longer?
4. If a computation does 10 FLOPs for every byte it reads, is an A100 limited by
   compute or by memory bandwidth? (Hint: 312 TFLOP/s ÷ 2 TB/s.)

## Setting up

The labs run in your browser through Colab or Binder, or on your own machine.
See {doc}`../getting-started/run-in-browser` and {doc}`../getting-started/run-locally`.
Lecture 1 starts with a setup cell that checks everything works.
