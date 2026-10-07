# 0. Getting Ready for the Lecture

By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖

This series assumes some background in Python, in how transformer language models
work, and in what limits a GPU's speed. You don't need to be an expert in any of
them. Start with the vocabulary below so we all use the same words, then use the
links to fill any gaps.

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
  **MLP** (also called a feed-forward network, or *FFN*) block, which transforms each token on its own.

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
: The attention **keys** and **values** of every token processed so far, stored and reused across steps so that they are not recomputed at every step.

Prefill and decode
: The two phases of handling a request. *Prefill* processes the whole prompt in
  one forward pass and fills the KV cache. *Decode* then generates the output one
  token per step. Lecture 1 is about their differences.

Request
: One prompt sent to the serving system, together with the response it gets back.

Batch
: A group of requests processed together in the same forward pass, so the
  weights are loaded once for all of them.

### Hardware

GPU
: The accelerator that runs the model, for example an NVIDIA A100 or H100.

FLOP and FLOP/s (FLOPs)
: A FLOP is one floating-point operation (an add or a multiply). FLOP/s (also often written as FLOPs for simplicity) is how
  many FLOP can be done per second. An A100 peaks at 312 *tera*FLOP/s (312 × 10¹²).

HBM and memory bandwidth
: HBM (high-bandwidth memory) is the GPU's main memory (well, at least for expensive ones like A100 or H100), which holds the weights
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

You will read and lightly edit Python in Jupyter notebooks. If you are not familiar with Python or Jupyter notebooks, here are some materials that can help.

- [The Python Tutorial](https://docs.python.org/3/tutorial/): sections 3–5 are enough.
- [Welcome to Colab](https://colab.research.google.com/notebooks/intro.ipynb): running cells, saving a copy.

## Transformers and LLMs

You should have a rough understanding of how transformers work. If you’re not familiar with them, here are some of my favorite resources (these are my personal favorites, not AI-generated!).

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar): this is one of my favorites, but note that it explains how the *original* transformer architecture worked. Modern-day transformers (*decoder-only* transformers) look different, as shown in the next resource below.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar): decoder-only transformers and token-by-token generation.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Polo Club, Georgia Tech): an interactive visualization of GPT-2 running live in your browser. Type a prompt and watch it flow through every layer.

## The KV cache

During generation, the model stores each token's attention **keys and values**
so it does not recompute them for every new token. This KV cache is central to
everything in this series: it lets decode process one token per step, and it
competes with the weights for GPU memory.

- [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) (Sebastian Raschka): what the cache stores and why, with diagrams, then a from-scratch implementation.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (kipply): KV-cache size and the cost of a generation step.

## What limits a GPU's speed

A computation is limited either by how fast the GPU can **do math** (FLOP/s) or by
how fast it can **move data** from memory (bytes/s). Which one depends on how many
operations it does per byte it loads, its *arithmetic intensity*. This is the
roofline model, and lecture 1 is built on it.

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He): compute-, memory- and overhead-bound, without math.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) (*How To Scale Your Model*): the roofline model with worked examples.

Numbers worth familiarizing for the GPU used in the labs, the
[NVIDIA A100-80GB](https://www.nvidia.com/en-us/data-center/a100/):

| | A100-80GB (SXM) |
|---|---|
| Peak 16-bit math (dense) | 312 TFLOP/s |
| Memory (HBM) bandwidth | ~2.0 TB/s |
| Memory capacity | 80 GB |

## Are you ready?

You're ready for lecture 1 if you can answer these. Click an answer to check it;
wrong answers can be retried.

```{quiz}
- q: How much memory does a 30-billion-parameter model take in 16-bit precision?
  options: [30 GB, 60 GB, 120 GB, 480 GB]
  answer: 1
  explain: 16 bits is 2 bytes per parameter, so 30 × 10<sup>9</sup> × 2 B = 60 GB.
- q: To produce a 100-token answer, about how many times does an LLM run the model (forward passes)?
  options: ["1", "About 100", "100 × the prompt length"]
  answer: 1
  explain: Generation is autoregressive, one token per forward pass. The first pass (prefill) handles the whole prompt and yields the first token; each later token needs one more pass.
- q: What does the KV cache store?
  options: [The model's weights, The attention keys and values of the tokens processed so far, The generated text, The gradients used for training]
  answer: 1
  explain: Keeping each token's keys and values means they are computed once instead of at every generation step.
- q: Why does the KV cache grow as a conversation gets longer?
  options: [It stores an entry for every token so far, The model's weights grow over time, It keeps a copy of every previous answer's weights]
  answer: 0
  explain: Every new token, prompt or output, adds its keys and values in every layer.
- q: A computation does 10 FLOPs for every byte it reads from memory. On an A100 (312 TFLOP/s, 2 TB/s), what limits it?
  options: [Compute (FLOP/s), Memory bandwidth]
  answer: 1
  explain: The A100 can do 312 × 10<sup>12</sup> ÷ 2 × 10<sup>12</sup> ≈ 156 FLOPs in the time it reads one byte. At only 10 FLOPs per byte, the math units wait for memory.
```

## Setting up

The labs run in your browser through Colab or Binder, or on your own machine.
See {doc}`../getting-started/run-in-browser` and {doc}`../getting-started/run-locally`.
Lecture 1 starts with a setup cell that checks everything works.
