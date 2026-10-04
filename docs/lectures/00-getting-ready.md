# Getting Ready for the Lecture

This series assumes some background in Python, in how transformer language models
work, and in what limits a GPU's speed. You don't need to be an expert in any of
them. The links below are the quickest way to fill any gaps. Items marked
*optional* go deeper than the lectures require.

## 1. Python and notebooks

You will read and lightly edit Python in Jupyter notebooks, and look at results
in pandas tables.

- [The Python Tutorial](https://docs.python.org/3/tutorial/): sections 3–5 are enough.
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html): selecting columns and rows, `describe()`, plotting.
- [Welcome to Colab](https://colab.research.google.com/notebooks/intro.ipynb): running cells, saving a copy.

## 2. Transformers and LLMs

You should know what a transformer layer contains (attention and an MLP), what
the weights are, and that an LLM generates text **one token at a time**, feeding
each new token back in.

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar): the architecture, visually.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar): decoder-only models and token-by-token generation.
- [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt) and
  [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention) (3Blue1Brown): video explanations.
- *Optional:* [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Andrej Karpathy).

## 3. The KV cache

During generation, the model stores each token's attention **keys and values**
so it does not recompute them for every new token. This KV cache is central to
everything in this series: it lets decode process one token per step, and it
competes with the weights for GPU memory.

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), section on self-attention during generation.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (kipply): KV-cache size and the cost of a generation step.

## 4. What limits a GPU's speed

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
