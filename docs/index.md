# Studying LLM Systems Without GPUs

By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖

**A hands-on lecture series on LLM serving systems that you can follow even when you don't have GPUs.**

The labs run on a **CPU-only simulator** of LLM inference clusters. It predicts
each operation's runtime from profiles measured on real A100 and H100 GPUs, so you
can experiment with hardware you don't have. The simulator we are using is
[Vidur-Agent](https://github.com/psu-paws/Vidur-Agent) (Kim et al., IISWC 2026), which extends Microsoft's
[Vidur](https://github.com/microsoft/vidur) (Agrawal et al., MLSys 2024).

Each lecture pairs an explanation with a runnable notebook. Click
**Open in Colab** or **Launch Binder** at the top of any lab page to run it in
your browser, with nothing to install. Start with {doc}`lectures/00-getting-ready`.

The website and lecture materials were built with the help of Claude Code. I’ve done my best to catch and fix any hallucinated content, but please reach out if you spot any issues or have suggestions!

```{list-table}
:header-rows: 1
:widths: 6 40 54

* - #
  - Lecture
  - Topics
* - 0
  - {doc}`lectures/00-getting-ready`
  - prerequisites and background reading
* - 1
  - {doc}`lectures/01-prefill-and-decode`
  - TTFT/TPOT; why prefill is compute-bound and decode is memory-bound
* - 2
  - {doc}`lectures/02-batching-and-load`
  - batching, throughput–latency, goodput
* - 3
  - {doc}`lectures/03-chunked-prefill`
  - chunked prefill, prefill–decode interference
* - 4
  - {doc}`lectures/04-pd-disaggregation`
  - prefill–decode disaggregation vs. chunked prefill
* - 5
  - {doc}`lectures/05-hardware-and-parallelism`
  - GPU type, tensor parallelism
* - 6
  - {doc}`lectures/06-cluster-routing`
  - replicas and routing (outline)
* - 7
  - {doc}`lectures/07-agentic-workloads`
  - agentic workloads (outline)
```

:::{warning}
Lectures marked **[TEMP]** are placeholder drafts and will be replaced.
:::

```{toctree}
:hidden:
:caption: Getting Started

getting-started/index
getting-started/run-in-browser
getting-started/run-locally
```

```{toctree}
:hidden:
:caption: Lectures
:maxdepth: 1
:glob:

lectures/*
```

```{toctree}
:hidden:
:caption: Reference

reference/python-api
reference/simulator-knobs
reference/how-the-simulator-works
reference/authoring
```
