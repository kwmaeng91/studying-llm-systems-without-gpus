# LLM Systems Without GPUs

**A hands-on lecture series on LLM serving systems that you can follow on a laptop.**

The labs use [Vidur-Agent](https://github.com/psu-paws/Vidur-Agent), a
**CPU-only simulator** of LLM inference clusters. It predicts each operation's
runtime from profiles measured on real A100 and H100 GPUs, so you can experiment
with hardware you don't have.

Each lecture pairs an explanation with a runnable notebook. Click
**Open in Colab** or **Launch Binder** at the top of any lab page to run it in
your browser, with nothing to install. Start with {doc}`lectures/00-example`.

```{list-table}
:header-rows: 1
:widths: 6 40 54

* - #
  - Lecture
  - Topics
* - 0
  - {doc}`lectures/00-example`
  - check your setup; run a first simulation
* - 1
  - **[TEMP]** {doc}`lectures/01-anatomy-of-a-request`
  - prefill vs. decode, TTFT/TPOT
* - 2
  - **[TEMP]** {doc}`lectures/02-batching-and-load`
  - batching, throughput–latency, goodput
* - 3
  - **[TEMP]** {doc}`lectures/03-chunked-prefill`
  - chunked prefill, prefill–decode interference
* - 4
  - **[TEMP]** {doc}`lectures/04-hardware-and-parallelism`
  - GPU type, tensor parallelism
* - 5
  - **[TEMP]** {doc}`lectures/05-cluster-routing`
  - replicas and routing (outline)
* - 6
  - **[TEMP]** {doc}`lectures/06-pd-disaggregation`
  - prefill–decode disaggregation (outline)
* - 7
  - **[TEMP]** {doc}`lectures/07-agentic-workloads`
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

reference/lab-api
reference/simulator-knobs
reference/how-the-simulator-works
reference/authoring
```
