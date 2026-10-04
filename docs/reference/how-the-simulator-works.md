# How the Simulator Works

The simulator is a **discrete-event simulator**. It never runs a neural network. It
only needs to know *how long* each step of a real serving system would take.

## 1. Profile once, on real GPUs

The authors ran each model's building blocks (QKV/O projections, MLP, attention
prefill and decode, RMSNorm, all-reduce, send/recv, ...) on real A100s, H100s, and
A40s over a grid of batch sizes, sequence lengths, and TP degrees. The measurements
ship in `data/profiling/` (~600 MB of CSVs), so nobody needs a GPU after that.

## 2. Fit runtime predictors

On startup, the simulator fits a random-forest regressor per operation that maps
(tokens, batch size, KV length, ...) to runtime, then precomputes a lookup table
over the range the simulation might need. This is the ~15 s pause the first time
you simulate a new (model, GPU, TP). The result is cached.

:::{warning}
Random forests **interpolate but do not extrapolate**. Past the profiled range
(e.g. a 4K-context model asked about 8K tokens) they return the value at the
edge. `lsg.catalog()` shows each model's profiled limits.
:::

### The course's stripped-down configuration

The full lookup table covers up to 600,000 tokens per request and batches of
512, which needs more than 10 GB of RAM. To run on free Colab and Binder
machines, `lsg` shrinks it (`LITE_GRID` in `llm_systems_wo_gpus.py`) to
**16,384 tokens per request**, **batches of 128**, and **prefill chunks of 4,096
tokens**. These limits belong to the course setup, not to the simulator: the full
Vidur-Agent handles far longer contexts and was validated against much longer,
real multi-turn agent traces. On a machine with more memory, raise the
`LITE_GRID` values to lift them.

## 3. Replay the workload

The simulator keeps an event queue: request arrivals, batch starts, batch ends,
KV transfers. At each batch boundary the replica scheduler (here `vllm_v1`) picks
the next batch exactly as vLLM would: decodes first, then prefills within the
token budget, subject to free KV-cache blocks. The batch's duration is the sum of
predicted operator times across layers, plus communication. Time then jumps
forward to the batch's end.

Because steps are predicted rather than executed, simulating minutes of a busy
GPU cluster takes seconds on a laptop CPU.

## How accurate is it?

Its authors report < 9% error in request latency against real vLLM deployments
across several models and GPUs (Agrawal et al., MLSys 2024), and validate the
agentic extensions against real agent traces in the IISWC'26 paper. Treat the absolute numbers as good estimates, and the
**trends** (which configuration is better, where the knee is) as the main lesson.
