# How the Simulator Works

Vidur is a **discrete-event simulator**. It never runs a neural network. It
only needs to know *how long* each step of a real serving system would take.

## 1. Profile once, on real GPUs

The authors ran each model's building blocks (QKV/O projections, MLP, attention
prefill and decode, RMSNorm, all-reduce, send/recv, ...) on real A100s, H100s, and
A40s over a grid of batch sizes, sequence lengths, and TP degrees. The measurements
ship in `data/profiling/` (~600 MB of CSVs), so nobody needs a GPU after that.

## 2. Fit runtime predictors

On startup, Vidur fits a random-forest regressor per operation that maps
(tokens, batch size, KV length, ...) to runtime, then precomputes a lookup table
over the range the simulation might need. This is the ~15 s pause the first time
you simulate a new (model, GPU, TP). The result is cached.

:::{warning}
Random forests **interpolate but do not extrapolate**. Past the profiled range
(e.g. a 4K-context model asked about 8K tokens) they return the value at the
edge. `vl.catalog()` shows each model's profiled limits.
:::

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

The original Vidur paper (Agrawal et al., MLSys 2024) reports < 9% error in
request latency against real vLLM deployments across several models and GPUs.
Vidur-Agent's validation against real agent traces is in `data/ground_truth/`
and the IISWC'26 paper. Treat the absolute numbers as good estimates, and the
**trends** (which configuration is better, where the knee is) as the main lesson.
