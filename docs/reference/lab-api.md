# `vidur_lab` API

The helper module used by every lab: [`labs/vidur_lab.py`](https://github.com/kwmaeng91/studying-llm-systems-without-gpus/blob/main/labs/vidur_lab.py).
It only builds a `python -m vidur.main …` command line and parses the output, so
everything here can also be done with the raw simulator CLI.

## `setup()`

Clones Vidur-Agent into `$VIDUR_HOME` (default `~/vidur-agent`) and installs any
missing dependencies. It is safe to call repeatedly.

## `simulate(**kwargs) → Result`

| Argument | Default | Meaning |
|---|---|---|
| `model` | `"meta-llama/Llama-2-7b-hf"` | Hugging Face model name; must appear in `catalog()` |
| `device` | `"a100"` | `a100`, `h100`, or `a40` |
| `network_device` | auto | interconnect profile for all-reduce; `<device>_dgx` (TP ≤ 8) by default |
| `tensor_parallel` | `1` | GPUs per replica |
| `num_replicas` | `1` | identical replicas behind the router |
| `qps` | `2.0` | Poisson arrival rate (req/s). `None` = all requests at t=0 |
| `num_requests` | `100` | requests to simulate |
| `prefill_tokens`, `decode_tokens` | `512`, `128` | int (all requests identical) or a sequence per request |
| `trace` | `None` | path to a trace CSV; overrides the two arguments above |
| `scheduler` | `"vllm_v1"` | replica scheduler (only `vllm_v1` works in Vidur-Agent today) |
| `batch_size_cap` | `128` | max requests per batch |
| `chunk_size` | `512` | token budget per step (chunked prefill) |
| `global_scheduler` | `"round_robin"` | router policy across replicas |
| `prefix_caching` | `False` | enable automatic prefix (KV) caching |
| `replica_groups` | `None` | dict or JSON path for heterogeneous / PD-disaggregated clusters |
| `seed` | `42` | random seed for arrivals |
| `extra` | `None` | `{"--any_vidur_flag": value}` passed through verbatim |
| `verbose` | `False` | print the command line and simulator log |

## `Result`

| Attribute | Type | Description |
|---|---|---|
| `requests` | DataFrame | one row per request, every metric Vidur records (see below) |
| `summary()` | Series | TTFT/TPOT p50/p99, E2E p50, queueing p50, throughput, makespan |
| `ttft`, `tpot`, `e2e` | Series | per-request latencies in seconds |
| `kv_cache_tokens` | int | KV-cache capacity of one replica, in tokens |
| `cdf(metric)` | DataFrame | CDF of a batch-level metric such as `batch_size` or `batch_num_tokens` |
| `out_dir` | Path | raw simulator output directory |
| `config` | dict | flags the run used |

Key columns of `requests`:

| Column | Meaning |
|---|---|
| `request_arrived_at` | arrival time (s) |
| `request_num_prefill_tokens`, `request_num_decode_tokens` | request length |
| `request_num_prefill_tokens_cached` | prefill tokens served from the prefix cache |
| `prefill_e2e_time` | **TTFT**: arrival → end of prefill |
| `decode_time_execution_plus_preemption_normalized` | **TPOT**: decode time / output tokens |
| `request_e2e_time` | arrival → last token |
| `request_scheduling_delay` | time spent waiting before first scheduled |
| `request_preemption_time`, `request_num_restarts` | time lost to preemption |
| `replica` | which replica served it |

## `sweep(param, values, **kwargs) → DataFrame`

Calls `simulate(**{param: v}, **kwargs)` for each value and stacks the summaries,
indexed by `param`.

## `make_trace(prefill_tokens, decode_tokens, n, name=None) → Path`

Writes a two-column trace CSV that `simulate(trace=...)` can replay.

## `catalog() → DataFrame`

Every (device, model) pair with profiling data, its supported TP degrees, and
the largest context length and batch size that were profiled.
