# Python API: `llm_systems_wo_gpus`

The package every lab imports as `lsg`:
[`labs/llm_systems_wo_gpus.py`](https://github.com/kwmaeng91/studying-llm-systems-without-gpus/blob/main/labs/llm_systems_wo_gpus.py).
It builds the backend simulator's command line and parses its output, so
everything here can also be done with the raw simulator CLI
({doc}`simulator-knobs`).

## `setup()`

Downloads the simulator into `$LSG_BACKEND_DIR` (default
`~/.llm-systems-wo-gpus/backend`) and installs any missing dependencies. It is safe to call repeatedly.

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
| `arrival_times` | `None` | exact arrival time (s) of each request; overrides `qps` |
| `scheduler` | `"vllm_v1"` | replica scheduler (only `vllm_v1` works today) |
| `batch_size_cap` | `128` | max requests per batch |
| `chunk_size` | `512` | token budget per step (chunked prefill) |
| `global_scheduler` | `"round_robin"` | router policy across replicas |
| `prefix_caching` | `False` | enable automatic prefix (KV) caching |
| `kv_blocks` | auto | override the replica's KV-cache capacity, in 16-token blocks |
| `max_tokens` | `16384` | longest request (and largest batched context) the runtime predictor is fitted for; raising it costs fitting time and memory |
| `replica_groups` | `None` | dict or JSON path for heterogeneous / PD-disaggregated clusters |
| `seed` | `42` | random seed for arrivals |
| `keep_steps` | `False` | record every forward pass, readable as `Result.steps` |
| `extra` | `None` | `{"--any_simulator_flag": value}` passed through verbatim |
| `verbose` | `False` | print the command line and simulator log |

## `Result`

| Attribute | Type | Description |
|---|---|---|
| `requests` | DataFrame | one row per request, every metric the simulator records (see below) |
| `summary()` | Series | TTFT/TPOT p50/p99, E2E p50, queueing p50, throughput, and the run's total execution time |
| `ttft`, `tpot`, `e2e` | Series | per-request latencies in seconds |
| `kv_cache_tokens` | int | KV-cache capacity of one replica, in tokens |
| `cache` | DataFrame | per replica: prompt tokens asked for, tokens served from the prefix cache, KV cache hit rate, blocks evicted |
| `cache_hit_rate` | float | KV cache hit rate: the fraction of all prompt tokens served from the prefix cache |
| `steps` | DataFrame | one row per forward pass, in order within each `replica`: prefill/decode tokens, batch size, `batch_execution_time` (needs `keep_steps=True`) |
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
| `replica` | which replica served it (with PD disaggregation: the decode replica; `prefill_replica` and `decode_replica` give both) |

## `sweep(param, values, **kwargs) → DataFrame`

Calls `simulate(**{param: v}, **kwargs)` for each value and stacks the summaries,
indexed by `param`.

## `make_trace(prefill_tokens, decode_tokens, n, name=None, **columns) → Path`

Writes a trace CSV that `simulate(trace=...)` can replay. `prefill_tokens` and
`decode_tokens` are ints or sequences of length `n`. The keyword arguments, each a
sequence of length `n`, describe multi-turn sessions:

| Argument | Meaning |
|---|---|
| `token_ids` | the request's prompt **and** output token ids (`len == prefill + decode`); required for prefix-cache matching |
| `session_id`, `turn_id` | which conversation a request belongs to, and its position in it |
| `dep` | turn ids that must all finish before this request is released |
| `think_time` | seconds between that release and the arrival (a tool call, or a user reading) |
| `request_id` | the id the request keeps in `Result.requests` |
| `block_size` | KV block size used for hashing (default 16) |

## Recorded agent traces

`gaia_sessions(num_sessions=20, max_tokens=None, seed=0) → DataFrame` loads real
multi-turn agent sessions recorded with
[GAIATrace](https://github.com/psu-paws/Vidur-Agent) (downloaded with the
simulator), one row per LLM request: `session`, `turn`, `role`, `dep`,
`num_prefill_tokens`, `num_decode_tokens`, `tool_time`, `token_ids`. Sessions
containing a request longer than `max_tokens` are skipped.

`gaia_trace(sessions) → Path` writes those rows as a trace CSV for `simulate`.

`decode(token_ids) → str` turns recorded token ids back into text (installs
`tiktoken` on first use).

## `quiz(questions) → HTML`

Renders clickable multiple-choice questions with instant feedback. Each question
is `{"q": str, "options": [str, ...], "answer": int, "explain": str}`, where
`answer` is the 0-based index of the correct option. See
[`docs/reference/authoring.md`](https://github.com/kwmaeng91/studying-llm-systems-without-gpus/blob/main/docs/reference/authoring.md)
in the repository for how to add a lecture.

## `catalog() → DataFrame`

Every (device, model) pair with profiling data, its supported TP degrees, and
the largest context length and batch size that were profiled.
