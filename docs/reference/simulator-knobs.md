# Simulator Configuration

The backend simulator is configured entirely through command-line flags (run
`python -m vidur.main -h` inside the simulator's directory for all ~300 of them).
`lsg.simulate(extra={...})` passes any of them through. Flags are grouped by the config object
they set. The tables below list the ones the course uses, with their defaults.

## Cluster and replica

| Flag | Default | Description |
|---|---|---|
| `--replica_config_model_name` | `meta-llama/Llama-2-7b-hf` | model to simulate |
| `--replica_config_device` | `a100` | GPU SKU: `a100`, `h100`, `a40` |
| `--replica_config_network_device` | `a100_pairwise_nvlink` | interconnect profile used for collectives |
| `--replica_config_tensor_parallel_size` | `1` | GPUs per pipeline stage |
| `--replica_config_num_pipeline_stages` | `1` | pipeline-parallel stages |
| `--cluster_config_num_replicas` | `1` | identical replicas |
| `--cluster_config_replica_groups_config` | — | JSON describing heterogeneous replica groups (overrides the above) |

## Workload

| Flag | Default | Description |
|---|---|---|
| `--synthetic_request_generator_config_num_requests` | `128` | number of requests |
| `--length_generator_config_type` | `fixed` | `trace` replays a CSV of lengths (use this; `fixed` is broken) |
| `--trace_request_length_generator_config_trace_file` | — | CSV with `num_prefill_tokens,num_decode_tokens[,session_id,turn_id,...]` |
| `--interval_generator_config_type` | `poisson` | arrival process: `poisson`, `gamma`, `static`, `uniform`, `trace` |
| `--poisson_request_interval_generator_config_qps` | `0.5` | arrival rate |

## Replica scheduler (`vllm_v1`)

| Flag | Default | Description |
|---|---|---|
| `--replica_scheduler_config_type` | `sarathi` | use `vllm_v1`; the other schedulers fail |
| `--vllm_v1_scheduler_config_chunk_size` | `4096` | token budget per iteration (chunked prefill); `llm_systems_wo_gpus` uses 512 |
| `--vllm_v1_scheduler_config_batch_size_cap` | `128` | max requests per iteration |
| `--vllm_v1_scheduler_config_session_priority` | off | prioritize earlier sessions (agentic workloads) |
| `--vllm_v1_scheduler_config_sjf_priority` | off | shortest-job-first ordering |

## KV cache

| Flag | Default | Description |
|---|---|---|
| `--cache_config_block_size` | `16` | tokens per KV block |
| `--cache_config_enable_prefix_caching` | off | reuse KV of identical prefixes (hash of block contents) |
| `--cache_config_memory_margin_fraction` | `0.1` | fraction of GPU memory reserved for activations |

## Global scheduler (router)

`--global_scheduler_config_type`: `random`, `round_robin`, `lor`, `lop`,
`sticky_round_robin`, `sticky_lor`, `load_aware`, `dynamo_kv`, …

## Execution-time predictor

These control how the simulator fits its runtime models. The course overrides the
prediction grid to keep memory under 1 GB (`LITE_GRID` in `llm_systems_wo_gpus.py`).

| Flag (prefix `--random_forest_execution_time_predictor_config_`) | Upstream default | Course value |
|---|---|---|
| `prediction_max_tokens_per_request` | `600000` | `16384` |
| `prediction_max_batch_size` | `512` | `128` |
| `prediction_max_prefill_chunk_size` | `4096` | `4096` |
| `k_fold_cv_splits` | `10` | `2` |
| `num_estimators` / `max_depth` / `min_samples_split` | grid search | `50` / `16` / `2` |
| `cache_dir` | `cache` | `~/.llm-systems-wo-gpus/predictor_cache` |

## Replica groups

A replica-groups JSON describes clusters that are not "N identical replicas": mixed
GPU types, mixed models, or prefill/decode pools.

```json
{
  "replica_groups": [
    {
      "role": "prefill",                 // "agg" (default), "prefill" or "decode"
      "num_replicas": 1,
      "replica_config": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "tensor_parallel_size": 2,
        "device": "a100",
        "network_device": "a100_dgx",
        "pd_disaggregation": 1
      },
      "replica_scheduler_config": {"type": "vllm_v1", "batch_size_cap": 128}
    },
    { "role": "decode", "...": "..." }
  ],
  "replica_groups_pools": [
    {"prefill": [0], "decode": [1], "cross_node": false}
  ]
}
```

`replica_groups_pools` pairs prefill groups with the decode groups they may hand
requests to. `cross_node` selects the inter-node link for the KV transfer. Example
configs are in the simulator's `data/replica_groups_configs/` directory.
