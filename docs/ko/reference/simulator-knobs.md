# Simulator 설정

Backend simulator는 전부 command-line flag로 설정합니다(약 300개 전부를 보려면 simulator
디렉터리에서 `python -m vidur.main -h`를 실행하세요). `lsg.simulate(extra={...})`로 그중
무엇이든 그대로 넘길 수 있습니다. Flag는 그것이 설정하는 config 객체별로 묶여 있습니다.
아래 표는 이 강의가 쓰는 flag들과 기본값입니다.

## Cluster와 replica

| Flag | 기본값 | 설명 |
|---|---|---|
| `--replica_config_model_name` | `meta-llama/Llama-2-7b-hf` | simulation할 모델 |
| `--replica_config_device` | `a100` | GPU SKU: `a100`, `h100`, `a40` |
| `--replica_config_network_device` | `a100_pairwise_nvlink` | collective에 쓰는 interconnect profile |
| `--replica_config_tensor_parallel_size` | `1` | pipeline stage당 GPU 수 |
| `--replica_config_num_pipeline_stages` | `1` | pipeline-parallel stage 수 |
| `--cluster_config_num_replicas` | `1` | 동일한 replica 수 |
| `--cluster_config_replica_groups_config` | — | 이종 replica group을 기술하는 JSON (위 설정을 덮어씀) |

## Workload

| Flag | 기본값 | 설명 |
|---|---|---|
| `--synthetic_request_generator_config_num_requests` | `128` | 요청 수 |
| `--length_generator_config_type` | `fixed` | `trace`는 길이 CSV를 재현 (이것을 쓰세요. `fixed`는 동작하지 않습니다) |
| `--trace_request_length_generator_config_trace_file` | — | `num_prefill_tokens,num_decode_tokens[,session_id,turn_id,dep,inter_request_latency,token_ids,...]` 열을 가진 CSV |
| `--interval_generator_config_type` | `poisson` | 도착 과정: `poisson`, `gamma`, `static`, `uniform`, `trace` |
| `--poisson_request_interval_generator_config_qps` | `0.5` | 도착률 |

## Replica scheduler (`vllm_v1`)

| Flag | 기본값 | 설명 |
|---|---|---|
| `--replica_scheduler_config_type` | `sarathi` | `vllm_v1`을 쓰세요. 다른 scheduler는 실패합니다 |
| `--vllm_v1_scheduler_config_chunk_size` | `4096` | iteration당 token 예산 (chunked prefill). `llm_systems_wo_gpus`는 512를 씁니다 |
| `--vllm_v1_scheduler_config_batch_size_cap` | `128` | iteration당 최대 요청 수 |
| `--vllm_v1_scheduler_config_session_priority` | 꺼짐 | 먼저 시작한 session을 우선 (agentic workload용) |
| `--vllm_v1_scheduler_config_sjf_priority` | 꺼짐 | cache hit를 뺀 남은 prompt token 기준 shortest-job-first 정렬 |
| `--vllm_v1_scheduler_config_sjf_active_priority` | 꺼짐 | 도착할 때만이 아니라 매 라운드마다 대기 queue를 다시 정렬 |
| `--vllm_v1_scheduler_config_sjf_starvation_timeout` | `0` | 이 초만큼 기다린 요청을 앞으로 올림 (0이면 안 함) |

## KV cache

| Flag | 기본값 | 설명 |
|---|---|---|
| `--cache_config_block_size` | `16` | KV block당 token 수 |
| `--cache_config_enable_prefix_caching` | 꺼짐 | 동일한 prefix의 KV 재사용 (block 내용의 사슬 해시). trace에 `token_ids`나 `block_hash_ids`가 필요합니다 |
| `--cache_config_num_blocks` | 자동 | GPU 메모리에서 유도하는 대신 KV cache 용량을 직접 지정 |
| `--cache_config_memory_margin_fraction` | `0.1` | activation용으로 남겨 두는 GPU 메모리 비율 |

## Global scheduler (router)

`--global_scheduler_config_type`: `random`, `round_robin`, `lor`, `lop`,
`sticky_round_robin`, `sticky_lor`, `load_aware`, `dynamo_kv`, …

## Execution-time predictor

Simulator가 runtime model을 어떻게 학습할지 제어합니다. 이 강의는 메모리를 1 GB 아래로
유지하려고 prediction grid를 덮어씁니다(`llm_systems_wo_gpus.py`의 `LITE_GRID`).

| Flag (접두사 `--random_forest_execution_time_predictor_config_`) | 원래 기본값 | 강의 설정값 |
|---|---|---|
| `prediction_max_tokens_per_request` | `600000` | `16384` |
| `prediction_max_batch_size` | `512` | `128` |
| `prediction_max_prefill_chunk_size` | `4096` | `4096` |
| `k_fold_cv_splits` | `10` | `2` |
| `num_estimators` / `max_depth` / `min_samples_split` | grid search | `50` / `16` / `2` |
| `cache_dir` | `cache` | `~/.llm-systems-wo-gpus/predictor_cache` |

## Replica group

Replica-groups JSON은 "동일한 replica N개"가 아닌 cluster를 기술합니다. GPU 종류가 섞여
있거나, 모델이 섞여 있거나, prefill/decode pool로 나뉜 경우죠.

```json
{
  "replica_groups": [
    {
      "role": "prefill",                 // "agg"(기본), "prefill", "decode"
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

`replica_groups_pools`는 prefill group과, 그 group이 요청을 넘길 수 있는 decode group을
짝지어 줍니다. `cross_node`는 KV 전송에 쓸 노드 간 링크를 고릅니다. 예시 설정은
simulator의 `data/replica_groups_configs/` 디렉터리에 있습니다.
