# Python API: `llm_systems_wo_gpus`

모든 실습이 `lsg`로 import하는 패키지입니다:
[`labs/llm_systems_wo_gpus.py`](https://github.com/kwmaeng91/studying-llm-systems-without-gpus/blob/main/labs/llm_systems_wo_gpus.py).
Backend simulator의 command line을 만들고 그 출력을 파싱할 뿐이므로, 여기 있는 것은 전부
simulator CLI로도 할 수 있습니다({doc}`simulator-knobs`).

## `setup()`

Simulator를 `$LSG_BACKEND_DIR`(기본값 `~/.llm-systems-wo-gpus/backend`)에 내려받고, 빠진
의존성을 설치합니다. 여러 번 호출해도 안전합니다.

## `simulate(**kwargs) → Result`

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` | `"meta-llama/Llama-2-7b-hf"` | Hugging Face 모델 이름. `catalog()`에 있어야 합니다 |
| `device` | `"a100"` | `a100`, `h100`, `a40` |
| `network_device` | 자동 | all-reduce에 쓰는 interconnect profile. 기본은 `<device>_dgx` (TP ≤ 8) |
| `tensor_parallel` | `1` | replica당 GPU 수 |
| `num_replicas` | `1` | router 뒤의 동일한 replica 수 |
| `qps` | `2.0` | Poisson 도착률 (req/s). `None`이면 모든 요청이 t=0에 도착 |
| `num_requests` | `100` | simulation할 요청 수 |
| `prefill_tokens`, `decode_tokens` | `512`, `128` | 정수(모든 요청 동일) 또는 요청별 수열 |
| `trace` | `None` | trace CSV 경로. 위 두 인자를 덮어씀 |
| `arrival_times` | `None` | 각 요청의 정확한 도착 시각(초). `qps`를 덮어씀 |
| `scheduler` | `"vllm_v1"` | replica scheduler (지금은 `vllm_v1`만 동작) |
| `batch_size_cap` | `128` | batch당 최대 요청 수 |
| `chunk_size` | `512` | step당 token 예산 (chunked prefill) |
| `global_scheduler` | `"round_robin"` | replica 사이의 router 정책 |
| `prefix_caching` | `False` | 자동 prefix (KV) caching 켜기 |
| `kv_blocks` | 자동 | replica의 KV cache 용량을 16 token block 단위로 직접 지정 |
| `max_tokens` | `16384` | runtime predictor가 학습된 최대 요청 길이(이자 batch된 최대 context). 올리면 학습 시간과 메모리가 늘어남 |
| `replica_groups` | `None` | 이종 / PD-disaggregated cluster를 위한 dict 또는 JSON 경로 |
| `seed` | `42` | 도착 생성용 난수 seed |
| `keep_steps` | `False` | 모든 forward pass를 기록. `Result.steps`로 읽음 |
| `extra` | `None` | `{"--아무_simulator_flag": 값}`을 그대로 전달 |
| `verbose` | `False` | command line과 simulator 로그 출력 |

## `Result`

| 속성 | 타입 | 설명 |
|---|---|---|
| `requests` | DataFrame | 요청당 한 행, simulator가 기록한 모든 지표 (아래 참고) |
| `summary()` | Series | TTFT/TPOT p50/p99, E2E p50, queueing p50, throughput, 그리고 실행 전체의 총 실행 시간 |
| `ttft`, `tpot`, `e2e` | Series | 요청별 latency (초) |
| `kv_cache_tokens` | int | replica 하나의 KV cache 용량 (token 수) |
| `cache` | DataFrame | replica별: 요구된 prompt token 수, prefix cache에서 제공된 token 수, KV cache hit rate, evict된 block 수 |
| `cache_hit_rate` | float | KV cache hit rate. 전체 prompt token 중 prefix cache에서 제공된 비율 |
| `steps` | DataFrame | forward pass당 한 행, `replica`별 실행 순서대로: prefill/decode token 수, batch size, `batch_execution_time` (`keep_steps=True` 필요) |
| `cdf(metric)` | DataFrame | `batch_size`나 `batch_num_tokens` 같은 batch 단위 지표의 CDF |
| `out_dir` | Path | simulator의 원본 출력 디렉터리 |
| `config` | dict | 이번 실행이 쓴 flag들 |

`requests`의 주요 열:

| 열 | 의미 |
|---|---|
| `request_arrived_at` | 도착 시각 (초) |
| `request_num_prefill_tokens`, `request_num_decode_tokens` | 요청 길이 |
| `request_num_prefill_tokens_cached` | prefix cache에서 제공된 prefill token 수 |
| `prefill_e2e_time` | **TTFT**: 도착 → prefill 종료 |
| `decode_time_execution_plus_preemption_normalized` | **TPOT**: decode 시간 / output token 수 |
| `request_e2e_time` | 도착 → 마지막 token |
| `request_scheduling_delay` | 처음 scheduling되기까지 기다린 시간 |
| `request_preemption_time`, `request_num_restarts` | preemption으로 잃은 시간 |
| `replica` | 어느 replica가 처리했는지 (PD disaggregation에서는 decode replica. `prefill_replica`와 `decode_replica`가 둘 다 알려줌) |

## `sweep(param, values, **kwargs) → DataFrame`

값마다 `simulate(**{param: v}, **kwargs)`를 호출하고 summary를 쌓아, `param`으로 색인된
표를 돌려줍니다.

## `make_trace(prefill_tokens, decode_tokens, n, name=None, **columns) → Path`

`simulate(trace=...)`가 재현할 수 있는 trace CSV를 씁니다. `prefill_tokens`와
`decode_tokens`는 정수이거나 길이 `n`의 수열입니다. 키워드 인자들은 각각 길이 `n`의
수열로, 여러 turn짜리 session을 기술합니다:

| 인자 | 의미 |
|---|---|
| `token_ids` | 요청의 prompt **와** output token id (`len == prefill + decode`). prefix cache 매칭에 필요 |
| `session_id`, `turn_id` | 요청이 어느 대화에 속하고 그 안에서 몇 번째인지 |
| `dep` | 이 요청이 풀리기 전에 모두 끝나야 하는 turn id들 |
| `think_time` | 그 풀림과 도착 사이의 초 (tool 호출이나 사용자가 읽는 시간) |
| `request_id` | `Result.requests`에서 이 요청이 갖는 id |
| `block_size` | 해시에 쓰는 KV block 크기 (기본 16) |

## 기록된 agent trace

`gaia_sessions(num_sessions=20, max_tokens=None, seed=0) → DataFrame`은
[GAIATrace](https://github.com/psu-paws/Vidur-Agent)로 기록된 실제 multi-turn agent
session을 불러옵니다(simulator와 함께 내려받습니다). LLM 요청당 한 행이고 열은 `session`,
`turn`, `role`, `dep`, `num_prefill_tokens`, `num_decode_tokens`, `tool_time`,
`token_ids`입니다. `max_tokens`보다 긴 요청이 들어 있는 session은 건너뜁니다.

`gaia_trace(sessions) → Path`는 그 행들을 `simulate`용 trace CSV로 씁니다.

`decode(token_ids) → str`은 기록된 token id를 다시 텍스트로 바꿉니다(처음 쓸 때
`tiktoken`을 설치합니다).

## `quiz(questions) → HTML`

즉시 피드백이 나오는 객관식 문제를 그립니다. 각 문제는
`{"q": str, "options": [str, ...], "answer": int, "explain": str}` 형태이고, `answer`는
정답 선택지의 0부터 시작하는 인덱스입니다. 강의를 추가하는 방법은 저장소의
[`docs/reference/authoring.md`](https://github.com/kwmaeng91/studying-llm-systems-without-gpus/blob/main/docs/reference/authoring.md)를
보세요.

## `catalog() → DataFrame`

Profiling 데이터가 있는 (device, model) 조합 전부와, 지원되는 TP 차수, 그리고 profiling된
최대 context 길이와 batch size를 보여 줍니다.
