# 1. Prefill과 Decode

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

모든 LLM 요청은 성격이 아주 다른 두 단계를 거칩니다:

**Prefill**
: 모델이 prompt 전체를 한 번의 forward pass로 읽으면서, 모든 prompt token의
  key/value (KV) 벡터를 **KV cache**에 써 넣습니다. 그리고 첫 번째 output token을
  내놓으며 끝납니다. 모델이 여러분의 질문을 "읽고" 이해하는 단계입니다.

**Decode**
: 모델이 나머지 token을 하나씩 생성합니다. 각 step은 요청당 *한 개*의 새 token에
  대해서만 forward pass를 하고, KV cache를 재사용합니다. 모델이 여러분의 질문에
  답하는 단계입니다.

사용자 입장에서 이 두 단계는 서로 다른 두 개의 latency로 느껴집니다:

| 지표 | 정의 | 주로 결정하는 것 |
|---|---|---|
| **TTFT** (time to first token) | 도착 → 첫 token | queueing + prefill |
| **TPOT** (time per output token) | 이후 token 사이의 평균 간격 | decode step |
| **E2E latency** | 도착 → 마지막 token | ≈ TTFT + TPOT × (output token 수 − 1) |

TTFT가 중요한 이유는, ChatGPT에 질문을 했을 때 금방 대답이 시작되기를 기대하기
때문입니다. 아무것도 나오지 않으면 고장 난 건 아닌지 의심하게 되죠. TPOT이 중요한
이유는, 일단 대답이 시작되면 최소한 읽는 속도만큼은 빠르게 token이 나와야 하기
때문입니다.

이 강의에서는 세 가지를 모두 simulation으로 측정하고, prefill과 decode가 *왜* 그렇게
다르게 scaling하는지 따져 봅니다.

:::{admonition} 학습 목표
- Simulation을 돌리고 latency 지표를 읽을 수 있다.
- Prompt 길이가 늘면 prefill 시간은 늘어나는데 token당 decode 시간은 거의 늘지 않는
  이유를 설명할 수 있다.
- Prefill과 decode가 각각 compute-bound인지 memory-bound인지 분류하고, 둘 다 기본
  원리에서 추정할 수 있다.
:::
<!-- cell -->
## 준비

이 페이지 오른쪽 위의 **Open in Colab**이나 **Launch Binder**를 누르면 브라우저에서
바로 실행할 수 있습니다. GPU는 필요 없습니다. 이 강의의 simulator는 LLM serving
system을 CPU 위에서 모델링합니다.

이 강의들의 simulation은
[Vidur-Agent](https://github.com/psu-paws/Vidur-Agent) ([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725))를
사용합니다. Vidur-Agent는 Microsoft의 LLM inference simulator인
[Vidur](https://github.com/microsoft/vidur)
([Agrawal et al., MLSys '24](https://arxiv.org/abs/2405.05465))를 확장한 것이고,
여기서는 이 둘을 simulation 엔진으로 씁니다. `llm_systems_wo_gpus`(`lsg`로 import)는
이 강의만을 위해 그 위에 얹은 쓰기 쉬운 wrapper입니다. Simulator를 설치해 주고, 수백
개의 command-line flag를 몇 개의 Python 인자로 바꿔 주고, 결과를 표로 돌려줍니다.

:::{note}
Colab의 무료 환경(RAM 약 12 GB, CPU 코어 2개)에서 돌아가도록, `lsg`는 simulator를
**축소한 설정**을 씁니다. Kernel latency를 요청당 **16,384 token**(prompt + output),
**128개 요청**의 batch, **4,096 token**의 prefill chunk까지만 예측하고, 그 범위를 넘으면
`simulate`가 에러를 냅니다. 이는 이 강의 설정의 한계이지 simulator 자체의 한계가
아닙니다. 원래의 Vidur-Agent는 훨씬 긴 context를 다루고, 훨씬 길고 실제적인 multi-turn
agent trace로 검증되었습니다.
:::

처음 실행할 때는 simulator를 내려받고 의존성을 설치하느라 1분 정도 걸립니다.
<!-- cell -->
## 우리가 simulation할 시스템

**Qwen2.5-32B-Instruct**를 **NVIDIA A100-80GB GPU 두 장**에서 serving하는 상황을
simulation합니다. 파라미터 32.8B는 16-bit precision으로 65.5 GB를 차지하는데, 80 GB
GPU 한 장에 올리면 KV cache 자리가 거의 남지 않습니다. 그래서 *tensor parallelism*
(TP-2)으로 모델을 GPU 두 장에 나누고, 각 GPU가 모든 weight matrix의 절반씩을 가집니다.

| | Qwen2.5-32B-Instruct |
|---|---|
| Layer 수 | 64 |
| Hidden size | 5,120 |
| Attention head (query / KV) | 40 / 8 |
| MLP hidden size | 27,648 |
| 파라미터 | 32.8 B (bf16 기준 65.5 GB) |

이 강의의 모든 simulation이 이 설정을 쓰므로, 한 번만 정의해 둡니다:
<!-- cell -->
## Simulation 돌리기

`lsg.simulate(...)`는 시스템과 workload를 기술하고, simulator를 돌려서 결과를
돌려줍니다. 여기서는 요청 10개(각각 prompt 512 token, output 128 token)가 초당
`qps=0.05`로, 즉 평균 20초에 하나씩 도착합니다. 요청 하나가 약 5초 걸리므로 거의
겹치지 않고, 대부분의 요청이 GPU를 독차지합니다.

새로운 (모델, GPU, parallelism) 조합을 처음 호출할 때는 simulator가 runtime predictor를
학습하느라 15초 정도 걸립니다. 그다음부터는 몇 초면 됩니다.
<!-- cell -->
`r.summary()`는 요청 전체를 집계합니다(p50 = 중앙값, p99 = 99 percentile).
`r.requests`에는 요청 하나당 한 행씩, simulator가 기록한 모든 지표가 들어 있습니다:
<!-- cell -->
위 표의 E2E 공식을 확인해 봅시다. TTFT + TPOT × 127이 측정된 E2E latency와 비슷하게
나와야 합니다.
<!-- cell -->
## Prefill은 prompt 길이에 비례한다

인자 하나만 바꿔 가며 실험하려면 `lsg.sweep(name, values, **other_args)`를 씁니다.
값마다 `simulate`를 한 번씩 호출하고, 값마다 summary 한 행이 담긴, 바로 그릴 수 있는
표를 돌려줍니다. 여기서는 prompt 길이를 바꾸고 output은 32 token으로 고정합니다.
<!-- cell -->
TTFT는 prompt 길이에 **선형으로** 늘어납니다. Prompt token당 약 0.2 ms입니다. 반면
TPOT은 16K token에서도 전혀 움직이지 않습니다. 매 decode step이 prompt 전체의 KV
cache를 읽는데도 말이죠.

## Prefill은 compute-bound이다

$N$개의 token에 대한 forward pass는 대략 $2 \times (\text{파라미터 수}) \times N$번의
부동소수점 연산이 듭니다. Weight 하나당 token 하나마다 곱셈 한 번, 덧셈 한 번이죠.
Weight는 메모리에서 한 번만 읽어서 $N$개 token 전부에 재사용하므로, prompt가 길면
메모리 전송량이 아니라 연산량이 시간을 결정합니다:

$$
\text{TTFT} \gtrsim \frac{2 \times 32.8\times 10^9 \times N}{2\ \text{GPUs} \times 312\ \text{TFLOP/s}}
$$
<!-- cell -->
"compute bound" 숫자는 모든 연산 유닛이 완전히 활용될 때(즉 각 GPU가 실제로 312
TFLOP/s를 내는 경우) prefill이 얼마나 빨리 끝날지를 보여 줍니다. "simulated" 숫자는
실제로 걸린 시간입니다. 둘은 대략 2$\times$ 차이가 나는데, prefill이 GPU의 연산
유닛을 절반 정도만 바쁘게 만든다는 뜻입니다. 실제 kernel에서는 흔한 일입니다. 연산
유닛을 100% 활용하기는 매우 어렵습니다. Prompt가 두 배가 되면 일도 두 배가 되고,
따라서 TTFT도 두 배가 됩니다. (Attention은 $N^2$로 늘어나는 항을 더하지만, 이 정도
context 길이에서는 무시할 만합니다.)

## Decode는 memory-bound이다

Decode step은 요청당 token **하나**를 처리합니다. Weight는 여전히 전부 GPU
메모리(HBM)에서 읽어야 하는데, 이번에는 곱셈-덧셈 한 번에만 쓰입니다. 한 step의
연산량은 $2 \times 32.8\times10^9$ FLOP으로, A100 두 장에서는 약 0.1 ms의 계산량입니다.
하지만 각 GPU는 65.5 GB weight의 절반을 메모리 시스템으로 흘려보내야 합니다:

$$
\text{TPOT} \gtrsim \frac{65.5\ \text{GB} \,/\, 2\ \text{GPUs}}{2.0\ \text{TB/s per GPU}} \approx 16\ \text{ms}.
$$
<!-- cell -->
여기서도 "compute bound"는 모든 연산 유닛이 완전히 활용될 때 한 step이 얼마나 걸릴지를
보여 줍니다. "simulated" 숫자와 비교해 보면 병목이 다른 곳에 있다는 것(즉 대부분의 연산
유닛이 놀고 있다는 것)이 분명합니다. 실제로 decode는 memory-bound입니다. 모델 weight를
메모리에서 읽는 데 시간의 대부분을 쓰고, 그동안 연산 유닛은 weight가 도착하기를 기다리며
놀고 있습니다.

"Memory bound"는 weight를 읽는 동안 각 GPU의 HBM 대역폭이 완전히 활용될 때 한 step이
얼마나 걸릴지를 보여 줍니다. 이 숫자는 "simulated"에 훨씬 가깝고, decode가 정말로
memory-bound임을 확인해 줍니다. 그래도 2$\times$ 정도 차이가 남는 이유는, batch size가
1이면 많은 kernel이 너무 작아 peak 대역폭에 도달하지 못하고, layer마다 고정 비용(kernel
launch, normalization layer, 두 GPU 사이의 통신)이 들기 때문입니다.

TPOT이 prompt 길이를 무시하는 이유도 여기에 있습니다. 16K token prompt의 KV cache는
$2 \times 64\ \text{layer} \times 8\ \text{KV head} \times 128 \times 2\ \text{B} \times 16\text{K} \approx 4\ \text{GB}$
이므로, 65.5 GB의 weight를 읽는 것에 비하면 약간의 오버헤드만 더해질 뿐입니다.
(Prompt가 더 길어지면 이야기가 달라지는데, 뒤에서 다룹니다.)

## Decode는 output 길이에 비례한다

Prompt를 고정하면 E2E latency는 output token 수에 대한 직선이 되고, 그 기울기가
TPOT입니다.
<!-- cell -->
## 시간은 어디에 쓰이나

요청 하나 전체로 보면 어느 단계가 더 중요할까요? 짧은 prompt부터 15K token 문서까지,
그리고 output 64개부터 1,024개까지 여러 모양의 요청을 격자로 simulation하고, 각 요청의
E2E latency를 **prefill**(TTFT)과 **decode**(첫 token 이후 전부)로 나눠 봅니다.
<!-- cell -->
거의 모든 모양에서 **decode가 지배적**입니다. Output token 하나는 약 39 ms가 걸리는데,
이는 prompt token을 약 190개(개당 0.2 ms) prefill하는 것과 비슷합니다. 그래서 4K token
prompt에 64 token 답변이라 해도 시간의 대부분은 decode에 쓰입니다. 15K token 문서에
요약처럼 짧은 64 token 답변을 내는, 꽤 극단적인 경우에만 저울이 prefill 쪽으로 기웁니다.

이것이 LLM serving 연구의 많은 부분(전부는 아닙니다!)이 decode를 겨냥하는 이유입니다.
요청 하나가 시간을 쓰는 곳이자, GPU의 연산 유닛이 놀고 있는 곳이니까요.

## 정리

| | Prefill | Decode (요청 하나) |
|---|---|---|
| Forward pass당 token 수 | prompt 전체 | 1 |
| 병목 | 연산량 (FLOP/s) | 메모리 대역폭 |
| 무엇에 비례하나 | prompt 길이 | output token 수 |
| 사용자가 느끼는 지표 | TTFT | TPOT |

Decode는 GPU의 연산 유닛을 거의 놀립니다. 2강에서는 serving system이 여러 요청을
**batching**해서 이 유닛들을 일하게 만드는 방법을 봅니다.

## 연습문제

:::{admonition} 직접 해보기
:class: exercise
각 문제는 위의 코드를 바꿔서 다시 돌려 보는 것입니다. 돌리기 *전에* 먼저 예상을
적어 보세요.

1. **GPU를 늘리면.** `SYSTEM`을 `tensor_parallel=4`로 바꾸고 "Prefill은 prompt 길이에
   비례한다"부터 "Decode는 memory-bound이다"까지의 셀을 다시 돌려 보세요(bound 계산의
   `n_gpus = 4`도 함께 바꾸세요). GPU가 두 배면 TTFT와 TPOT은 어떻게 될까요? 어느 쪽이
   예상에서 더 많이 벗어나고, 그 이유는 무엇일까요? (힌트: GPU들은 매 layer마다 부분
   결과를 주고받아야 하는데, decode step에는 그 시간을 가릴 만한 일이 별로 없습니다.)
2. **더 빠른 GPU.** Qwen2.5-32B는 A100에서만 profiling되어 있으므로, GPU 한 장에 올라가는
   모델로 바꿔 봅시다: `dict(model="meta-llama/Meta-Llama-3-8B", device="a100",
   tensor_parallel=1)`. Prompt 4,096 token에 output 128 token인 요청 하나를
   simulation한 뒤, `device`를 `"h100"`으로 바꿔 다시 돌려 보세요. H100은 A100보다
   FLOP/s가 약 3.2배(989 vs. 312 TFLOP/s)지만 메모리 대역폭은 약 1.7배(3.35 vs. 2.0
   TB/s)밖에 되지 않습니다. TTFT와 TPOT이 각각 얼마나 좋아질지 예상하고 확인해 보세요.
3. **언제 prefill이 이기나.** "시간은 어디에 쓰이나"에서, 64 token 답변일 때 prefill과
   decode가 같은 시간이 되는 prompt 길이를 찾아보세요. 먼저 위에서 측정한 prompt token당
   약 0.2 ms와 output token당 약 39 ms로 추정한 다음, 코드의 격자를 바꿔 확인해 보세요.
   (요청당 16,384 token 제한을 잊지 마세요.)
:::

## 이해도 확인

답을 눌러 확인하세요. 틀리면 다시 시도할 수 있습니다.
