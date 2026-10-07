# 2. Batching, 부하, 그리고 throughput–latency trade-off

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

1강에서 혼자 돌아가는 decode step은 **memory-bound**임을 보았습니다. 각 GPU가 token
하나를 만들려고 65.5 GB weight의 절반을 흘려보내는 동안, 연산 유닛은 거의 놀고
있었죠. 모든 serving system은 이를 **batching**으로 해결합니다. 한 번의 forward
pass가 여러 요청을 동시에 처리하므로, 메모리에서 읽은 weight 하나를 그 요청 전부가
함께 씁니다.

이 강의에서는 두 가지를 묻습니다:

1. Batching이 TTFT와 TPOT에 어떤 영향을 주고, throughput은 얼마나 늘어나는가?
2. 요청이 무작위 시점에 도착할 때, latency가 터지기 전까지 시스템은 얼마나 많은 부하를
   감당할 수 있는가?

:::{admonition} 학습 목표
- Prefill을 batching하면 TTFT가 선형으로 늘어나는데 decode를 batching하면 TPOT이 거의
  그대로인 이유를 설명할 수 있다.
- Batching의 한계가 언제 드러나는지 설명할 수 있다.
- 부하가 늘 때 tail latency (p99)가 중앙값 (p50)보다 먼저 나빠지는 이유를 설명하고,
  goodput을 정의할 수 있다.
:::
<!-- cell -->
## 준비

{doc}`01-prefill-and-decode`와 같은 설정입니다. A100 두 장 위의
Qwen2.5-32B-Instruct.
<!-- cell -->
:::{warning}
**비현실적이지만 교육적인 설정: prefill만 하는 요청과 decode만 하는 요청.**

실제 요청에는 prompt와 output이 둘 다 있고, 실제 scheduler는 이들을 섞기도 합니다
(3강에서 다룹니다). 다만 prefill과 decode를 섞으면 추론이 복잡해지고 동작이 시스템마다
달라집니다. 이 강의에서는 교육 목적으로 단순화해서, 모든 요청을 prefill만 하거나
decode만 하도록 만듭니다:

- **prefill만**: 보통의 prompt에 output token은 **1개**뿐 (`decode_tokens=1`);
- **decode만**: **1 token**짜리 prompt (`prefill_tokens=1`)에 보통 길이의 output.

실제 workload 중에 이렇게 생긴 것은 없습니다.
:::
<!-- cell -->
## Prefill batching하기

512 token prompt를 가진 prefill 전용 요청 $B$개가 동시에 도착하면 어떻게 될까요?
Scheduler는 한 번의 forward pass가 처리할 수 있는 token 수에 상한을 둡니다
(`chunk_size`, 3강에서 제대로 다룰 knob입니다). 여기서는 $B$개의 prompt가 **한 번의**
forward pass에 모두 들어가도록 16,384로 올립니다.
<!-- cell -->
`qps=None`은 모든 요청이 시각 0에 도착한다는 뜻입니다. Batch 안의 모든 요청은 공유하는
forward pass가 끝나는 바로 그 순간에 첫 token을 받습니다.
<!-- cell -->
TTFT는 batch size에 **선형으로** 늘어납니다. 요청이 하나 늘 때마다 약 80 ms씩
더해지죠. Throughput은 (~30% 정도로) 조금밖에 좋아지지 않는데, 그마저도 대부분 요청
1개에서 4개로 갈 때 생깁니다. 512 token prompt 하나로는 GPU를 다 채우지 못하기
때문입니다. 1강의 결론에서 따라오는 결과입니다. Prefill은 이미 **compute-bound**입니다.
Prompt $B$개의 batch는 연산량이 그냥 $B$배이고, GPU는 이미 연산하느라 바빴으므로 얻을
것이 별로 없습니다.

## Decode batching하기

이번에는 decode 전용 요청 $B$개(각각 output 256 token)가 함께 도착하면 어떻게 되는지
봅시다. 매 forward pass가 $B$개 요청 각각에 대해 token 하나씩을 만들어 냅니다.
<!-- cell -->
정반대의 그림입니다. 요청 1개에서 128개까지 TPOT은 약 40 ms에 머무는데, throughput은
100배 넘게 늘어납니다. Decode는 **memory-bound**입니다. 어차피 매 step마다 weight를
전부 읽어야 하고, 같은 batch에 요청이 더 들어와도 그 weight를 거의 공짜로 재사용합니다.
(TPOT 그래프의 작은 요철은 simulator가 학습한 runtime model의 잡음이지, 실제 현상은
아닙니다.)

| | Prefill batching | Decode batching |
|---|---|---|
| Latency | batch size에 거의 선형으로 증가 | 거의 그대로 |
| Throughput | 거의 좋아지지 않음 | batch size에 거의 선형으로 증가 |
| 이유 | 이미 compute-bound | memory-bound: weight를 batch가 공유 |

:::{note}
Decode를 더 많이 batching하지 못하게 막는 것은 무엇일까요? 실행 중인 모든 요청이 자기
KV cache를 GPU 메모리에 들고 있습니다. 여기서 token 하나의 KV cache는
$2 \times 64\ \text{layer} \times 8\ \text{KV head} \times 128 \times 2\ \text{B} = 256$ KB이고,
weight를 올리고 나면 GPU 두 장에 약 350,000 token 분량의 자리가 남습니다. 256 token짜리
요청이라면 scheduler의 상한인 동시 실행 요청 128개가 먼저 걸립니다. 대화가 길어지면
KV cache 메모리가 한계가 됩니다.
:::
<!-- cell -->
## 부하: 시간에 걸쳐 도착하는 요청

실제 서비스는 요청을 깔끔한 batch 단위로 받지 않습니다. 요청은 무작위 시점에 도착하고,
scheduler는 자리가 나는 대로 새 요청을 실행 중인 batch에 넣습니다(*continuous
batching*). 도착을 비율 `qps`(초당 요청 수)의 **Poisson process**로 모델링하고, 여전히
output 256 token의 decode 전용 요청으로 부하를 쓸어 봅니다.
<!-- cell -->
초당 약 10–11 요청에서 뚜렷한 **knee**를 기준으로 두 구간이 나타납니다:

- **Knee 아래**에서는 throughput이 부하에 선형으로 늘고, TTFT와 TPOT은 거의 평평합니다.
  도착한 요청이 바로 batch에 합류하고, 위에서 본 것처럼 batch에 요청을 더해도 TPOT이
  크게 나빠지지 않기 때문입니다.
- **Knee를 넘으면** throughput은 평평해지고 TTFT는 치솟습니다. 시스템이 batch size를 더
  키울 수 없어서 요청이 queue에서 기다리기 때문입니다. 여기에는 prefill이 전혀 없지만
  (decode 전용 요청이니까요), 기다림만으로도 TTFT가 늘어납니다. 앞선 요청이 끝나서 자리가
  날 때까지 새 요청은 시작할 수 없으니까요.
<!-- cell -->
### p50보다 p99가 먼저 나빠진다

초당 10 요청에서 TTFT 중앙값은 여전히 약 70 ms이지만, p99는 이미 수백 ms까지 올라가
있습니다. Poisson 도착은 **몰려서** 오고, 그 몰림에 걸린 운 나쁜 요청들은 batch가 꽉
찬 상태를 만나 기다리게 됩니다. 그래서 tail latency 목표를 약속하는 서비스는 원래
용량보다 낮은 지점에서 돌려야 합니다.

## Goodput: SLO를 지키는 throughput

TTFT와 TPOT은 둘 다 중요하고, 각각 자기 목표를 지켜야 합니다. 예를 들어 TTFT는 사용자가
지루해하지 않도록 200–500 ms 아래, TPOT은 보통의 읽는 속도를 따라가도록 50–100 ms
아래여야 한다고들 합니다. 이를 한 숫자로 요약하는 흔한 방법이 **goodput**입니다.
service-level objective (SLO)를 여전히 만족하는 가장 높은 부하를 말합니다.
<!-- cell -->
## 정리

| | 무엇을 보았나 |
|---|---|
| Prefill batching | batch가 커지면 latency가 늘고 throughput은 거의 그대로: GPU는 이미 compute-bound였다 |
| Decode batching | latency는 거의 그대로이고 throughput은 거의 선형으로 증가: weight를 한 번 읽어 batch 전체가 쓴다 |
| Continuous batching | batch가 다 모이기를 기다리지 않고, 자리가 나는 대로 요청이 실행 중인 batch에 합류한다 |
| 부하가 올라가면 | 뚜렷한 knee를 기준으로 두 구간. knee를 넘으면 batch를 더 키울 수 없어 요청이 queue에 쌓이고 TTFT가 오르며 throughput은 평평해진다 |
| Batch를 제한하는 것 | KV cache. 실행 중인 요청은 모두 자기 context를 GPU 메모리에 들고 있다 |
| Goodput | SLO를 지키는 부하. 원래 token throughput보다 한참 아래다 |

## 다음 이야기: prefill과 decode 섞기

지금까지는 모든 요청이 prefill 전용이거나 decode 전용이었습니다. 실제 요청에는 둘 다
있으므로, 실행 중인 batch에는 다른 요청들이 decode하는 동안에도 새 prompt가 계속
들어옵니다.

:::{admonition} 생각해 보기
:class: exercise
Prefill과 decode를 같은 batch에 섞으면 어떻게 될까요? 위에서 측정한 값을 떠올리면서,
4,000 token짜리 prompt가 옆에서 decode하고 있는 요청들의 TPOT에 무슨 일을 할지 생각해
보세요.
:::

다음 강의에서 이 질문에 답합니다.

## 연습문제

:::{admonition} 직접 해보기
:class: exercise
각 문제는 위의 코드를 바꿔서 다시 돌려 보는 것입니다. 돌리기 *전에* 먼저 예상을
적어 보세요.

1. **더 짧은 prompt.** "Prefill batching하기"에서 prompt 길이를 512에서 128 token으로
   바꾸고(`prefill_tokens=`와 throughput 계산식 둘 다) 다시 돌려 보세요. Prefill
   batching이 전보다 더 도움이 되나요, 덜 되나요? Prefill이 compute-bound라는 1강의
   내용으로 이유를 설명해 보세요.
2. **Knee는 어디인가.** 동시에 decode할 수 있는 요청은 최대 128개이고(scheduler의
   batch-size 상한), 요청 하나는 TPOT 한 번 정도가 걸리는 step을 `decode_tokens`번
   필요로 합니다. 여기서 출발해, output이 256 token일 때 시스템이 감당할 수 있는 최대
   도착률을 추정하고 부하 sweep의 knee와 비교해 보세요. 그다음 (a) `decode_tokens=128`,
   (b) `batch_size_cap=64`(`lsg.sweep`과 `lsg.simulate`의 인자)일 때의 knee를 예측하고
   sweep을 다시 돌려 확인해 보세요.
3. **더 빡빡한 SLO.** "Goodput"에서 TPOT 목표를 50 ms에서 40 ms로 조이고 다시 돌려
   보세요. Goodput은 어떻게 되고, 10 ms 차이가 왜 그렇게 크게 작용할까요? 부하에 따라
   TPOT p99가 어떻게 자라는지 보세요.
:::

## 이해도 확인

답을 눌러 확인하세요. 틀리면 다시 시도할 수 있습니다.
