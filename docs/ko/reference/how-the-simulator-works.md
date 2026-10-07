# Simulator는 어떻게 동작하나

이 simulator는 **discrete-event simulator**입니다. 신경망을 전혀 돌리지 않습니다. 실제
serving system의 각 step이 *얼마나 걸릴지*만 알면 됩니다.

## 1. 실제 GPU에서 한 번 profiling

저자들은 각 모델의 구성 요소(QKV/O projection, MLP, attention의 prefill과 decode, RMSNorm,
all-reduce, send/recv 등)를 실제 A100, H100, A40에서 batch size, sequence 길이, TP 차수의
격자에 걸쳐 돌렸습니다. 그 측정값이 `data/profiling/`(약 600 MB의 CSV)에 들어 있어서, 그
뒤로는 아무도 GPU가 필요 없습니다.

## 2. Runtime predictor 학습

시작할 때 simulator는 연산마다 random-forest regressor를 학습해서 (token 수, batch size,
KV 길이 등)을 실행 시간으로 매핑하고, simulation이 필요로 할 범위에 대한 조회표를 미리
계산합니다. 새로운 (모델, GPU, TP) 조합을 처음 simulation할 때 약 15초 멈추는 것이 이
때문입니다. 결과는 캐시됩니다.

:::{warning}
Random forest는 **내삽은 하지만 외삽은 못 합니다**. Profiling된 범위를 벗어나면(예: 4K
context 모델에 8K token을 묻는 경우) 경계값을 돌려줍니다. `lsg.catalog()`가 모델별
profiling 한계를 보여 줍니다.
:::

### 이 강의의 축소 설정

원래 조회표는 요청당 600,000 token과 512개 batch까지 다루는데, 10 GB가 넘는 RAM이
필요합니다. 무료 Colab과 Binder 머신에서 돌리려고 `lsg`는 이를
(`llm_systems_wo_gpus.py`의 `LITE_GRID`) **요청당 16,384 token**, **batch 128개**,
**prefill chunk 4,096 token**으로 줄입니다. 이 한계는 simulator가 아니라 이 강의 설정의
것입니다. 원래의 Vidur-Agent는 훨씬 긴 context를 다루고, 훨씬 길고 실제적인 multi-turn
agent trace로 검증되었습니다. 메모리가 더 많은 머신에서는 `LITE_GRID` 값을 올리면 한계가
풀립니다.

## 3. Workload 재현

Simulator는 이벤트 큐를 유지합니다. 요청 도착, batch 시작, batch 종료, KV 전송 같은
것들이죠. Batch 경계마다 replica scheduler(여기서는 `vllm_v1`)가 vLLM이 하는 것과 똑같이
다음 batch를 고릅니다. Decode를 먼저, 그다음 token 예산 안에서 prefill을, 비어 있는 KV
cache block의 제약을 받으면서요. Batch의 소요 시간은 layer들에 걸친 예측 연산 시간의 합에
통신 시간을 더한 것입니다. 그다음 시간이 batch가 끝나는 시점으로 건너뜁니다.

Step을 실행하는 대신 예측하기 때문에, 바쁜 GPU cluster의 몇 분을 노트북 CPU에서 몇 초 만에
simulation할 수 있습니다.

## 얼마나 정확한가

저자들은 여러 모델과 GPU에서 실제 vLLM 배포 대비 요청 latency 오차가 9% 미만이라고
보고하고 ([Agrawal et al., MLSys '24](https://arxiv.org/abs/2405.05465)), agentic 확장은
실제 agent trace로 검증했습니다
([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725)).
절댓값은 괜찮은 추정치로 받아들이고, **추세**(어떤 구성이 더 나은지, knee가 어디인지)를
핵심 교훈으로 삼으세요.
