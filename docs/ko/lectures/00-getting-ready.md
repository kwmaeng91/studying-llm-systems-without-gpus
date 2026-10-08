# 0. 강의를 듣기 전에

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

이 강의 시리즈는 Python, transformer 언어 모델의 동작 방식, 그리고 GPU의 속도를 무엇이
제한하는지에 대한 약간의 배경 지식을 가정합니다. 어느 것도 전문가일 필요는 없습니다.
아래 용어들을 보시고, 잘 모르는 부분은 아래 추천 링크로 배우세요!

## 용어

### 모델

Token
: LLM이 읽고 쓰는 텍스트의 단위입니다. 단어, 단어의 일부, 문장부호일 수 있습니다. 예를 들어
  "Unbelievable!"은 `Un`, `believable`, `!` 와 같이 세 token으로 찢어질 수 있습니다. 영어에서 token 하나는
  평균적으로 단어의 약 3/4입니다. 이 강의의 모든 길이와 속도는 token으로 셉니다.

Tokenizer
: 텍스트를 token ID(정수)의 나열로, 그리고 그 반대로 바꿔 주는 고정된 절차입니다. 모델이
  아는 모든 token의 집합을 *vocabulary*라고 하며, Qwen2.5는 약 152,000개입니다.

Large language model (LLM)
: Token의 나열이 주어지면 다음 token을 예측하는 신경망입니다.
  요즘 LLM은 *transformer*입니다(아래 참고).

Parameter (weight)
: 모델 안에 학습되어 있는 숫자들로, 대부분 큰 행렬로 배열되어 있습니다. 모델 크기는
  파라미터 수로 말합니다. "32B"는 320억 개라는 뜻입니다. 16-bit precision(*fp16* 또는
  *bf16*)으로 저장하면 파라미터 하나가 2바이트입니다.

Layer
: Transformer는 똑같이 생긴 layer를 쌓은 것입니다(Qwen2.5-32B는 64개). 각 layer에는 각
  token이 앞선 token들을 볼 수 있게 해 주는 **attention** 블록과, 각 token을 독립적으로
  변환하는 **MLP**(feed-forward network, *FFN*이라고도 합니다) 블록이 있습니다.

### 모델 돌리기

Inference
: 학습된 모델로 출력을 만들어 내는 일입니다. *training*과 대비되는 말이죠. 이 강의는
  전부 inference에 대한 것입니다.

Forward pass
: 입력 token들을 모든 layer에 한 번 통과시키는 것입니다. 다음 token에 대한 예측이
  나옵니다.

Prompt와 output
: Prompt는 입력 텍스트(사용자의 메시지에 지시문이나 맥락을 더한 것)이고, output은 모델이
  그에 대한 답으로 생성하는 텍스트입니다.

Autoregressive generation
: Token을 하나씩 생성하는 방식입니다. Token 하나를 예측하고, 그것을 입력에 붙이고, 모델을
  다시 돌립니다. 100 token짜리 답변에는 100 step이 듭니다.

KV cache
: 지금까지 처리한 모든 token의 attention **key**와 **value**를 저장해 두고 step마다
  재사용하는 것입니다. 매 step마다 다시 계산하지 않기 위해서죠.

Prefill과 decode
: 요청을 처리하는 두 단계입니다. *Prefill*은 prompt 전체를 한 번의 forward pass로 처리하며
  KV cache를 채웁니다. 그다음 *decode*가 step당 token 하나씩 출력을 생성합니다. 1강이 이
  둘의 차이에 대한 것입니다.

Request (요청)
: Serving system에 보낸 prompt 하나와, 그에 대해 돌려받는 응답입니다.

Batch
: 같은 forward pass에서 함께 처리되는 요청들의 묶음입니다. Weight를 한 번만 읽어 그 전부에
  씁니다.

### 하드웨어

GPU
: 모델을 돌리는 가속기입니다. 예를 들어 NVIDIA A100이나 H100.

FLOP과 FLOP/s (FLOPs)
: FLOP은 부동소수점 연산 한 번(덧셈이나 곱셈)입니다. FLOP/s(간단히 FLOPs라고도 씁니다)는
  초당 몇 번의 FLOP을 할 수 있는지를 말합니다. A100의 최대치는 312 TFLOP/s
  (312 × 10¹²)입니다.

HBM과 메모리 대역폭
: HBM(high-bandwidth memory)은 GPU의 주 메모리로(적어도 A100이나 H100처럼 비싼
  GPU들은 HBM을 씁니다), weight와 KV cache를 담습니다. 초당 바이트로 재는 *대역폭*은 HBM과 연산
  유닛 사이에서 데이터가 얼마나 빨리 움직일 수 있는지를 말합니다.

Compute-bound와 memory-bound
: 데이터를 읽는 것보다 계산이 더 오래 걸리면 *compute-bound*, 데이터를 읽는 것이 더 오래
  걸리면 *memory-bound*입니다.

Tensor parallelism (TP)
: 모든 weight 행렬을 여러 GPU에 쪼개서 매 forward pass의 일(과 메모리 사용량)을 나누는
  방법을 뜻합니다. "TP2"는 GPU 두 장이 일을 나눠 한다는 뜻입니다.

### Serving과 지표

Serving system
: 요청을 받아 GPU에 scheduling하고 출력을 흘려보내는 소프트웨어입니다. vLLM, SGLang,
  TensorRT-LLM 같은 것들이죠.

Latency와 throughput
: *Latency*는 요청 하나가 걸리는 시간이고, *throughput*은 시스템이 초당 끝내는 일의
  양입니다(요청/초 또는 token/초).

TTFT, TPOT, E2E
: *Time to first token*, *time per output token*, 그리고 *end-to-end* latency(도착부터
  마지막 token까지). 1강에서 정확히 정의합니다.

QPS
: 시스템에 도착하는 초당 질의(요청) 수, 즉 부하입니다.

p50과 p99
: Percentile입니다. p50은 중앙값이고, p99는 요청의 99%가 그 이하인 값으로, 가장
  느린("tail") 요청들을 재는 지표입니다.

SLO
: Service-level objective. 시스템이 지키겠다고 약속하는 latency 목표입니다. "p99 TTFT
  500 ms 이내" 같은 것이죠.

## Python과 notebook

Jupyter notebook에서 Python을 읽고 조금씩 고치게 됩니다. Python이나 Jupyter notebook이
익숙하지 않다면 다음 자료가 도움이 됩니다.

- [The Python Tutorial](https://docs.python.org/3/tutorial/): 3–5장이면 충분합니다.
- [Welcome to Colab](https://colab.research.google.com/notebooks/intro.ipynb): 셀 실행하기,
  사본 저장하기.

## Transformer와 LLM

Transformer가 어떻게 동작하는지 대략은 알고 있어야 합니다. 익숙하지 않다면, 제가 좋아하는
자료들을 소개합니다(AI가 고른 게 아니라 제가 개인적으로 좋아하는 것들입니다!).

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar):
  제가 가장 좋아하는 자료 중 하나지만, *오리지널* transformer 구조를 설명한다는 점에 유의하세요.
  요즘 transformer(*decoder-only* transformer)는 다르게 생겼고, 그건 아래 자료에 나옵니다.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar):
  decoder-only transformer와 token 생성.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Polo Club,
  Georgia Tech): 브라우저에서 GPT-2가 실제로 도는 것을 보여 주는 웹사이트. Prompt를
  입력하고 그것이 모든 layer를 통과하는 과정을 지켜보세요.

## KV cache

생성 중에 모델은 각 token의 attention **key와 value**를 저장해 두어서, 새 token마다 그것을
다시 계산하지 않습니다. 이 KV cache는 이 시리즈 전체의 중심입니다. Decode가 step당 token
하나를 처리할 수 있게 해 주고, GPU 메모리를 두고 weight와 경쟁합니다.

- [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
  (Sebastian Raschka): cache가 무엇을 왜 저장하는지 그림과 함께 설명하고, 밑바닥부터 구현해
  봅니다.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
  (kipply): KV cache 크기와 생성 step의 비용.

## GPU의 속도를 제한하는 것

계산은 GPU가 **연산을 얼마나 빨리 할 수 있는지**(FLOP/s), 아니면 메모리에서 **데이터를
얼마나 빨리 옮길 수 있는지**(bytes/s) 중 하나에 묶입니다. 어느 쪽인지는 바이트 하나를 읽을
때 연산을 몇 번 하는지, 즉 *arithmetic intensity*에 달려 있습니다. 이것이 roofline
model이고, 1강은 그 위에 세워져 있습니다.

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
  (Horace He): compute-, memory-, overhead-bound를 수식 없이 설명합니다.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)
  (*How To Scale Your Model*): 예제와 함께 보는 roofline model.

실습에서 쓰는 GPU인 [NVIDIA A100-80GB](https://www.nvidia.com/en-us/data-center/a100/)에
대해 익혀 두면 좋은 숫자들:

| | A100-80GB (SXM) |
|---|---|
| 16-bit 연산 최대치 (dense) | 312 TFLOP/s |
| 메모리(HBM) 대역폭 | 약 2.0 TB/s |
| 메모리 용량 | 80 GB |

## 준비되셨나요?

아래 문제에 답할 수 있다면 1강을 들을 준비가 된 것입니다. 답을 눌러 확인하세요. 틀리면
다시 시도할 수 있습니다.

```{quiz}
- q: 파라미터 300억 개짜리 모델은 16-bit precision으로 메모리를 얼마나 차지하나요?
  options: [30 GB, 60 GB, 120 GB, 480 GB]
  answer: 1
  explain: 16 bit는 파라미터당 2바이트이므로 30 × 10<sup>9</sup> × 2 B = 60 GB입니다.
- q: 100 token짜리 답변을 만들려면 LLM은 모델을 대략 몇 번 돌리나요(forward pass)?
  options: ["1번", "약 100번", "100 × prompt 길이"]
  answer: 1
  explain: 생성은 autoregressive해서 forward pass 하나당 token 하나입니다. 첫 pass(prefill)가 prompt 전체를 처리하며 첫 token을 내고, 이후 token마다 pass가 하나씩 더 듭니다.
- q: KV cache는 무엇을 저장하나요?
  options: [모델의 weight, 지금까지 처리한 token들의 attention key와 value, 생성된 텍스트, 학습에 쓰는 gradient]
  answer: 1
  explain: 각 token의 key와 value를 들고 있으면, 매 생성 step마다 다시 계산하지 않고 한 번만 계산하면 됩니다.
- q: 대화가 길어질수록 KV cache가 커지는 이유는?
  options: [지금까지의 모든 token에 대해 항목을 저장하므로, 모델의 weight가 시간이 지나며 커지므로, 이전 답변마다 weight 사본을 보관하므로]
  answer: 0
  explain: 새 token은 prompt든 output이든 모든 layer에 자기 key와 value를 더합니다.
- q: 어떤 계산이 메모리에서 읽는 바이트 하나당 FLOP 10번을 합니다. A100(312 TFLOP/s, 2 TB/s)에서 무엇이 이 계산을 제한하나요?
  options: [연산 (FLOP/s), 메모리 대역폭]
  answer: 1
  explain: A100은 바이트 하나를 읽는 동안 312 × 10<sup>12</sup> ÷ 2 × 10<sup>12</sup> ≈ 156번의 FLOP을 할 수 있습니다. 바이트당 FLOP이 10번뿐이면 연산 유닛은 메모리를 기다립니다.
```

## 환경 준비

실습은 Colab이나 Binder를 통해 브라우저에서, 또는 여러분의 컴퓨터에서 돌아갑니다.
{doc}`../getting-started/run-in-browser`와 {doc}`../getting-started/run-locally`를
보세요. 1강은 모든 것이 잘 동작하는지 확인하는 설치 셀로 시작합니다.
