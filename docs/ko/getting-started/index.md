# 이 강의는 어떻게 진행되나요

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

## Simulator

각 실습은 LLM inference simulator를 돌립니다. 모델, GPU 종류, cluster 구성,
scheduler, 그리고 workload를 주면, simulator가 workload를 요청 하나씩 재현하면서
실제 serving system의 로그처럼 latency와 throughput 수치를 알려줍니다.

이 simulator는 모델을 실제로 **돌리지 않습니다**. 대신 각 연산(matrix multiply,
attention, all-reduce 등)이 목표 GPU에서 얼마나 걸릴지를, 실제 하드웨어에서 측정한
데이터로 학습한 모델을 이용해 예측합니다.
{doc}`../reference/how-the-simulator-works`를 참고하세요.

## `llm_systems_wo_gpus` 패키지

Simulator에는 수백 개의 command-line flag가 있습니다. 실습에서는 그 위에 얹은 작은
Python 인터페이스인
[`labs/llm_systems_wo_gpus.py`](https://github.com/kwmaeng91/studying-llm-systems-without-gpus/blob/main/labs/llm_systems_wo_gpus.py)를
사용합니다. 교육용으로 저희가(정확히는 제 Claude Code agent가) 만든 것입니다:

```python
import llm_systems_wo_gpus as lsg
lsg.setup()                                   # 최초 1회 clone + 설치

r = lsg.simulate(model="meta-llama/Llama-2-7b-hf", device="a100",
                qps=4, prefill_tokens=512, decode_tokens=128)
r.summary()        # TTFT / TPOT / E2E percentile, throughput
r.requests         # 요청 하나당 한 행 (pandas DataFrame)

lsg.sweep("qps", [1, 2, 4, 8])                # 값마다 summary 한 행
```

전체 API는 {doc}`../reference/python-api`에 있습니다. 혹시
[Vidur-Agent](https://github.com/psu-paws/Vidur-Agent)나
[Vidur](https://github.com/microsoft/vidur)를 연구에 쓰실 생각이라면, 이 교육용
패키지는 쓰지 **않기를** 강력히 권합니다. 대신 실제 simulator를 직접 다루는 법을
익히세요. 이 교육용 패키지는 단순함을 위해 할 수 있는 것을 많이 제한해 두었습니다.

## 어디서 실행하나요

| 방법 | 준비물 | 첫 실행 | 추천 대상 |
|---|---|---|---|
| {doc}`Google Colab <run-in-browser>` | Google 계정 | 약 1–2분 | 대부분의 학생 |
| {doc}`Binder <run-in-browser>` | 없음 | 서버 시작까지 2–10분 | 계정조차 만들기 싫을 때 |
| {doc}`로컬 설치 <run-locally>` | Python 3.10+, git | 약 2분 | 긴 프로젝트, 작업 저장 |

모든 실습은 **1 GB 미만의 RAM**과 CPU 코어 2개면 충분하므로, 세 가지 방법 모두
문제없이 동작합니다.
