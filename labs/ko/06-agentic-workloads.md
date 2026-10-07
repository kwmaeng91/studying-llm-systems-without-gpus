# 6. Agentic Workload

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

지금까지의 모든 workload는 *요청* 단위였습니다. Prompt가 도착하고, 답변이 흘러나오고,
요청은 사라집니다. Agent는 다르게 동작합니다. "이 식당 메뉴에서 두 날짜 사이에 사라진
메인 요리를 찾아라" 같은 과제 하나를 받으면, 수십 개의 LLM 요청을 냅니다. 계획을 세우고,
worker를 고르고, 검색 tool을 부르고, 결과를 읽고, 다른 tool을 부르고, 코드를 쓰고,
실행하고, 에러를 읽고, 다시 시도하고, 마침내 답을 내놓습니다. 사용자는 그 사슬 전체를
기다립니다.

앞에서 본 것과 다른 점이 세 가지 있습니다. 첫째, 요청들이 서로 *의존적*입니다. Turn
*k+1*은 turn *k*가 끝나고 그 tool 호출이 돌아올 때까지 시작하지 못할 수 있습니다. 둘째,
prompt가 길고 반복적입니다. 매 turn이 대화 전체와 새 tool 출력을 다시 보내기 때문인데,
5강의 문제가 가장 극단적인 형태로 나타난 것입니다. 셋째, 사용자가 기다리는 대상은 요청이
아니라 과제입니다. 과제가 6분 걸린다면 p99 TTFT가 200 ms라는 말은 별 의미가 없습니다.

이 강의는 합성 workload 대신 기록된 trace로 작업합니다. Trace는
[GAIATrace](https://github.com/psu-paws/Vidur-Agent)
([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725))에서 왔고, 두 agent 시스템이
GAIA ([Mialon et al., ICLR '24](https://openreview.net/forum?id=fibxvahvs3)) 과제를 푸는
동안 낸 모든 LLM 요청을 기록한 것입니다. GAIA는 웹 브라우징, 파일 처리, 코드 실행이 있어야
풀리는 질문들의 benchmark입니다. 이 corpus에는 multi-agent 시스템인
[OWL](https://github.com/camel-ai/owl)
([Hu et al., NeurIPS '25](https://arxiv.org/abs/2505.23885))과, 단일 agent에 요약기를 붙인
[MiroThinker](https://github.com/MiroMindAI/MiroThinker)가 들어 있고, 여기서는 OWL을
씁니다. 각 요청에는 prompt와 output의 token id, 그 요청이 의존한 turn들, 그리고 따로 측정한
tool latency가 들어 있습니다. Trace는 simulator와 함께 배포되므로, `lsg.setup()`을 하고 나면
이미 여러분 디스크에 있습니다. 참고로 GAIATrace를 더 많은 설정과 데이터셋으로 확장하는
작업도 진행 중입니다!

:::{admonition} 학습 목표
- Agent session의 구조를 설명할 수 있다: turn, 역할, 의존 관계, fan-out, tool 시간.
- Agent 트래픽이 token 수로는 prompt 중심인데 시간으로는 decode 중심인 이유를 설명할 수
  있다.
- 실제 agent trace에서 KV cache hit rate를 측정하고, hit가 어디서 오는지 설명할 수 있다.
- Task 단위 latency를 따져 볼 수 있다. 임계 경로가 무엇이고, serving system의 어떤 knob이
  그것을 줄일 수 있고 없는지.
:::
<!-- cell -->
## 준비
<!-- cell -->
`lsg.gaia_sessions`는 기록된 session을 LLM 요청당 한 행씩 불러옵니다. OWL이 돌린 165개
과제 중 24개를 씁니다. 이 강의의 runtime predictor가 학습된 범위보다 긴 요청이 들어 있는
session은 건너뛰므로, 가장 큰 과제들이 빠집니다(다시 말하지만, 이 강의는 단순함을 위해
축소된 simulator를 씁니다).
<!-- cell -->
이 표는 OWL이 특정 질문을 어떻게 푸는지 살짝 엿보게 해 줍니다.
각 행은 모델에 보낸 요청 하나이고, 계획, 조율, 웹 검색 등을 수행합니다.
여기서는 모델 두 개가 쓰인다는 점에 주의하세요. 깊은 사고가 필요 없는 일에는 gpt-4o를,
사고가 필요한 일에는 gpt-oss-120b를 씁니다. 이건 그냥 GAIATrace 저자들(바로 저와 제
학생들입니다!)이 그렇게 정한 것입니다.

- `session` / `turn`: 어떤 과제이고, 그 안에서 몇 번째인지.
- `role`: 시스템 안의 어떤 agent가 낸 요청인지. OWL은 *multi-agent* 시스템입니다. planner가
  과제를 쪼개고, coordinator가 각 하위 과제를 worker에게 맡기고, worker들(웹 검색, 코드,
  브라우저, 문서 읽기)이 그것을 수행합니다.
- `model`: 기록 당시 그 요청을 처리한 모델. OWL은 두 개를 쓰는데, GAIATrace에서는 이들을
  *main*과 *sub* 모델이라고 부르고 각각에 서로 다른 역할을 줍니다.
- `dep`: 이 요청이 풀리기 전에 끝나야 하는 turn들.
- `tool_time`: 이 turn이 기다린 tool 호출이 실제로 얼마나 걸렸는지. Agent 밖에서 tool을 다시
  돌려 측정했습니다.

:::{note}
Trace는 GAIATrace의 일부로 CC-BY-4.0으로 공개되어 있습니다. 과제 자체는 GAIA benchmark에서
왔고, 그 질문들은 [데이터셋 약관](https://huggingface.co/datasets/gaia-benchmark/GAIA)을
따릅니다. 아래에 나오는 것은 기록된 실행 하나에서 가져온 짧은 발췌로, 실제 agent prompt가
무엇으로 이루어져 있는지 보여 주려는 것입니다.
:::

## 모델이 실제로 보는 것

Trace에는 token id가 들어 있으므로 요청을 다시 읽어 볼 수 있습니다. 한 session의 앞쪽
turn들을 봅시다. (`lsg.decode`는 trace를 만들 때 쓴 것과 같은 tokenizer를 쓰고, 처음 한 번
작은 사전 파일을 내려받습니다.)
<!-- cell -->
이것이 planner입니다. 사용자의 과제("I went to Virtue restaurant & bar ...")와 쓸 수 있는
worker들의 설명을 받아서 하위 과제 목록을 내놓습니다. 모든 것이 모델의 chat
template(`<|start|>`, `<|message|>`, `<|end|>`)으로 표시되어 있고,
`<|channel|>final`과 `<|channel|>analysis` 표시가 답변과 모델 자신의 추론을 구분합니다.

두 turn 뒤, 웹 검색 worker가 tool을 부르기로 합니다. 그 출력 전체가 36 token짜리 JSON,
즉 호출 그 자체입니다:
<!-- cell -->
Agent가 그 호출을 실행하고(1.2초 걸립니다), *결과를 다음 prompt의 일부로 다시 보냅니다*.
아래는 그다음 요청의 prompt 끝부분인데, 정확히 이전 prompt + 이전 출력 + tool 결과입니다:
<!-- cell -->
이것이 이 강의 전체의 바탕이 되는 메커니즘입니다. 요청 사이에는 아무것도 기억되지 않으므로
매 turn이 대화 전체를 다시 보내고, prompt는 그 worker가 끝날 때까지 한 가지(branch)를 따라
계속 자랍니다. 거의 모든 prompt는 앞선 어떤 prompt에 텍스트가 덧붙은 것인데, 이는 5강이
prefix cache에 필요하다고 말한 조건 그 자체입니다.

## Agent session의 모양

Session 하나를 통째로 봅시다. Prompt가 어떻게 자라는지, 각 turn을 어떤 역할이 내는지,
그리고 tool 시간이 어디로 가는지.
<!-- cell -->
각 점은 sub-agent(plan, coordinate, web search, ...)가 낸 요청 하나이고, agent가 tool을 쓰면
막대가 그 tool의 실행 시간을 보여 줍니다.
Prompt 길이(각 점의 y축)는 몇 개의 turn에 걸쳐 자라다가 뚝 떨어집니다. Coordinator가 하위
과제를 새 worker에게 넘길 때마다 그 worker는 자기 system prompt로 새 대화를 시작하기
때문입니다. 한 worker가 이어 가는 구간 안에서는 매 turn이 tool 결과를 prompt에 덧붙입니다.
다시 말하지만 이는 OWL이 그렇게 설계된 것이지 보편적인 것은 아닙니다.

가운데 평평하게 늘어선 큰 turn 열두 개는, 웹 worker가 긴 페이지를 가져와 열두 조각으로
쪼갠 뒤 각 조각을 병렬로 요약해 달라고 모델에 부탁하는 장면입니다. 이것 역시 prompt(prefill)
길이를 작게 유지하려고 OWL 저자들이 택한 특정 설계이고 본질적인 것은 아닙니다.
이 그래프에서 벌어지는 모든 것을 이해할 필요는 없고, 많은 부분이 이 agent 특유의 설계에서
오는 것이니, multi-agent 시스템이 복잡하다는 감각 정도면 충분합니다.

## Token, 시간, 그리고 tool

이제 sub-agent별로 prompt(입력)와 output token을 봅시다:
<!-- cell -->
역할에 따라 sub-agent마다 다른 양상을 보이는 것을 볼 수 있습니다. coordinator는 prompt
길이는 제각각이지만 output 길이가 비슷한 뚜렷한 무리를 이루고, 웹 검색 agent는 prompt
길이는 다양한데 output 길이는 비슷하며, 웹 요약 agent는 prompt가 길고 output은 상대적으로
짧습니다. 각 sub-agent가 무슨 일을 하는지 생각해 보면 납득이 갑니다.
전체적으로는 prompt(입력)가 output보다 token 수로 약 11:1 정도 길어서, prefill 중심의
트래픽입니다.

다만 prompt token과 output token의 비용은 다릅니다. 우리가 simulation하는 replica에서
prompt token 하나는 약 0.25 ms의 GPU 시간이 드는 반면, output token 하나는 약 45 ms, 거의
200배입니다. Decode가 memory-bound이기 때문이죠(1강). Prompt 9,000 token에 output 180
token인 turn은 prompt에 2초 남짓, 답변에 8초를 씁니다. 그러니 prefill 중심이라고 해서 늘
prefill이 병목이 되는 것은 아닙니다(그럴 수도, 아닐 수도 있습니다).
또 하나 고려할 것은 prefix cache(5강)입니다. Prefix cache hit rate가 높으면 prefill은 훨씬
싸집니다.

## Agentic workload에서 prefix caching 다시 보기

이제 이 trace가 prefix cache hit를 얼마나 얻는지 봅시다.
이런 agentic trace는 prefix를 많이 재사용할까요?
먼저, 완벽하고 무한히 큰 prefix cache라면 어떻게 될지를 5강의 규칙 그대로 계산해 봅니다.
꽉 찬 16 token block마다 사슬 해시를 구하고, 각 prompt의 앞쪽 block 중 몇 개가 전에 본
것인지 셉니다.
<!-- cell -->
이 workload에서 전체 prompt token의 약 60%는 전에 계산된 적이 있어 prefix caching의 혜택을
받을 수 있고, 이 숫자는 GAIATrace 논문
([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725))과 대체로 들어맞습니다.
이 값이 높은 이유는 system prompt가 길고, sub-agent들이 자기 과거 대화 기록을 자주 들여다보기
때문입니다. 5강의 multi-turn chat과 비슷하죠.
다만 이는 다른 논문들(예: Agentic AI Workload Characteristics,
[Yuan et al., IISWC '26](https://arxiv.org/abs/2605.26297))이 보고한 값보다는 훨씬 낮습니다.
그쪽은 87--99% 정도였습니다.
이는 OWL의 설계 때문입니다. multi-agent 시스템인 OWL은 여러 sub-agent를 돌리는데, 서로 다른
agent 사이에는 prompt 공유가 제한적입니다.

그래도 이 값은 Mooncake
([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079))가 보고한 값보다는 (조금) 높습니다.
Mooncake는 Kimi chatbot에 오는 tool/agent 성격 트래픽에서 약 59%, 일반 대화에서 약 40%였습니다.
다시 말하지만, agentic 시스템을 어떻게 설계했느냐가 prefix cache hit rate를 크게 좌우합니다.
agentic 시스템의 올바른 설계가 무엇인지 아직 합의가 없으므로, 이 숫자는 합의가 모일 때까지
앞으로도 계속 흔들릴 것입니다.

## Trace를 서비스해 보기

이제 trace를 simulator에 넣어 봅시다.
여기서는 간단히, 어떤 sub-agent가 냈든 기록 당시 어떤 모델이 처리했든, 모든 요청이 A100 두
장 위의 Qwen2.5-32B replica 하나로 간다고 가정합니다. 3–5강과 같은 설정이고, prefix cache는
그 replica의 GPU 메모리에 있습니다.
Tool latency는 trace에서 가져오므로, turn은 실제 tool이 걸린 만큼 정확히 기다립니다.
`lsg.gaia_trace`가 token id와 의존 그래프를 포함해 simulator 형식으로 trace를 써 주고,
그래서 turn *k+1*은 turn *k*가 끝나고 그 tool 호출이 돌아온 뒤에야 풀립니다. 새 과제는 10초에
하나씩 replica 하나로 보냅니다.

사용자는 과제 전체를 기다리므로 **task 완료 시간**을 잽니다. Session의 첫 요청이 도착한
순간부터 마지막 요청이 끝나는 순간까지, tool 시간까지 포함해서요.
<!-- cell -->
측정된 KV cache hit rate 60%는 손으로 계산한 이상치와 1 퍼센트포인트 안에서 맞아떨어집니다.
실행 중에 block이 evict되기는 합니다(`r.cache`가 수만 개를 보고합니다). 다만 이 replica는
충분히 커서, 잃는 것은 대부분 아무도 다시 찾지 않는 block입니다. Prefill 작업량은 60%
줄고, TTFT는 6배 좋아지며, 중앙값 과제는 약 1분 30초 빨리 끝납니다.

과제 단위의 이득은 요청 단위의 이득보다 훨씬 작습니다. TTFT가 6배 좋아져도 과제는 4분의 1만
줄어듭니다. 앞 문장의 뒷부분이 여기서 작동합니다. 과제는 시간의 대부분을 token 생성과 tool
대기에 쓰는데, prefix caching은 둘 중 어느 것도 건드리지 못합니다.

## 과제의 시간은 어디로 가나

Session을 뜯어 봅시다. 각 turn의 막대는 도착부터 첫 token까지(기다림과 prefill), 그다음
마지막 token까지(decode)를 나타냅니다. Turn 사이의 빈 구간은 tool 호출과 agent 자신의
처리입니다.
<!-- cell -->
첫째, 과제가 질의들의 사슬로 실행되는 것을 볼 수 있습니다. 대체로 순차적이지만 가끔
병렬이죠. 많은 agentic 시스템이 대체로 순차적이고, 더 병렬적인 구조를 넣으려는 시도들이
연구되고 있습니다(예: LATS,
[Zhou et al., ICML '24](https://arxiv.org/abs/2310.04406)).
둘째, 파란색(decode) 부분이 지배적인 것을 볼 수 있습니다. Prefix cache hit rate가 높고,
지금은 GPU에 다른 경쟁이 없기 때문입니다.
Prefix cache hit rate가 낮고 같은 GPU에 다른 요청이 많다면 막대의 모양은 달라질 것입니다
(2강에서 배웠듯, prefill은 decode처럼 깔끔하게 batching되지 않으니까요).
다시 말하지만, 가운데의 병렬 과제 열두 개는 OWL이 아주 긴 웹 스크랩 텍스트를, context 길이를
크게 늘리지 않으면서 요약하려는 장면입니다.

## 정리

| | 무엇을 보았나 |
|---|---|
| 과제의 모양 | agentic 과제는 대체로 순차적인 요청들의 연속이지만 가끔 병렬이다 |
| Token | 보통 input token이 output token보다 많지만, agent의 역할에 따라 다르다 |
| 시간 | 여전히 decode가 지배한다 (특히 batch size가 작고 prefix cache hit rate가 높을 때) |
| Prefix caching | hit rate는 약 60%이고 TTFT를 6배 개선하지만, 중앙값 과제 시간은 4분의 1 정도만 줄인다 |

## 연습문제

:::{admonition} 직접 해보기
:class: exercise

1. **Cache가 tool 호출을 견디나?** Caching 비교를 작은 KV cache(`kv_blocks=4000`, 5강)로
   다시 돌리고 `r.cache`를 보세요. Block이 몇 개나 evict되고 hit rate는 어떻게 되나요? 그리고
   tool을 가장 오래 기다린 session들이 다른 것보다 더 많이 잃나요?
2. **Agent를 위한 chunk size.** 위 실행들은 기본값 `chunk_size=512`를 썼습니다. replica
   하나짜리 실행에서 512, 2048, 4096을 쓸어 보고 TTFT와 과제 시간을 보고하세요. 채팅에서는
   나빴던 chunk size(3강)가 여기서는 왜 더 나아 보이고, prefix caching을 켜면 답이 달라지나요?
3. **Fan-out의 비용.** Session 하나짜리 타임라인에서, 열두 개를 기다리는 join turn(`dep`에
   항목이 열두 개인 turn)이 *첫* 의존 turn이 끝난 뒤 얼마나 더 기다리는지 재 보세요. 그중
   얼마가 자기 형제들 뒤에 줄 서서 기다린 시간이고, 그것을 줄이려면 scheduler가 무엇을
   알아야 할까요?
4. **두 pool의 크기 정하기.** `model` 열을 이용해, 이 workload에서 두 모델이 각각 얼마나
   많은 GPU 시간을 필요로 할지 계산해 보세요. 모델별로 prompt와 output token을 세고, 위에서
   측정한 token당 약 0.25 ms와 약 45 ms로 값을 매기면 됩니다. GPU가 여덟 장이라면 어떻게
   나누겠습니까? 그리고 어느 pool이 먼저 포화될까요? 그다음 `lsg.gaia_sessions`가 보여 주는,
   어떤 역할이 임계 경로에 있는지와 비교해 확인해 보세요.
5. **다른 표본.** `seed=1`, `num_sessions=48`로 특성 분석을 다시 돌려 보세요.
   prompt:output 비율과 이상적인 hit rate는 얼마나 안정적인가요? 이는 하나의 trace에 맞춰
   시스템을 튜닝하는 것에 대해 무엇을 말해 주나요?
:::

## 이해도 확인

답을 눌러 확인하세요. 틀리면 다시 시도할 수 있습니다.
