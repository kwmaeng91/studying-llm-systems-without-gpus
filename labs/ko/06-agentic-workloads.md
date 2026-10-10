# 6. Agentic Workload (1): Workload는 어떻게 생겼나

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
[MiroThinker](https://github.com/MiroMindAI/MiroThinker)가 들어 있고, 여기서는 둘 다
봅니다. 각 요청에는 prompt와 output의 token id, 그 요청이 의존한 turn들, 그리고 따로 측정한
tool latency가 들어 있습니다. Trace는 simulator와 함께 배포되므로, `lsg.setup()`을 하고 나면
이미 여러분 디스크에 있습니다. 참고로 GAIATrace를 더 많은 설정과 데이터셋으로 확장하는
작업도 진행 중입니다!

:::{admonition} 학습 목표
- Agent session의 구조를 설명할 수 있다: turn, 역할, 의존 관계, fan-out, tool 시간.
- Agent 트래픽이 token 수로는 prompt 중심인데 시간으로는 decode 중심인 이유를 설명할 수
  있다.
- Agent의 prompt token이 어디서 오는지, 그리고 prompt가 왜 자라는지 설명할 수 있다.
- 단일 agent 시스템과 multi-agent 시스템이 만들어 내는 트래픽이 어떻게 다른지 말할 수 있다.
:::

Agent 트래픽을 다루는 두 강의 중 첫 번째입니다. 여기서는 workload가 *무엇인지*를 보고,
{doc}`07-agentic-workloads-2`에서 그것을 GPU에 올립니다.
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

기록된 두 시스템은 서로 닮지 않았으니, 더 단순한 쪽부터 봅시다.
MiroThinker는 ReAct 스타일 루프를 도는 단일 agent입니다. 생각하고, tool을 부르고, 결과를
읽고, 다시 생각하죠. 여기에 도우미가 하나 붙습니다. Tool이 아주 긴 것(예: 스크랩한 웹
페이지)을 돌려주면, 그것이 agent의 context에 들어가기 전에 별도의 **요약기(summarizer)**
모델이 먼저 압축합니다.

기록된 MiroThinker 과제 24개를 불러와서 가장 긴 것을 그립니다. (이 session들에는 이 강의의
runtime predictor가 다루는 범위보다 훨씬 긴 요청이 들어 있어서 `max_tokens=None`을 줍니다.
지금은 simulation이 아니라 trace를 들여다보는 것뿐이니까요.)
<!-- cell -->
Agent 자신의 요청(`main`)은 계단처럼 올라갑니다. 매 turn은 이전 turn에 tool 결과를 더한
것이라서 prompt는 자라기만 합니다. 처음에 약 3,100 token이던 것이 서른네 turn 뒤에는
22,000 token이 됩니다. Agent serving을 다루는 논문들이 보통 보고하는 모양이 이것이고,
따지기도 쉽습니다. 대화 하나가 단조롭게 자라고, 한 번에 하나의 요청만 떠 있습니다.

`summarizer` 요청은 이 trace가 잡아냈지만 공개된 다른 trace들에는 대개 없는 부분이고,
대화를 전혀 따라가지 않습니다. 각 호출은 긴 스크랩 페이지를 더 싼 모델에 넘기고 수백 token을
돌려받으며, main agent가 보는 것은 그 결과입니다. Main agent 자신의 prompt가 이 정도로
완만하게만 자라는 이유도 여기 있습니다. 긴 문서를 요약기가 대신 흡수해 주는 것이죠.
이 과제에서 요청의 3분의 1이 요약기 호출이므로, "prefill이 단조롭게 자란다"는 말은 serving
system이 실제로 받는 것의 일부만 설명합니다.

이제 multi-agent 시스템을 봅시다. 여기에는 자라날 단일 대화라는 것 자체가 없습니다.
OWL은 과제를 planner, coordinator, 그리고 여러 worker에게 나누고, 각자가 자기 context를
가집니다.
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

:::{admonition} 직접 해보기
:class: exercise
위 코드의 session id(`sessions.session == 4`)를 다른 숫자로 바꾸고 다시 돌려 보세요.
과제마다 sub-agent들이 꽤 다른 패턴으로 협력해서 문제를 풉니다. 어떤 숫자가 가장 흥미로운
그림을 만드나요?
:::

## Token, 시간, 그리고 tool

이제 표본 전체를, 두 시스템을 나란히 놓고 봅시다. 이 트래픽이 serving system에 무엇을 하는지는
세 가지 숫자가 결정합니다. Token이 얼마나 들어가고 나오는지, 각 종류의 token이 얼마나
걸리는지, 그리고 tool이 도는 동안 아무것도 실행되지 않는 시간이 얼마나 되는지.
<!-- cell -->
두 시스템 모두 **prefill 중심**이고, 단일 agent 쪽이 훨씬 더 그렇습니다. Output token 하나당
prompt token이 OWL은 11개, MiroThinker는 30개입니다. 이유는 산점도에 보입니다. OWL의
sub-agent들은 뚜렷한 무리를 이룹니다. Coordinator는 worker id로 답하고, 웹 검색 worker는
수십 token짜리 tool 호출을 내놓고, 요약기는 페이지 조각을 읽습니다. 반면 MiroThinker에는
둘뿐입니다. 대화를 따라 prompt가 자라는 main agent와, 수만 token짜리 prompt에 짧은 답을
내놓으며 오른쪽 멀리 자리 잡은 요약기.

Tool의 양상도 다릅니다. MiroThinker는 turn의 3분의 2가 tool을 기다리는 반면 OWL은 3분의
1이고, 기다리는 시간도 중앙값 기준 네 배(1.5초 대 0.4초) 깁니다. 그동안 두 시스템 모두 GPU
근처에도 가지 않습니다.

다만 prompt token과 output token의 비용은 다릅니다. 우리가 simulation하는 replica에서
prompt token 하나는 약 0.25 ms의 GPU 시간이 드는 반면, output token 하나는 약 45 ms, 거의
200배입니다. Decode가 memory-bound이기 때문이죠(1강). Prompt 9,000 token에 output 180
token인 turn은 prompt에 2초 남짓, 답변에 8초를 씁니다. 그러니 prefill 중심이라고 해서 늘
prefill이 병목이 되는 것은 아닙니다(그럴 수도, 아닐 수도 있습니다).
또 하나 고려할 것은 prefix cache(5강)입니다. Prefix cache hit rate가 높으면 prefill은 훨씬
싸지는데, 다음 강의가 바로 거기서 시작합니다.

:::{admonition} 직접 해보기
:class: exercise
기록 당시에는 모델을 두 개 썼으므로, 이 시스템의 실제 배포는 하나가 아니라 두 개의 serving
문제입니다. `model` 열을 이용해 각 모델이 이 workload에서 얼마나 많은 GPU 시간을 필요로
할지 계산해 보세요. 모델별로 prompt와 output token을 세고, 위의 token당 약 0.25 ms와 약
45 ms로 값을 매기면 됩니다. GPU가 여덟 장이라면 OWL에는 어떻게 나누고, MiroThinker에는
어떻게 나누겠습니까? 각 경우 어느 pool이 먼저 포화될까요?
:::

## 정리

| | 무엇을 보았나 |
|---|---|
| 과제는 요청이 아니다 | GAIA 과제 하나가 서로 의존하는 수십 개의 LLM 요청이고, 사용자는 그 사슬 전체를 기다린다 |
| 단일 agent | 자라기만 하는 대화 하나, 즉 깔끔한 prompt 길이의 계단. 여기에 대화를 전혀 따라가지 않는 요약기 요청이 섞인다 |
| Multi-agent | 단일 대화가 없다. Worker마다 자기 대화를 시작하고, fan-out은 한순간에 열두 개의 요청을 풀어놓는다 |
| Token | 둘 다 prefill 중심이고, OWL은 11:1, MiroThinker는 30:1 |
| 시간 | output token 하나는 여전히 prompt token의 약 200배이므로, prefill 중심이라고 prefill이 병목인 것은 아니다 |
| Tool | OWL은 turn의 3분의 1, MiroThinker는 3분의 2가 tool을 기다리고, 중앙값은 각각 0.4초와 1.5초 |

{doc}`07-agentic-workloads-2`에서는 이 trace들을 GPU에 올립니다. 그 많은 prompt 텍스트 중
얼마나가 재사용 가능한지, 여기서 prefix caching이 얼마나 값어치를 하는지, 그리고 과제의
시간이 실제로 어디로 가는지.

## 이해도 확인

답을 눌러 확인하세요. 틀리면 다시 시도할 수 있습니다.
