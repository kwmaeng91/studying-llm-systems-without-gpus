# 4. Prefill–Decode Disaggregation

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

3강에서 prefill과 decode가 같은 GPU를 쓰면 서로 간섭하고, chunked prefill이 prompt를
잘게 쪼개 그 간섭을 길들인다는 것을 보았습니다. 하지만 없애지는 못합니다. Prefill
chunk를 실은 step은 순수 decode step보다 느리고(3강에서 약 40 ms 대신 약 105 ms),
`chunk_size`는 TTFT와 TBT 중 *어느 쪽을* 희생할지 고르게 해 줄 뿐입니다.

이 강의에서는 더 과감한 아이디어를 봅니다. **두 단계가 아예 GPU를 공유하지 않게 하는
것**입니다. Prefill은 한 무리의 GPU에서, decode는 다른 무리에서 돌리고, prompt 처리가
끝난 요청을 앞쪽에서 뒤쪽으로 옮깁니다. 이것이 **prefill–decode (PD) disaggregation**
이고, DistServe ([Zhong et al., OSDI '24](https://arxiv.org/abs/2401.09670))와 Splitwise
([Patel et al., ISCA '24](https://arxiv.org/abs/2311.18677))가 제안했으며,
Mooncake ([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079))나 NVIDIA
[Dynamo](https://github.com/ai-dynamo/dynamo) 같은 실제 시스템에서 쓰입니다.

이 강의 내내 같은 수의 GPU에서 3강의 chunked prefill 설정과 비교하면서, 각각이 이기는
경우를 찾아봅니다.

:::{admonition} 학습 목표
- PD disaggregation이 prefill–decode interference를 어떻게 없애는지, 그리고 요청이
  disaggregated cluster를 어떻게 지나가는지 설명할 수 있다.
- Disaggregation이 TPOT/TBT에는 도움이 되지만 TTFT와 용량에는 해가 될 수 있는 이유와,
  prefill:decode GPU 비율이 왜 그렇게 중요한지 설명할 수 있다.
- 주어진 workload에서 chunked prefill과 PD disaggregation 중 무엇이 나은지 판단할 수 있다.
:::
<!-- cell -->
## 준비

{doc}`03-chunked-prefill`와 같은 설정입니다. Qwen2.5-32B-Instruct이고, 모델 복사본
하나(**replica**)가 A100 두 장에서 돕니다.
<!-- cell -->
## Disaggregation은 어떻게 동작하나

Disaggregated cluster에는 두 종류의 replica **pool**이 있습니다:

- **Prefill replica**는 prefill만 돌립니다. 새 요청은 먼저 여기로 옵니다. Prompt 처리가
  끝나면 prefill replica는 첫 token을 만들어 냈고(즉 TTFT가 여기서 결정됩니다) 요청의
  KV cache도 다 만들어 둔 상태입니다.
- **Decode replica**는 decode step만 돌립니다. 요청의 **KV cache가** GPU 사이의
  네트워크를 통해 prefill replica에서 decode replica로 **전송되고**, 요청은 끝날 때까지
  거기서 decode합니다.

Decode replica는 prefill 작업을 절대 돌리지 않으므로, 다른 사용자가 어떤 prompt를 보내든
decode 중인 사용자는 순수 decode step의 속도로 계속 token을 받습니다.

여기에는 대가가 있습니다. Chunked prefill에서는 decode 중인 요청을 들고 있는 GPU가
prefill도 처리하고, decode는 거의 공짜로 얹혀 갔습니다(3강). Disaggregation에서는
prefill이 prefill GPU만 쓸 수 있습니다. 그 GPU들이 바쁘면, decode GPU에 연산 여유가
있더라도 새 요청은 기다립니다. **interference를 하드웨어의 고정 분할과 맞바꾸는
셈입니다.**

## Disaggregated cluster 기술하기

`lsg.simulate`는 cluster를 기술하는 `replica_groups`를 받습니다. 각각 역할(`"prefill"`
또는 `"decode"`), replica 수, 설정을 가진 replica group의 목록과, 어떤 prefill replica가
어떤 decode replica에게 요청을 넘기는지 말해 주는 "pool"로 이루어집니다. 아래 helper는
TP=2 replica(A100 두 장)를 단위로 그것을 만들어 주어서, 비교하는 모든 구성이 온전한
replica 단위가 되도록 합니다.

한 가지 더: disaggregation에서는 cluster에 두 pool을 아는 router가 필요합니다
(`global_scheduler="load_aware"`). 이 router는 새 요청을 가장 덜 바쁜 prefill replica로
보내고, 그다음 자리가 있는 decode replica로 넘깁니다.
<!-- cell -->
## GPU 네 장에서 정면 승부

3강의 섞인 workload를 다시 씁니다. 85%는 채팅 turn(입력 256, 출력 256 token), 15%는
문서(입력 3,800, 출력 32 token)입니다. GPU 네 장으로는 이렇게 구성할 수 있습니다:

- **Chunked:** 평범한 replica 두 개, 각각 chunked prefill(`chunk_size=512`), 요청은
  둘 사이를 번갈아 갑니다;
- **PD 1:1:** prefill replica 하나와 decode replica 하나.
<!-- cell -->
두 설계는 같은 throughput을 내지만, latency는 반대 방향으로 움직입니다:

- **TPOT이 크게 좋아집니다.** Disaggregation에서 p99 TPOT은 chunked 설계의 약 절반이고,
  p50보다 간신히 높습니다. 모든 요청이 순수 decode step의 속도로 decode합니다.
  Interference가 사라졌습니다.
- **TTFT는 나빠집니다.** 중앙값 TTFT가 두 배 넘게 깁니다. Chunked 설계에서는 새 prompt가
  *어느* replica에서든 시작할 수 있지만, 여기서는 모든 prompt가 하나뿐인 prefill
  replica를 기다립니다.

첫 번째 지점은 각 replica가 돌린 step을 보면(`keep_steps=True`) 확인할 수 있습니다. PD의
decode replica에서 가장 긴 step도 순수 decode step인 반면, chunked replica들은 prefill
chunk를 실은 step을 자주 돌립니다:
<!-- cell -->
PD cluster에서 replica 0은 긴 prefill step을 돌리는 prefill replica이고, replica 1은
decode step만 돌리는 decode replica입니다. 바로 이 분리가 TPOT을 평평하게 유지합니다.

:::{admonition} 직접 해보기
:class: exercise
**더 짧은 문서.** `mixed_trace`에서 문서 길이를 3,800에서 1,000 token으로 바꾸고 이 정면
승부를 다시 돌려 보세요. Disaggregation이 여전히 TPOT을 그만큼 개선하나요? 왜 그럴까요?
:::

## 각 pool을 따로 튜닝하기

단계를 분리하면 두 번째 이점이 생깁니다. 각 pool을 자기 일에 맞게 설정할 수 있다는
것이죠. 3강에서는 token 예산 하나가 상충하는 두 목표를 동시에 만족시켜야 했습니다.
Prefill replica에는 보호해야 할 decode 사용자가 없으므로, 큰 예산을 써서 가능한 한 적은
step으로 prompt를 prefill할 수 있습니다. Prefill pool을 3강의 예산(512)과 큰
예산(4,096, 우리 helper의 기본값)으로 비교해 봅시다:
<!-- cell -->
큰 예산은 TPOT을 전혀 희생하지 않고 중앙값 TTFT를 거의 절반으로 줄입니다. 실제
시스템에서는 이 아이디어가 더 멀리 갑니다. 두 pool이 서로 다른 parallelism을, 심지어 서로
다른 GPU 종류를 쓸 수도 있습니다. compute-bound인 prefill과 memory-bound인 decode에 각각
맞춰서요.

## 부하를 올리면

GPU 네 장짜리 두 설계에 대해 도착률을 쓸어 봅시다.
<!-- cell -->
Disaggregation은 어느 부하에서도 TPOT을 평평하게 유지하는 반면, chunked 설계의 tail
TPOT은 decode step에 prefill chunk가 더 많이 들어오면서 꾸준히 자랍니다. TTFT는 정반대
이야기를 합니다. 하나뿐인 prefill replica가 먼저 포화되어, 8 req/s에서는 queue가 너무
길어져 TTFT가 수 초에 이릅니다. 반면 GPU 네 장 모두에서 prefill할 수 있는 chunked
설계는 아직 3초 아래입니다.

그래서 어느 설계가 나은지는 무엇을 중요하게 보느냐에 달려 있습니다. 사용자가 주로 긴
답변을 streaming으로 받고 끊김에 불평한다면 disaggregation이 매력적입니다. 답변이 얼마나
빨리 시작되는지가 더 중요하거나 cluster가 뜨겁게 돌아간다면, chunked 설계가 하드웨어를 더
유연하게 씁니다.

:::{admonition} 직접 해보기
:class: exercise
**Goodput.** SLO가 p99 TTFT < 2 s, p99 TPOT < 60 ms라고 합시다. 이 sweep으로 chunked
설계와 PD 1:1의 goodput(2강)을 구해 보세요. 다음 절을 읽고 나면, 여덟 장에서 chunked
설계(replica 4개)와 PD 3:1에 대해, 더 높은 부하도 sweep에 추가해서 반복해 보세요. 어느
설계가 이기고, 그 답은 GPU 수에 따라 달라지나요?
:::

## 비율이 중요하다: GPU 여덟 장

GPU 네 장에서는 가능한 분할이 1:1뿐이었습니다. 실제 배포에서는 각 단계에 replica를 몇 개씩
줄지 고릅니다. GPU 여덟 장(replica 네 개)이면 chunked 설계와 세 가지 분할을 비교할 수
있습니다. 올바른 분할은 workload에 달려 있으므로, 두 가지 workload를 돌립니다. 평소의
것(문서 15%)과 문서가 많은 것(문서 40%), 둘 다 8 req/s입니다.
<!-- cell -->
세 가지 교훈:

1. **분할이 맞으면 disaggregation이 두 지표 모두에서 이깁니다.** 평소 workload에서 PD
   3:1은 같은 여덟 장의 GPU 위에서 chunked 설계보다 tail TTFT도 낮고 tail TPOT은 훨씬
   낮습니다. 여기서는 prefill이 비싼 단계이므로(3,800 token짜리 prompt 하나가 decode
   token 수백 개만큼의 연산을 먹습니다) GPU를 더 많이 가질 자격이 있습니다.
2. **분할이 틀리면 크게 집니다.** PD 1:3은 prefill pool을 굶겨서 TTFT가 수 초(문서가
   많으면 수십 초)로 폭발하는데, decode replica 세 개는 workload가 필요로 하는 것보다
   훨씬 많은 용량을 들고 놀고 있습니다.
3. **올바른 분할은 workload에 달려 있습니다.** PD 2:2는 평소 workload에서는 괜찮지만,
   문서 비율이 40%로 오르면 TTFT가 수 초로 뜁니다. 반면 chunked 설계는 어느 GPU나 어떤
   일이든 할 수 있으므로 완만하게 나빠집니다. 하루 중 트래픽 구성이 바뀌는 실제 배포라면,
   pool을 다시 조정하거나 불균형한 시간대를 감수해야 합니다.

:::{admonition} 직접 해보기
:class: exercise
**최적 분할 찾기.** 위의 두 workload에 채팅만 있는 workload(`mixed_trace(0.0)`)를
추가하고 다시 돌려 보세요. 이제 어떤 분할이 가장 좋은가요? TTFT와 TPOT 둘 다에서
chunked 설계를 이기는 PD 분할이 있나요? 돌리기 전에 먼저 예상을 적어 보세요.
:::

## 보호할 것이 없을 때

Disaggregation은 긴 prefill을 decode에서 떼어 놓으려고 존재합니다. Prompt가 전부 짧다면
어떨까요? GPU 네 장에서 채팅 turn만 돌리면서, 각 replica가 얼마나 바쁜지(step을 돌리는
시간의 비율)도 함께 재 봅시다.
<!-- cell -->
Prompt가 짧으면 chunked 설계에는 애초에 interference가 거의 없으므로, disaggregation은
TPOT을 조금밖에 못 개선하고 tail TTFT는 오히려 나쁩니다. 한편 PD의 prefill replica는
대부분의 시간을 놀고 있습니다. 네 장 중 두 장이 주로 일을 기다리는 동안, 하나뿐인
decode replica가 모든 decode를 혼자 감당합니다. 부하가 더 오르면 그 decode replica가 먼저
차오르는 반면, chunked 설계는 여전히 decode를 네 장 모두에 흩을 수 있습니다.

:::{note}
Simulator에서는 decode pool이 과부하되면 요청을 queue에 넣는 대신 `batch_size_cap` 관련
에러를 내며 실행이 멈춥니다. 연습문제에서 이 에러를 보면 `qps`를 낮추거나 decode replica를
늘리세요.
:::

## KV cache를 옮기는 비용

아직 숫자에 드러나지 않은 비용이 하나 있습니다. **KV cache 전송**입니다. 요청의 KV
cache는 prefill replica에서 decode replica로 이동해야 합니다. 우리 모델에서 token 하나의
KV cache는 256 KB이므로(2강):
<!-- cell -->
(이 숫자들은 KV cache 전체가 링크 하나를 지나간다고 가정한 것입니다. TP=2에서는 각 GPU가
자기 절반을 병렬로 보냅니다.) 서버 안의 빠른 링크에서는 전송이 몇 ms면 끝납니다. 서버
사이라면 긴 prompt의 KV cache를 옮기는 데 수십에서 수백 ms가 걸립니다. 그래도 그것을 만든
prefill(3,800 token에 약 650 ms)보다는 훨씬 짧으므로, 실제 시스템은 prefill이 돌아가는
동안 KV cache를 layer 단위로 **streaming**해서 대부분을 가립니다. Simulator도 같은 겹침을
가정하기 때문에 우리 결과에는 전송이 드러나지 않았습니다. 다만 네트워크가 느리거나 겹침이
덜하면, 전송 시간은 두 번째 token까지의 시간에 그대로 더해지고, 언제나 cluster가 다른
트래픽에 써야 할 네트워크 대역폭을 잡아먹습니다.

## 정리

| | Chunked prefill (3강) | PD disaggregation |
|---|---|---|
| Interference | 줄어듦, `chunk_size`로 조절 | 없어짐 |
| TPOT / TBT tail | 부하와 긴 prompt에 따라 자람 | 평평, 순수 decode step 속도 |
| TTFT | 어느 replica나 prefill 가능 | prefill pool만 가능. 작게 잡으면 queue가 생김 |
| 하드웨어 사용 | 어느 GPU나 어떤 일이든, 구성 변화에 적응 | 고정 분할. 한쪽이 노는 동안 다른 쪽이 과부하될 수 있음 |
| 튜닝 | 설정 하나가 두 단계를 모두 맞춰야 함 | 각 pool을 자기 단계에 맞게 설정(chunk size, parallelism, GPU 종류) |
| 추가 비용 | 없음 | KV cache 전송. 빠른 interconnect 필요 |
| 유리한 때 | 짧은 prompt, 변하는 트래픽 구성, TTFT가 가장 중요할 때 | 긴 prompt, 빡빡한 TBT/TPOT 목표, 구성이 안정적이고 분할을 잘 잡았을 때 |

## 이해도 확인

답을 눌러 확인하세요. 틀리면 다시 시도할 수 있습니다.
