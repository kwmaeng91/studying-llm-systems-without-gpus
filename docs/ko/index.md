# GPU 없이 LLM 시스템 공부하기

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

**GPU가 없어도 따라할 수 있는, LLM serving system에 대한 실습 중심 강의입니다.**

이 강의는 한국어와 영어로 모두 제공됩니다. 왼쪽 사이드바 위의
**English / 한국어** 버튼을 눌러 언어를 바꿀 수 있습니다. 참고로, 제 모국어는
한국어이지만 이 강의는 영어로 썼고 한국어 번역은 Claude가 했습니다 🌏 (그래서 말투가 좀 트*어드바이저 페이지 같습니다).

실습은 LLM inference cluster를 흉내 내는 **CPU 전용 simulator** 위에서 돌아갑니다.
실제 A100, H100 GPU에서 측정한 profile을 바탕으로 각 연산의 실행 시간을 예측하기
때문에, 가지고 있지 않은 하드웨어로도 실험해 볼 수 있습니다. 여기서 쓰는 simulator는
[Vidur-Agent](https://github.com/psu-paws/Vidur-Agent)
([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725); 참고로 저희 연구실에서
만들었습니다!)이고, 이는 Microsoft의
[Vidur](https://github.com/microsoft/vidur)
([Agrawal et al., MLSys '24](https://arxiv.org/abs/2405.05465))를 확장한 것입니다.

각 강의는 설명과 실행 가능한 notebook이 한 쌍으로 되어 있습니다. 실습 페이지 위쪽의
**Open in Colab**이나 **Launch Binder**를 누르면 아무것도 설치하지 않고 브라우저에서
바로 실행할 수 있습니다. {doc}`lectures/00-getting-ready`부터 시작하세요.

이 웹사이트와 강의 자료는 Claude Code의 도움을 받아 만들었습니다. 사실과 다른 내용이
없는지 최대한 확인하고 고쳤지만, 혹시 이상한 부분이나 제안할 점을 발견하시면 꼭
알려주세요!

```{list-table}
:header-rows: 1
:widths: 6 40 54

* - #
  - 강의
  - 주제
* - 0
  - {doc}`lectures/00-getting-ready`
  - 미리 알아두면 좋은 것들과 배경 자료
* - 1
  - {doc}`lectures/01-prefill-and-decode`
  - TTFT/TPOT, prefill이 compute-bound이고 decode가 memory-bound인 이유
* - 2
  - {doc}`lectures/02-batching-and-load`
  - batching, throughput–latency, goodput
* - 3
  - {doc}`lectures/03-chunked-prefill`
  - chunked prefill, prefill–decode interference
* - 4
  - {doc}`lectures/04-pd-disaggregation`
  - prefill–decode disaggregation과 chunked prefill 비교
* - 5
  - {doc}`lectures/05-prefix-caching`
  - 요청 사이의 KV 재사용, prompt 구성, eviction
* - 6
  - {doc}`lectures/06-agentic-workloads`
  - 실제 agent trace: session, tool call, task 단위 latency
```

:::{note}
7강 (Scheduling Agentic Workloads)은 아직 초안이라 영어로만 제공됩니다:
{doc}`../lectures/07-scheduling-agentic-workloads`.
:::

```{toctree}
:hidden:
:caption: 시작하기

getting-started/index
getting-started/run-in-browser
getting-started/run-locally
```

```{toctree}
:hidden:
:caption: 강의
:maxdepth: 1

lectures/00-getting-ready
lectures/01-prefill-and-decode
lectures/02-batching-and-load
lectures/03-chunked-prefill
lectures/04-pd-disaggregation
lectures/05-prefix-caching
lectures/06-agentic-workloads
```

```{toctree}
:hidden:
:caption: 레퍼런스

reference/python-api
reference/simulator-knobs
reference/how-the-simulator-works
```
