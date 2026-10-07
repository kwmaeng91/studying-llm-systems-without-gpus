# 로컬에서 실행하기

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

**Python 3.10–3.12**, **git**, **4 GB RAM**이 있는 Linux, macOS, Windows(WSL2)
머신이면 충분합니다.

## 1. 강의 자료 받기

```bash
git clone https://github.com/kwmaeng91/studying-llm-systems-without-gpus
cd studying-llm-systems-without-gpus
```

## 2. 환경 만들기

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows (WSL)도 같은 명령
pip install -r requirements-lab.txt jupyterlab
```

## 3. Jupyter 실행

```bash
jupyter lab docs/lectures/
```

아무 `.ipynb`이나 열고 첫 셀을 실행하세요. 처음 실행할 때 `lsg.setup()`이 simulator를
`~/.llm-systems-wo-gpus/backend`에 받아 둡니다.

## 파일이 저장되는 곳

| 경로 | 내용 | 지워도 되나요? |
|---|---|---|
| `~/.llm-systems-wo-gpus/backend/` | simulator 소스와 profiling 데이터 (약 700 MB) | 네, 다음 `setup()` 때 다시 받습니다 |
| `~/.llm-systems-wo-gpus/predictor_cache/` | 학습된 runtime predictor (모델/GPU/TP 조합당 약 70 MB) | 네, 필요할 때 다시 학습합니다 |
| `~/.llm-systems-wo-gpus/runs/` | 매 실행의 simulator 원본 출력 | 네 |
| `~/.llm-systems-wo-gpus/traces/` | `make_trace`가 만든 workload CSV | 네 |

`~/.llm-systems-wo-gpus` 말고 다른 디렉터리를 쓰고 싶다면 `LSG_WORK_DIR` 환경 변수를,
simulator만 옮기고 싶다면 `LSG_BACKEND_DIR`을 설정하세요.

## 문제 해결

`RuntimeError: Simulation failed`
: 에러 바로 위에 simulator의 로그가 찍혀 있습니다. 가장 흔한 원인은 profiling되지 않은
  조합(예: A100에서 TP2로 돌리는 Llama-3-8B)입니다. `lsg.catalog()`로 확인하세요.

새로운 모델이나 GPU의 첫 simulation이 느립니다
: Simulator는 (모델, GPU, TP) 조합마다 runtime predictor를 한 번 학습하고 캐시합니다.
  그다음 실행부터는 몇 초면 끝납니다.

메모리 부족
: 실습에서는 simulator의 prediction grid를 작게 잡아 두었습니다
  (`llm_systems_wo_gpus.py`의 `LITE_GRID` 참고). 이 값을 올리면 메모리 사용량이 금방
  늘어납니다. 원래 기본값은 10 GB 이상을 필요로 합니다.
