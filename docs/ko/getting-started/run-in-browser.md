# 브라우저에서 실행하기

글쓴이: [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)), [Claude Code](https://claude.com/claude-code) 🤖

모든 실습 페이지 오른쪽 위에는 버튼이 두 개 있습니다:

<span class="launch-btn colab">Open in Colab</span> <span class="launch-btn binder">Launch Binder</span>

## Google Colab (추천)

1. 실습 페이지에서 **Open in Colab**을 누릅니다. GitHub에서 notebook이 바로 열립니다.
2. **런타임 → 모두 실행**을 고르거나, <kbd>Shift</kbd>+<kbd>Enter</kbd>로 셀을 하나씩
   실행합니다.
3. 첫 번째 셀이 `llm_systems_wo_gpus` 패키지와 simulator를 받고 의존성을 설치합니다.
   1분 정도 걸립니다.
4. 수정한 내용을 남기고 싶다면 **파일 → 드라이브에 사본 저장**을 쓰세요.

:::{tip}
GPU 런타임은 **필요 없습니다**. 기본 CPU 런타임이면 충분합니다. GPU 런타임을 켜 봤자
놀기만 하면서 할당량만 깎아먹습니다.
:::

Colab 세션은 일시적입니다. 세션이 다시 시작되면 설치 셀이 다시 돌고, predictor
cache(모델/GPU 조합마다 약 15초)도 다시 만들어집니다.

## Binder (계정 불필요)

1. **Launch Binder**를 누릅니다. [mybinder.org](https://mybinder.org)가 이 강의용
   컨테이너 이미지를 만들거나 재사용해서 JupyterLab을 띄워 줍니다.
2. 이 이미지에는 simulator와 기본 모델용 predictor cache가 이미 들어 있어서, 첫 번째
   simulation이 바로 시작됩니다.

Binder는 무료이고 로그인도 필요 없지만, 이미지를 새로 빌드해야 할 때는 실행까지 몇 분이
걸릴 수 있고, 가만히 두면 10분쯤 뒤에 세션이 종료됩니다. 자리를 뜨기 전에 **notebook을
내려받으세요** (*File → Download*).
