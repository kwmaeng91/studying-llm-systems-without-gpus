"""Build the Korean notebooks from the English ones.

The English notebooks in ``docs/lectures`` are the source of truth: they carry the
code and the executed outputs. A Korean notebook is the same notebook with its
markdown cells replaced by the translations in ``labs/ko/NN-name.md`` (cells
separated by a ``<!-- cell -->`` line) and, where a translated quiz exists in
``labs/ko/NN-name.quiz.json``, the quiz cell's HTML output regenerated in Korean.

    python labs/build_korean.py            # every lecture with a translation
    python labs/build_korean.py 05         # only this one
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN = ROOT / "docs" / "lectures"
KO_SRC = ROOT / "labs" / "ko"
KO_OUT = ROOT / "docs" / "ko" / "lectures"
SEP = "<!-- cell -->"

sys.path.insert(0, str(ROOT / "labs"))
import llm_systems_wo_gpus as lsg  # noqa: E402


def build(translation: Path) -> None:
    nb_path = EN / (translation.stem + ".ipynb")
    nb = json.loads(nb_path.read_text())
    blocks = [b.strip("\n") for b in translation.read_text().split(SEP)]
    markdown = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    if len(blocks) != len(markdown):
        raise SystemExit(f"{translation.name}: {len(blocks)} translated cells, "
                         f"{len(markdown)} markdown cells in {nb_path.name}")
    for cell, text in zip(markdown, blocks):
        cell["source"] = [l + "\n" for l in text.split("\n")]
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

    quiz = translation.with_suffix(".quiz.json")
    if quiz.exists():
        html = lsg.quiz_html(json.loads(quiz.read_text()))
        for cell in nb["cells"]:
            for out in cell.get("outputs", []):
                if "text/html" in out.get("data", {}) and "lsg-quiz" in "".join(out["data"]["text/html"]):
                    out["data"]["text/html"] = [html]
                    out["data"].pop("text/plain", None)

    KO_OUT.mkdir(parents=True, exist_ok=True)
    (KO_OUT / nb_path.name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{translation.name} + {nb_path.name} -> docs/ko/lectures/{nb_path.name}")


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for f in sorted(KO_SRC.glob("*.md")):
        if f.name.endswith(".quiz.json"):
            continue
        if not prefixes or any(f.name.startswith(p) for p in prefixes):
            build(f)
