"""Convert labs/src/*.py (jupytext percent format) into executed notebooks.

    python labs/build_notebooks.py            # all lectures
    python labs/build_notebooks.py 01 03      # only lectures whose name starts with these

The executed .ipynb files land in docs/lectures/ and are committed, so the docs
build (and Read the Docs) never has to run the simulator.
"""

import sys
import time
from pathlib import Path

import jupytext
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "labs" / "src"
OUT = ROOT / "docs" / "lectures"


def build(src: Path) -> None:
    nb = jupytext.read(src)
    dst = OUT / (src.stem + ".ipynb")
    t = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT)}}).execute()
    jupytext.write(nb, dst)
    print(f"{src.name} -> {dst.relative_to(ROOT)}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    prefixes = sys.argv[1:]
    for src in sorted(SRC.glob("*.py")):
        if not prefixes or any(src.name.startswith(p) for p in prefixes):
            build(src)
