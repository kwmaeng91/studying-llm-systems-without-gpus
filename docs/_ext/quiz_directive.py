"""``{quiz}`` directive: clickable questions on Markdown pages.

Renders exactly like ``lsg.quiz(...)`` in the lecture notebooks. The body is a
YAML list of questions::

    ```{quiz}
    - q: Is prefill compute-bound or memory-bound?
      options: [Compute-bound, Memory-bound]
      answer: 0          # 0-based index of the correct option
      explain: Each weight is reused for every prompt token.
    ```
"""

import sys
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs"))
from llm_systems_wo_gpus import quiz_html  # noqa: E402


class QuizDirective(Directive):
    has_content = True

    def run(self):
        questions = yaml.safe_load("\n".join(self.content))
        return [nodes.raw("", quiz_html(questions), format="html")]


def setup(app):
    app.add_directive("quiz", QuizDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
