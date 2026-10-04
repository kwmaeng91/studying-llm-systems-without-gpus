# LLM Systems Without GPUs

A hands-on lecture series on LLM inference serving where every experiment runs on a
CPU, using the [Vidur-Agent](https://github.com/psu-paws/Vidur-Agent) simulator.

- **Read online:** https://studying-llm-systems-without-gpus.readthedocs.io *(after the RTD project is created)*
- **Run a lab:** the "Open in Colab" / "Launch Binder" buttons on each lab page.

## Layout

```
docs/                  Sphinx site (sphinx_rtd_theme + MyST-NB)
  lectures/*.ipynb     executed lab notebooks (generated; do not edit by hand)
  lectures/*.md        lectures without code
  getting-started/     how to run in Colab / Binder / locally
  reference/           vidur_lab API, simulator flags, how Vidur works
labs/
  vidur_lab.py         the helper every notebook imports
  src/*.py             lab sources (jupytext "percent" format), edit these
  templates/           starting point for a new lecture
  build_notebooks.py   src/*.py -> executed docs/lectures/*.ipynb
binder/                mybinder.org environment (+ predictor cache warm-up)
```

## Authoring workflow

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lab.txt -r requirements-docs.txt jupytext nbclient

python labs/build_notebooks.py          # execute all labs
python labs/build_notebooks.py 03       # just lecture 3
sphinx-build -b html docs docs/_build/html
```

Notebooks are committed **with outputs**, so Read the Docs never runs the
simulator (`nb_execution_mode = "off"`).

See `docs/reference/authoring.md` for adding a lecture.

## Changing the repo location

The GitHub location `kwmaeng91/studying-llm-systems-without-gpus` is used in
`docs/conf.py` (Colab/Binder buttons) and in each lab's setup cell (where Colab
downloads `vidur_lab.py` from). If the repo moves, update both:

```bash
grep -rl "kwmaeng91/studying-llm-systems-without-gpus" docs labs README.md
```
