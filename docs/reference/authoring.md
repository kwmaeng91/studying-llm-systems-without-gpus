# Adding a Lecture

Lectures live in two places:

| File | Role |
|---|---|
| `labs/src/NN-name.py` | **source**, edited by hand ([jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html)) |
| `docs/lectures/NN-name.ipynb` | executed notebook, generated, committed, and rendered by the site |

A lecture with no code can be a plain `docs/lectures/NN-name.md` instead.

## Steps

1. Copy the template:
   ```bash
   cp labs/templates/lecture-template.py labs/src/03-my-topic.py
   ```
2. Write it. Markdown cells start with `# %% [markdown]`, code cells with `# %%`.
   Keep the setup cell unchanged: it is what makes the notebook work on Colab,
   Binder, and locally.
3. Execute it into the site:
   ```bash
   python labs/build_notebooks.py 03
   ```
4. Add it to the table in `docs/index.md`. The sidebar picks up every file in
   `docs/lectures/` automatically, sorted by name.
5. Preview:
   ```bash
   sphinx-build -b html docs docs/_build/html && xdg-open docs/_build/html/index.html
   ```

Lab pages (`.ipynb`) automatically get **Open in Colab** / **Launch Binder**
buttons. Those links point to the GitHub repo set in `docs/conf.py`, so they only
work once the notebook is pushed.

## Useful MyST syntax

````md
:::{admonition} Try it
:class: exercise
1. An exercise, rendered as a blue box.
:::

:::{note}
A note.
:::

Inline math $x^2$, display math:
$$ \text{TPOT} \approx \frac{\text{weight bytes}}{\text{bandwidth}} $$

Cross-reference another page: {doc}`01-prefill-and-decode`
````

Inside a `.py` source, prefix every markdown line with `# `.

## Check-your-understanding questions

End a lecture with `lsg.quiz(...)`. It renders clickable multiple-choice questions
with instant feedback, in Jupyter, in Colab, and on this website:

```python
# %% cellView="form" tags=["remove-input"]
#@title Questions (run this cell)
lsg.quiz([
    {"q": "Is prefill compute-bound or memory-bound?",
     "options": ["Compute-bound", "Memory-bound"], "answer": 0,
     "explain": "Each weight is reused for every prompt token."},
])
```

The `remove-input` tag hides the code (and so the answers) on the website, and
`cellView="form"` folds it away in Colab. Strings may contain HTML, e.g. `<code>`.

On a Markdown page (`.md`, no notebook), use the `{quiz}` directive instead. It
renders identically:

````md
```{quiz}
- q: Is prefill compute-bound or memory-bound?
  options: [Compute-bound, Memory-bound]
  answer: 0
  explain: Each weight is reused for every prompt token.
```
````

## Keeping notebooks light

- Every simulation should finish in seconds and use < 1 GB of RAM, so labs run on
  free Colab/Binder. `llm_systems_wo_gpus` already shrinks the simulator's prediction grid for this.
- Check that the model/GPU/TP combination you want exists with `lsg.catalog()`.
