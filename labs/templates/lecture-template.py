# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # N. Lecture Title
#
# One or two paragraphs motivating the topic.
#
# :::{admonition} Learning goals
# - Goal 1
# - Goal 2
# :::

# %% [markdown]
# ## Setup
#
# Keep this cell as-is: it works on Colab, Binder and a local install.

# %%
import os, sys, urllib.request
sys.path.insert(0, os.path.abspath("../../labs"))
try:
    import llm_systems_wo_gpus as lsg
except ImportError:  # outside the course repo (e.g. Colab): fetch the helper
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/kwmaeng91/studying-llm-systems-without-gpus/main/labs/llm_systems_wo_gpus.py",
        "llm_systems_wo_gpus.py")
    import llm_systems_wo_gpus as lsg
lsg.setup()

import matplotlib.pyplot as plt

# %% [markdown]
# ## Section heading
#
# Explanation, then an experiment.

# %%
r = lsg.simulate(qps=2, prefill_tokens=512, decode_tokens=128)
r.summary()

# %% [markdown]
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
# 1. First exercise.
# 2. Second exercise.
# :::
