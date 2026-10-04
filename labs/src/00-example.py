# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example: Your First Simulation
#
# This page checks that your environment works. Click **Open in Colab** or
# **Launch Binder** at the top right to run it in your browser. No GPU is needed:
# the course's simulator models an LLM serving system on the CPU.

# %% [markdown]
# ## Setup
#
# The first run downloads the simulator and installs its dependencies (~1 minute).

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
# ## One simulation
#
# Serve Llama-2-7B on one simulated A100. 100 requests (512 prompt tokens, 128
# output tokens each) arrive at an average of 2 per second. The first call for a
# new model/GPU takes ~15 s while the simulator fits its runtime predictors; later calls
# take a few seconds.

# %%
r = lsg.simulate(model="meta-llama/Llama-2-7b-hf", device="a100",
                qps=2, num_requests=100, prefill_tokens=512, decode_tokens=128)
r.summary()

# %% [markdown]
# ## A sweep
#
# `lsg.sweep` runs one simulation per value and returns a table, which you can plot.

# %%
df = lsg.sweep("qps", [1, 2, 4, 8, 12, 16], num_requests=200)
plt.plot(df.index, df["TTFT p99 (ms)"], "o-")
plt.xlabel("arrival rate (req/s)"); plt.ylabel("TTFT p99 (ms)"); plt.yscale("log")
plt.grid(alpha=.3);

# %% [markdown]
# :::{admonition} Try it
# :class: exercise
# Change `device="a100"` to `device="h100"` above and rerun. What changes?
# :::
