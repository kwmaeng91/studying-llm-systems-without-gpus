# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [TEMP] 7. Hardware and Tensor Parallelism
#
# By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖
#
# Up to now everything ran on one A100. Real deployments have to choose:
#
# - **Which GPU?** H100s are faster but cost more per hour.
# - **How many GPUs per model replica?** A 70B model (140 GB in fp16) does not
#   fit on one 80 GB GPU, so it *must* be split. Even a model that fits can be
#   split to cut latency.
#
# **Tensor parallelism (TP)** splits every weight matrix across $k$ GPUs. Each GPU
# then reads $1/k$ of the weights per step, so memory-bound decode gets up to
# $k\times$ faster, but the GPUs must **all-reduce** their partial results twice per
# transformer layer. The simulator models both effects from profiled kernel and NCCL
# runtimes.
#
# :::{admonition} Learning goals
# - Predict how TP changes latency, throughput, and per-GPU efficiency.
# - Explain where the KV-cache memory goes and why TP also *adds* capacity.
# - Choose a deployment for a given SLO and cost budget.
# :::

# %% [markdown]
# ## Setup

# %%
import os, sys, urllib.request
sys.path.insert(0, os.path.abspath("../../labs"))
try:
    import llm_systems_wo_gpus as lsg
except ImportError:
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/kwmaeng91/studying-llm-systems-without-gpus/main/labs/llm_systems_wo_gpus.py",
        "llm_systems_wo_gpus.py")
    import llm_systems_wo_gpus as lsg
lsg.setup()

import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({"figure.figsize": (6, 3.5), "axes.grid": True, "grid.alpha": .3})

# %% [markdown]
# ## What can we simulate?
#
# The simulator can only simulate (model, GPU, TP) combinations that someone profiled on
# real hardware. `lsg.catalog()` lists them:

# %%
lsg.catalog()

# %% [markdown]
# ## Sweeping GPU type and TP degree
#
# For each configuration we measure:
#
# - **Latency** at low load (one request at a time): TTFT and TPOT.
# - **Capacity**: offline throughput with 256 requests queued at once, total and
#   per GPU. Per-GPU throughput is a proxy for cost efficiency.
# - **KV-cache capacity**: how many tokens of context fit in memory after the weights.

# %%
def characterize(model, device, tp):
    lo = lsg.simulate(model=model, device=device, tensor_parallel=tp, qps=0.1, num_requests=10)
    hi = lsg.simulate(model=model, device=device, tensor_parallel=tp, qps=None, num_requests=256)
    s_lo, s_hi = lo.summary(), hi.summary()
    return {"model": model.split("/")[-1], "GPU": device, "TP": tp,
            "TTFT (ms)": s_lo["TTFT p50 (ms)"], "TPOT (ms)": s_lo["TPOT p50 (ms)"],
            "tok/s": s_hi["throughput (tok/s)"],
            "tok/s per GPU": s_hi["throughput (tok/s)"] / tp,
            "KV tokens": lo.kv_cache_tokens}

rows = []
for device in ("a100", "h100"):
    for tp in (1, 2, 4, 8):
        rows.append(characterize("meta-llama/Llama-2-7b-hf", device, tp))
    for tp in (2, 4, 8):
        rows.append(characterize("meta-llama/Llama-2-70b-hf", device, tp))
hw = pd.DataFrame(rows).round(1)
hw

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for (model, device), g in hw.groupby(["model", "GPU"]):
    style = "-" if device == "a100" else "--"
    ax[0].plot(g["TP"], g["TPOT (ms)"], "o" + style, label=f"{model} / {device}")
    ax[1].plot(g["TP"], g["tok/s per GPU"], "o" + style, label=f"{model} / {device}")
ax[0].set(xlabel="tensor-parallel degree", ylabel="TPOT (ms)", title="Latency", yscale="log")
ax[1].set(xlabel="tensor-parallel degree", ylabel="tok/s per GPU", title="Efficiency", yscale="log")
for a in ax:
    a.set_xticks([1, 2, 4, 8])
ax[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# Things to notice:
#
# 1. **TP lowers latency, sub-linearly.** Going from TP2 to TP8 on Llama-2-70B cuts
#    TPOT by about 2–2.6×, not 4×, because each all-reduce costs a roughly fixed time
#    regardless of how small the shards get.
# 2. **TP lowers per-GPU efficiency.** Each GPU does less useful math per step, and
#    communication time does not shrink. If you only care about throughput per
#    dollar and the model fits, use the *smallest* TP and add replicas instead
#    (lecture 8).
# 3. **H100 vs. A100** is 1.6–2.8× faster across these metrics. Memory-bound
#    decode (TPOT) gains the least, roughly in line with bandwidth (3.35 vs.
#    2.0 TB/s). Compute-bound prefill (TTFT) gains the most, from the H100's ~3×
#    FLOPs.
# 4. **KV-cache capacity grows super-linearly with TP.** 70B at TP2 has 160 GB of
#    HBM and 140 GB of weights, which leaves room for only ~54k tokens of KV
#    cache, i.e. a few dozen concurrent long chats. At TP4 the same weights are
#    spread over 320 GB, so KV capacity grows ~10×. A larger KV cache allows
#    larger batches. Our short 640-token requests never fill even TP2's cache,
#    which is why TP2 looks so efficient here. Exercise 1 asks what happens
#    with long contexts.
#
# :::{note}
# The per-token KV size is
# $2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times 2\ \text{B}$.
# Llama-2-70B uses grouped-query attention with 8 KV heads, so it needs
# $2 \times 80 \times 8 \times 128 \times 2 = 320$ KB per token, *less* than
# Llama-2-7B's 512 KB, despite having 10× the parameters.
# :::

# %% [markdown]
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
# 1. Rerun the 70B rows with long requests (`prefill_tokens=3000, decode_tokens=500`)
#    and `batch_size_cap=128`. Does TP2 still win on tokens/s per GPU? Look at
#    `kv_cache_tokens` to explain why or why not.
# 2. Price the configurations: assume $2/h for an A100 and $4/h for an H100. Which
#    deployment of Llama-2-70B gives the lowest **cost per million output tokens**
#    while keeping TPOT under 30 ms?
# 3. Compare one replica at TP8 against two replicas at TP4 (`num_replicas=2`) for
#    Llama-2-70B at 8 req/s. Which has better p99 TTFT? Which has better TPOT?
# :::
