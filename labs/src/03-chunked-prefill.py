# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [TEMP] 3. Scheduling: Chunked Prefill and the Prefill–Decode Interference
#
# Lecture 2 kept things simple: every request was either prefill-only or
# decode-only. Real requests have both phases, so a serving system constantly faces
# the following situation:
#
# > Several users are in the middle of receiving their answers (their requests are
# > **decoding**). A new request with a long prompt arrives and needs a **prefill**.
# > What should the scheduler run next?
#
# This sounds like a small detail, but the answer decides whether the new user
# waits seconds for their first token, or the existing users see their text
# **stall** mid-sentence. In this lecture we try the obvious answers first, see why
# each of them fails, and then build up to **chunked prefill**, the technique used
# by modern serving systems such as vLLM and SGLang.
#
# :::{admonition} Learning goals
# - Explain why the three "obvious" ways to schedule a new prefill (make it wait,
#   pause decodes, or batch them together) each hurt either TTFT or TPOT.
# - Explain how chunked prefill bounds the length of every step, and why the decodes
#   that share a step with a prefill chunk are almost free.
# - Use `chunk_size` to trade TTFT against TPOT, and explain why the trade-off
#   becomes sharper under heavy load.
# :::

# %% [markdown]
# ## Setup
#
# Same setup as {doc}`01-prefill-and-decode`: Qwen2.5-32B-Instruct on two A100s.

# %%
import os, sys, urllib.request
sys.path.insert(0, os.path.abspath("../../labs"))
try:
    import llm_systems_wo_gpus as lsg
except ImportError:  # outside the course repo (e.g. Colab): fetch the package
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/kwmaeng91/studying-llm-systems-without-gpus/main/labs/llm_systems_wo_gpus.py",
        "llm_systems_wo_gpus.py")
    import llm_systems_wo_gpus as lsg
lsg.setup()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({"figure.figsize": (6, 3.5), "axes.grid": True, "grid.alpha": .3})

SYSTEM = dict(model="Qwen/Qwen2.5-32B-Instruct", device="a100", tensor_parallel=2)

# %% [markdown]
# ## The scenario
#
# Throughout the first half of this lecture we use one small, concrete scenario:
#
# - At $t = 0$, **16 chat users** send a short prompt (256 tokens) and start
#   receiving a 256-token answer.
# - At $t = 2$ s, while all 16 are still decoding, a **document** request arrives:
#   a 3,800-token prompt (e.g., "summarize this report") with a 32-token answer.
#
# Recall the two numbers that matter from lectures 1 and 2. A decode step for the
# 16 chat users takes about 40 ms, because decode is memory-bound and batching 16
# requests costs about the same as one. Prefilling 3,800 tokens takes much longer,
# because prefill is compute-bound and its cost grows with the prompt length. Let's
# measure both:

# %%
chat = lsg.simulate(**SYSTEM, qps=None, num_requests=16, prefill_tokens=256, decode_tokens=256)
doc = lsg.simulate(**SYSTEM, qps=None, num_requests=1, prefill_tokens=3800, decode_tokens=1,
                   chunk_size=4096)
decode_step = chat.tpot.median()   # one decode step for the 16 chat users
prefill_time = doc.ttft.iloc[0]    # one 3,800-token prefill, alone on the GPUs
chat_done = chat.e2e.max()         # when the last chat user finishes (no document)
print(f"decode step (16 users): {1e3 * decode_step:.0f} ms")
print(f"prefill of 3,800 tokens: {1e3 * prefill_time:.0f} ms")
print(f"chat users finish at:    {chat_done:.1f} s")

# %% [markdown]
# The prefill is about 16× longer than a decode step. Whatever the scheduler does,
# this 650 ms of work has to be done somewhere. The question is **who waits for it**.
#
# To run the scenario in the simulator, two `simulate` arguments help:
# `arrival_times` gives every request an exact arrival time (instead of random
# arrivals at a given `qps`), and `keep_steps=True` records every forward pass
# ("step") the GPUs run, which we read back from `r.steps`.

# %%
def run_scenario(chunk_size):
    """16 chat users from t=0, plus a 3,800-token document arriving at t=2 s."""
    r = lsg.simulate(**SYSTEM, num_requests=17, chunk_size=chunk_size, keep_steps=True,
                     prefill_tokens=[256] * 16 + [3800], decode_tokens=[256] * 16 + [32],
                     arrival_times=[0] * 16 + [2.0])
    steps = r.steps
    # The GPUs are never idle in this scenario, so each step starts when the previous one ends.
    steps["end (s)"] = steps["batch_execution_time"].cumsum()
    steps["start (s)"] = steps["end (s)"] - steps["batch_execution_time"]
    return r, steps

# %% [markdown]
# ## What if... the new request waits?
#
# The simplest policy: let the requests that are already running finish, and only
# then start the new one. Early serving systems such as FasterTransformer worked
# this way (*request-level batching*): a batch of requests ran together until all
# of them were done, and only then was the next batch formed.
#
# The chat users are happy: nothing interrupts them, and they keep receiving a token
# every ~40 ms. But the document request has to wait until the last chat user
# finishes before its prefill can even start:

# %%
ttft_wait = (chat_done - 2.0) + prefill_time
print(f"document TTFT if it waits: {ttft_wait:.1f} s")

# %% [markdown]
# Over 9 seconds before the user sees the first word of the summary, and almost all
# of it is spent waiting, not computing. On a busy server it is even worse: there
# is *always* someone decoding, so a new request could wait indefinitely
# (*starvation*). Meanwhile, as the chat users finish one by one, the running batch
# shrinks and the GPUs do less and less useful work per step.
#
# **Verdict: great TPOT, terrible TTFT.**
#
# ## What if... we pause the decodes and prefill right away?
#
# The opposite policy: as soon as a new request arrives, pause everyone who is
# decoding, run the new request's prefill on its own, then resume the decodes. This
# *prefill-first* policy was the default in vLLM before its V1 engine.
#
# Now the document gets its first token quickly: it only waits for the current
# decode step to finish, then runs its 650 ms prefill. But during that prefill,
# **none of the 16 chat users receives a token**. From each user's point of view,
# the text stream freezes:

# %%
stall_pause = prefill_time + decode_step  # no token during the prefill, then the next decode step
print(f"document TTFT: ~{prefill_time:.2f} s")
print(f"longest gap between two tokens for a chat user: {1e3 * stall_pause:.0f} ms "
      f"(normally {1e3 * decode_step:.0f} ms)")

# %% [markdown]
# A 0.7 s freeze in the middle of a sentence is very noticeable, and it is not a
# one-off: on a real server, long prompts keep arriving, and every one of them
# freezes *every* user who is decoding at that moment. This is called a
# **generation stall**, and it shows up as a high tail (p99) TPOT.
#
# **Verdict: good TTFT, but every new prefill stalls everyone else.**
#
# ## What if... we batch the prefill and the decodes together?
#
# Lecture 2 showed that batching is the key to efficiency. So why not put the new
# prefill and the 16 decodes into the *same* step? The decodes then make progress
# during the prefill instead of being paused. Orca (OSDI 2022), which introduced
# scheduling at the granularity of a single step (*iteration-level scheduling*),
# could form such mixed batches.
#
# This is what the simulator's scheduler does when its token budget per step
# (`chunk_size`) is large enough to hold the whole prompt, e.g. 4,096 tokens. Let's
# look at the steps around $t = 2$ s:

# %%
r_mixed, steps_mixed = run_scenario(chunk_size=4096)
cols = ["start (s)", "end (s)", "batch_num_prefill_tokens", "batch_num_decode_tokens",
        "batch_execution_time"]
steps_mixed[(steps_mixed["end (s)"] > 1.95) & (steps_mixed["start (s)"] < 2.8)][cols].round(3)

# %% [markdown]
# Each row is one step. Before the document arrives, every step decodes 16 tokens
# (one per chat user) in ~38 ms. Then one step processes all 3,800 prompt tokens
# **plus** the 16 decode tokens, and takes about 650 ms. After that, the document
# joins the decodes (17 per step).
#
# Two observations:
#
# 1. **The decodes ride along almost for free.** The mixed step takes about as long
#    as the prefill alone (compare with the 652 ms measured above). The prefill keeps
#    the math units busy, and the 16 extra tokens add almost nothing to it.
# 2. **But the stall is still there.** Each chat user still gets only one token in
#    that 650 ms step. A step cannot end before its biggest piece of work is done, so
#    every request in the batch moves at the pace of the slowest one.

# %%
gap_mixed = steps_mixed["batch_execution_time"].max()
print(f"document TTFT: {r_mixed.ttft.iloc[16]:.2f} s")
print(f"longest gap between two tokens for a chat user: {1e3 * gap_mixed:.0f} ms")

# %% [markdown]
# **Verdict: slightly better than pausing, but the stall remains.**
#
# ## What do we actually want?
#
# Looking back at the three attempts:
#
# | Policy | Document TTFT | Chat users' longest gap | Problem |
# |---|---|---|---|
# | 1. New request waits | ~9 s | ~40 ms | new requests starve |
# | 2. Pause decodes, prefill first | ~0.7 s | ~690 ms | everyone stalls |
# | 3. Batch prefill and decodes together | ~0.7 s | ~650 ms | everyone still stalls |
#
# The root cause in policies 2 and 3 is the same: a **single step that is very
# long**, because it contains a whole long prompt. Policy 1 avoids long steps only
# by refusing to make progress on the new request. What we want is:
#
# - **every step stays short**, so decoding users keep receiving tokens at a steady pace;
# - **every step still makes progress on the prefill**, so the new request is not
#   starved.
#
# ## The solution: chunked prefill
#
# Nothing forces us to prefill a prompt in one step. The prompt can be split into
# **chunks**, processed over several consecutive steps. The first chunk computes the
# KV cache of the first tokens; the next chunk attends to that cached KV and appends
# its own, and so on. The result is exactly the same as prefilling the whole prompt
# at once (this is the same mechanism decode uses to attend to earlier tokens).
#
# **Chunked prefill**, introduced by Sarathi-Serve (OSDI 2024) and now used by
# default in vLLM and SGLang, builds every step under a fixed **token budget**
# (`chunk_size` in our simulator):
#
# 1. Every request that is decoding gets its next token first (1 token each).
# 2. The remaining budget goes to prefills. A prompt longer than the remaining
#    budget is cut, and the rest of it continues in the next step.
#
# With a budget of 512 tokens, each step in our scenario holds 16 decode tokens and
# a 496-token chunk of the document:

# %%
r_chunk, steps_chunk = run_scenario(chunk_size=512)
steps_chunk[(steps_chunk["end (s)"] > 1.95) & (steps_chunk["start (s)"] < 2.9)][cols].round(3)

# %%
gap_chunk = steps_chunk["batch_execution_time"].max()
print(f"document TTFT: {r_chunk.ttft.iloc[16]:.2f} s")
print(f"longest gap between two tokens for a chat user: {1e3 * gap_chunk:.0f} ms")

# %% [markdown]
# The document is now prefilled over 8 steps of ~105 ms each. The chat users keep
# receiving a token every ~105–110 ms instead of freezing for 650 ms, and the
# document's TTFT grows only from ~0.67 s to ~0.84 s.
#
# Why is the total cost so small? Look at the step times. A 512-token step takes
# ~105 ms, about the same as prefilling 512 tokens alone (lecture 2), yet it also
# produces 16 decode tokens, which on their own would have needed a ~38 ms step.
# Prefill chunks keep the math units busy, decodes need the memory bandwidth to
# load the weights, and **a step that mixes them uses both**. Chunked prefill gets
# the efficiency of batching (policy 3) without its long steps.
#
# The timeline below puts the four policies side by side. Each bar is one step:
# orange steps only decode, blue steps contain prefill work. The triangle marks the
# document's first token. (The simulator's scheduler always builds steps under a
# token budget, so it cannot run policies 1 and 2 directly. We draw them from the
# step times measured above.)

# %%
def first_step_end_after(t, step):
    return np.ceil(t / step) * step   # decode steps of length `step` from t=0

t0 = first_step_end_after(2.0, decode_step)  # when the step running at t=2 s ends
timelines = {
    "1. new request waits": (
        [(t, decode_step, "decode") for t in np.arange(0, 3.2, decode_step)], ttft_wait + 2.0),
    "2. pause decodes,\n    prefill first": (
        [(t, decode_step, "decode") for t in np.arange(0, t0, decode_step)]
        + [(t0, prefill_time, "prefill")]
        + [(t, decode_step, "decode") for t in np.arange(t0 + prefill_time, 3.2, decode_step)],
        t0 + prefill_time),
}
for name, (r, st) in {"3. batch together\n    (no chunking)": (r_mixed, steps_mixed),
                      "4. chunked prefill\n    (512 tokens/step)": (r_chunk, steps_chunk)}.items():
    timelines[name] = ([(s, d, "prefill" if p > 0 else "decode") for s, d, p in
                        zip(st["start (s)"], st["batch_execution_time"], st["batch_num_prefill_tokens"])],
                       2.0 + r.ttft.iloc[16])

fig, ax = plt.subplots(figsize=(10, 3.6))
for i, (name, (steps, first_token)) in enumerate(timelines.items()):
    for start, dur, kind in steps:
        ax.barh(i, dur, left=start, height=0.6, color="C0" if kind == "prefill" else "C1",
                edgecolor="white", linewidth=0.8)
    if first_token < 3.0:
        ax.plot(first_token, i - 0.45, "kv", ms=8)
    else:
        ax.text(2.98, i, f"first token at {first_token:.1f} s →", ha="right", va="center",
                fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.axvline(2.0, color="k", ls=":", lw=1)
ax.text(2.0, -0.8, "document arrives ", ha="right", va="center", fontsize=8)
ax.set(xlim=(1.8, 3.0), ylim=(len(timelines) - 0.5, -1.1), xlabel="time (s)",
       yticks=range(len(timelines)), yticklabels=list(timelines))
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color="C1", label="decode-only step"),
                   Patch(color="C0", label="step with prefill work"),
                   Line2D([], [], color="k", marker="v", ls="", label="document's first token")],
          loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
fig.tight_layout()

# %% [markdown]
# ## Choosing the chunk size
#
# The token budget is a knob, and it trades one latency against the other:
#
# - A **smaller** budget makes every step shorter, so decoding users see smaller
#   gaps. But the prompt is split into more steps, and every step has a fixed cost:
#   it must load all the weights from memory, even if it processes few tokens. So the
#   prefill takes longer in total, and TTFT grows.
# - A **larger** budget prefills faster but makes steps longer. At 4,096, a whole
#   3,800-token prompt fits in one step, and we are back to policy 3.
#
# Let's run the scenario with different budgets:

# %%
rows = []
for c in [128, 256, 512, 1024, 2048, 4096]:
    r, st = run_scenario(chunk_size=c)
    rows.append({"chunk_size": c, "document TTFT (s)": r.ttft.iloc[16],
                 "chat users' longest gap (ms)": 1e3 * st["batch_execution_time"].max()})
budget = pd.DataFrame(rows).set_index("chunk_size")
budget.round(2)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(budget.index, budget["chat users' longest gap (ms)"], "o-", color="C1")
ax[0].set(ylabel="longest gap between tokens (ms)", title="Chat users: decode stall")
ax[1].plot(budget.index, budget["document TTFT (s)"], "o-", color="C0")
ax[1].set(ylabel="TTFT (s)", title="Document: time to first token")
for a in ax:
    a.set_xscale("log", base=2)
    a.set_xticks(budget.index, [f"{c:,}" for c in budget.index])
    a.minorticks_off()
    a.set_xlabel("chunk_size (token budget per step)")
    a.set_ylim(bottom=0)
fig.tight_layout()

# %% [markdown]
# Going from 4,096 down to 512 shrinks the stall by about 6× at a small cost in
# TTFT. Below that, the returns diminish: a step can never be shorter than a plain
# decode step (~38 ms), while TTFT keeps growing because the fixed per-step cost is
# paid more and more times. In practice, budgets of a few hundred to a few thousand
# tokens are common.
#
# ## Under real load
#
# The scenario above had a single long prompt. On a real server, requests keep
# arriving. We now simulate a mixed workload: 85% of requests are chat turns (256
# tokens in, 256 out) and 15% are documents (3,800 tokens in, 32 out), arriving in
# random order. `lsg.make_trace` writes the per-request lengths to a trace file that
# the simulator replays.

# %%
rng = np.random.default_rng(0)
n = 300
is_long = rng.random(n) < 0.15
trace = lsg.make_trace(prefill_tokens=np.where(is_long, 3800, 256),
                      decode_tokens=np.where(is_long, 32, 256), n=n, name="mixed")
pd.read_csv(trace).value_counts().rename("requests")

# %% [markdown]
# We run the trace at a moderate load (2 req/s) and a heavy load (4 req/s),
# sweeping `chunk_size` from 128 to 4,096 tokens per step. Since the stalls hit
# only some of the tokens, we look at the **tail**: p99 TPOT.

# %%
chunks = [128, 256, 512, 1024, 2048, 4096]
cols = ["TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT p50 (ms)", "TPOT p99 (ms)", "throughput (tok/s)"]
res = {q: lsg.sweep("chunk_size", chunks, **SYSTEM, trace=trace, num_requests=n, qps=q)
       for q in (2, 4)}
res[4][cols].round(0)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for q, df in res.items():
    ax[0].plot(chunks, df["TPOT p99 (ms)"], "o-", label=f"{q} req/s")
    ax[1].plot(chunks, df["TTFT p50 (ms)"], "o-", label=f"{q} req/s")
ax[0].set(ylabel="TPOT p99 (ms)", title="Decode stalls (tail TPOT)", ylim=(0, None))
ax[1].set(ylabel="TTFT p50 (ms)", title="Time to first token (median)", yscale="log")
for a in ax:
    a.set_xscale("log", base=2)
    a.set_xticks(chunks, [f"{c:,}" for c in chunks])
    a.minorticks_off()
    a.set_xlabel("chunk_size (token budget per step)")
    a.legend()
fig.tight_layout()

# %% [markdown]
# Read the heavy-load (4 req/s) curves from right to left:
#
# - **Large budget (4,096).** Every document is prefilled in one step, so every
#   document stalls all the users decoding next to it. p99 TPOT is about 4× a plain
#   decode step.
# - **Medium budget (512).** Documents are split into ~8 chunks. p99 TPOT drops by
#   about 40%, while median TTFT roughly doubles but stays under a second.
# - **Tiny budget (128, 256).** Stalls almost disappear, but the steps are now so
#   small that the fixed per-step cost dominates: the GPUs spend most of their time
#   loading weights to process only a few tokens. Prefill throughput falls below the
#   rate at which new prompts arrive, requests pile up in the queue, and TTFT
#   explodes into many seconds.
#
# At 2 req/s, the GPUs have spare capacity, so even tiny chunks keep up (except
# 128), and the stalls are milder because fewer documents arrive. **Scheduling
# matters most when the system is busy**, and that is also when choosing the budget
# is hardest.
#
# ## Who pays for the stall?
#
# Finally, let's split TPOT by request type, with a medium and the largest budget.

# %%
for c in (512, 4096):
    r = lsg.simulate(**SYSTEM, trace=trace, num_requests=n, qps=4, chunk_size=c)
    kind = np.where(r.requests["request_num_prefill_tokens"] > 1000, "document", "chat")
    print(f"chunk_size={c}")
    print((1e3 * r.tpot.groupby(kind).describe(percentiles=[.5, .99])[["50%", "99%"]])
          .round(1).rename(columns={"50%": "TPOT p50 (ms)", "99%": "TPOT p99 (ms)"}), "\n")

# %% [markdown]
# Look at the **chat** requests first. None of them has a long prompt, yet with the
# large budget their TPOT is about 10% worse, only because they share steps with
# *other users'* long prefills. The **documents** suffer most at the tail: they
# produce only 32 tokens, so a single 650 ms stall caused by another document's
# prefill raises their average TPOT a lot.
#
# Note that TPOT is an *average* over a request's output tokens, so it hides how
# bumpy the stream is. A chat user who sees one 650 ms freeze among 255 smooth
# tokens has a TPOT only a few ms higher, but certainly notices the freeze. This
# is why the scenario above measured the longest gap directly, and why some
# systems also report the *time between tokens* (TBT) distribution.
#
# This **interference** between requests is what makes it hard to give latency
# guarantees on a shared GPU, and chunked prefill is the first line of defense
# against it.
#
# ## Summary
#
# | Policy | TTFT | TPOT | Used by |
# |---|---|---|---|
# | New request waits for decodes | very bad (starvation) | good | FasterTransformer (request-level batching) |
# | Pause decodes, prefill first | good | bad (stalls) | vLLM before V1 |
# | Prefill + decodes in one step | good | bad (stalls) | Orca |
# | **Chunked prefill** | slightly worse than above | good, tunable | Sarathi-Serve, vLLM V1, SGLang |
#
# Chunked prefill bounds the work in every step with a token budget, so a long
# prompt can no longer freeze everyone else, and it fills the leftover compute of
# decode steps with useful prefill work. The budget (`chunk_size`) trades TTFT
# against TPOT, and the trade-off is sharpest under heavy load.
#
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
# 1. In the scenario, replace the 3,800-token document with a 1,000-token one. How
#    long is the stall with `chunk_size=4096`? With 512? Is chunking still worth it?
# 2. Raise the fraction of documents in the mixed workload to 40%. Does the best
#    `chunk_size` for p99 TPOT change?
# 3. Suppose your SLO is p99 TTFT < 2 s **and** p99 TPOT < 100 ms. Find the
#    largest `qps` you can sustain, and the `chunk_size` that achieves it.
# 4. A different fix is to never mix the two kinds of work: run prefills and
#    decodes on *separate* GPUs. This is prefill–decode disaggregation, covered in
#    {doc}`06-pd-disaggregation`. What new cost does it introduce?
# :::
#
# ## Check your understanding
#
# Click an answer to check it. Wrong answers can be retried.

# %% cellView="form" tags=["remove-input"]
#@title Questions (run this cell)
lsg.quiz([
    {"q": "A server makes every new request wait until all currently decoding requests finish. What goes wrong?",
     "options": ["TPOT of the decoding requests becomes very high",
                 "New requests can wait a very long time for their first token",
                 "The KV cache overflows"],
     "answer": 1,
     "explain": "The running requests are never interrupted, so their TPOT is fine. But a new request cannot even start its prefill until they all finish, and on a busy server someone is always decoding."},
    {"q": "A 4,000-token prompt arrives while 16 requests are decoding. The scheduler pauses the decodes and runs the prefill alone. What do the 16 users see?",
     "options": ["Nothing changes", "Their text stream freezes for about as long as the prefill takes",
                 "Their TTFT increases"],
     "answer": 1,
     "explain": "None of them receives a token during the prefill. Their TTFT is already past; what grows is the gap between two of their tokens (a generation stall)."},
    {"q": "Putting a whole 4,000-token prefill and 16 decodes into the same step…",
     "options": ["removes the stall, because the decodes make progress in the same step",
                 "still stalls the decodes, because the step lasts as long as the prefill",
                 "makes the prefill much slower"],
     "answer": 1,
     "explain": "The decodes do ride along almost for free, but each of them still gets only one token in a step that takes as long as the whole prefill."},
    {"q": "Why do decodes that share a step with a prefill chunk cost almost nothing extra?",
     "options": ["Decode tokens skip the attention layers",
                 "The step already loads all the weights for the chunk, so the decodes reuse them, and they add only a few FLOPs",
                 "The simulator ignores decode tokens in mixed steps"],
     "answer": 1,
     "explain": "Decode is memory-bound: its cost is loading the weights. In a mixed step that cost is already paid by the prefill chunk, and 16 extra tokens are tiny next to a 500-token chunk's arithmetic."},
    {"q": "Splitting a prompt into chunks over several steps changes the model's output.",
     "options": ["True", "False"], "answer": 1,
     "explain": "Each chunk attends to the KV cache written by earlier chunks, so the result is the same as prefilling the whole prompt at once."},
    {"q": "You reduce chunk_size from 4,096 to 512. What happens to p99 TPOT and to the TTFT of long prompts?",
     "options": ["Both decrease", "p99 TPOT decreases, long-prompt TTFT increases somewhat",
                 "p99 TPOT increases, long-prompt TTFT decreases", "Both increase"],
     "answer": 1,
     "explain": "Shorter steps mean shorter stalls for decoding users, but a long prompt now needs several steps to prefill."},
    {"q": "Under heavy load, a tiny chunk_size (e.g., 128) makes TTFT explode. Why?",
     "options": ["Every step must load all the weights, so tiny steps waste most of the GPU time and prefill throughput falls below the arrival rate",
                 "The KV cache cannot hold 128-token chunks",
                 "Decode requests are starved"],
     "answer": 0,
     "explain": "A step has a fixed cost of loading the weights. With tiny budgets that cost is paid for very few tokens, so prompts arrive faster than they can be prefilled, and the queue grows."},
    {"q": "Chat requests with short prompts get a worse p99 TPOT when the workload also contains long documents. Why?",
     "options": ["Their own prompts become longer",
                 "They share steps with other users' long prefills (interference)",
                 "Long documents use a different model"],
     "answer": 1,
     "explain": "A step is only as fast as its biggest piece of work. When a long prefill lands in a step, every request in that step waits for it."},
])
