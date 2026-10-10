# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 7. Agentic Workloads (2): Serving Them
#
# By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖
#
# {doc}`06-agentic-workloads` looked at what agent traffic *is*: dozens of
# dependent requests per task, prompts that grow turn after turn, and long gaps
# while a tool runs. This lecture puts the same two traces on a GPU. How much of
# all that repeated prompt text can the prefix cache actually save? What does that
# buy the person waiting for the task? And when the task is still slow, where did
# its time go?
#
# :::{admonition} Learning goals
# - Measure KV cache hit rates on a real agent trace and explain where the hits
#   come from, and why a single agent and a multi-agent system differ.
# - Explain why a 6× better TTFT can be worth only a quarter of the task time.
# - Take a task apart into prefill, decode, queueing and tool time, and say which
#   of them a serving system can do anything about.
# :::

# %% [markdown]
# ## Setup

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

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({"figure.figsize": (6, 3.5), "axes.grid": True, "grid.alpha": .3})

SYSTEM = dict(model="Qwen/Qwen2.5-32B-Instruct", device="a100", tensor_parallel=2)

# %% [markdown]
# We load the same 24 OWL tasks as last lecture, and 24 MiroThinker tasks. This
# time both have to be simulated, so both are capped: `gaia_sessions` skips a task
# if any single request in it is longer than `max_tokens`, because the course's
# runtime predictor was only fitted that far. OWL's default cap (16,384 tokens) is
# enough; MiroThinker's requests are much longer, so we allow 32,768 and still lose
# the largest tasks. Keep that in mind when reading the MiroThinker numbers: the
# heaviest sessions are not in the sample.

# %%
sessions = lsg.gaia_sessions(num_sessions=24, seed=0)                    # OWL, multi-agent
miro = lsg.gaia_sessions(num_sessions=24, seed=0, agent="mirothinker", max_tokens=32768)
print(f"OWL: {len(sessions)} requests, MiroThinker: {len(miro)} requests")

# %% [markdown]
# ## How much of this is reusable?
#
# Now, let's see how many prefix cache hits these traces get.
# Are these agentic traces reusing a lot of prefixes?
# First, we compute what a perfect, infinitely large
# prefix cache would do, using exactly the rule from lecture 5: chain-hash every
# full 16-token block, then count how many leading blocks of each prompt have been
# seen before.

# %%
def ideal_hit_rate(df, block=16):
    seen, hits, total = set(), 0, 0
    for _, r in df.iterrows():
        ids = json.loads(r.token_ids)
        h, hashes = None, []
        for b in range(len(ids) // block):           # name every full block
            h = hash((h, tuple(ids[b * block:(b + 1) * block])))
            hashes.append(h)
        matched = 0
        for name in hashes[:r.num_prefill_tokens // block]:   # walk the prompt
            if name not in seen:
                break
            matched += 1
        hits += matched * block
        total += r.num_prefill_tokens
        seen.update(hashes)                          # the prompt and the output are cached
    return hits / total

print(f"OWL         : {ideal_hit_rate(sessions):.1%}")
print(f"MiroThinker : {ideal_hit_rate(miro):.1%}")

# %% [markdown]
# About 60% of OWL's prompt tokens and 71% of MiroThinker's have been computed
# before and can benefit from prefix caching. OWL's number roughly matches the
# GAIATrace paper ([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725)).
# These numbers are high, because the system prompt is long, and agents often look at their past conversation history, similar
# to the multi-turn chat behavior from lecture 5.
#
# However, both are lower than what some other papers (like Agentic AI Workload Characteristics,
# [Yuan et al., IISWC '26](https://arxiv.org/abs/2605.26297)) reported,
# where the reported numbers were more like 87--99%.
# Splitting MiroThinker by role shows where that gap comes from.

# %%
pd.DataFrame({role: {"prompt tokens": g.num_prefill_tokens.sum(),
                     "ideal hit rate": ideal_hit_rate(g)}
              for role, g in miro.groupby("role")}).round(3)

# %% [markdown]
# The main agent alone sits at 87%, right in the range those papers report: it is
# one conversation that only ever appends, so almost every prompt is the previous
# prompt plus a little. The summariser is at ~0%. Each of its calls is a different
# scraped page behind the same short instruction, so there is nothing to reuse but
# the instruction itself — and those calls carry a fifth of the prompt tokens, which
# drags the system-wide number down to 71%. A paper that only instruments the main
# agent would report 87% for the very same system.
#
# OWL is lower again, at 60%, for a related reason: as a multi-agent system it runs
# many sub-agents, each starting its own conversation with its own system prompt, so
# there is limited sharing between them.
#
# For context, Mooncake
# ([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079)) reported about 59% for tool- and agent-style
# traffic reaching the Kimi chatbot, and about 40% on ordinary conversation.
# Again, how you designed the agentic system significantly affects the prefix cache hit rate.
# Since we still do not have a consensus on what the right design for an agentic system is, this number will probably fluctuate in the future until we converge to a decision.
#
# :::{admonition} Try it
# :class: exercise
# 1. Rerun `lsg.gaia_sessions` with `seed=1` and `num_sessions=48`, then recompute
#    the ideal hit rate for both systems. How stable is it across samples, and what
#    does that say about tuning a system to one trace?
# 2. Do the same per-role split for OWL. Which worker would you quote if you wanted
#    to publish the most flattering hit rate, and which one is the real problem?
# :::
#
# ## Serving the trace
#
# Now, let's try giving the traces to the simulator.
# Here, we simply assume that every request, whichever sub-agent issued it and whichever
# model served it in the recording, goes to a single replica of Qwen2.5-32B on two A100s —
# the same setup as lectures 3--5, with prefix caching living in that replica's GPU memory.
# The tool latencies come from the trace, so a turn waits exactly as long as the real tool did.
# `lsg.gaia_trace` writes it in the simulator's format, including the token ids and the dependency graph, so turn
# *k+1* is released only after turn *k* finishes and its tool call returns. We send
# one new task every ten seconds to a single replica.
#
# Because a user waits for the whole task, we measure **task completion time**: from
# the moment a session's first request arrives to the moment its last one finishes,
# tool time included.

# %%
def task_times(result):
    """Wall-clock time of each session, from its first arrival to its last completion."""
    q = result.requests.assign(session=lambda d: d["Request Id"] // 1000,
                               done=lambda d: d["request_arrived_at"] + d["request_e2e_time"])
    return q.groupby("session").apply(
        lambda d: d["done"].max() - d["request_arrived_at"].min(), include_groups=False)

out = {}
for name, df in [("OWL", sessions), ("MiroThinker", miro)]:
    trace = lsg.gaia_trace(df, name=name.lower())
    for pc in (False, True):
        r = lsg.simulate(**SYSTEM, trace=trace, num_requests=len(df), qps=0.1,
                         prefix_caching=pc, max_tokens=65536)
        t = task_times(r)
        out[f"{name}, caching {'on' if pc else 'off'}"] = {
            "KV cache hit rate": r.cache_hit_rate,
            "task time p50 (s)": t.median(), "task time p90 (s)": t.quantile(0.9),
            **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT p50 (ms)", "throughput (tok/s)"]]}
pd.DataFrame(out).round(2)

# %% [markdown]
# The measured KV cache hit rates land within a percent of the hand-computed ideals.
# Blocks are evicted during the run (`r.cache` reports tens of thousands), but this
# replica is large enough that the ones it loses are mostly blocks nobody comes back
# to.
#
# Caching helps MiroThinker more than OWL, in every column. Its prompts are far
# longer, so prefill was a bigger share of its time to begin with: TTFT falls 10×
# rather than 6×, and the median task finishes a third sooner rather than a quarter.
# The tails differ too. MiroThinker's p99 TTFT halves, while OWL's does not move at
# all — OWL's worst waits are the fan-out we saw last lecture, a dozen summarise
# requests released at the same instant, and the last of them is slow because it is
# queued behind its own siblings, not because its prefill is expensive. Prefix
# caching does nothing about queueing.
#
# And in both systems the task-level gain is much smaller than the request-level
# one: a 6--10× better TTFT buys a quarter to a third off the task. A task spends
# most of its time generating tokens and waiting for tools, and prefix caching
# touches neither.
#
# :::{admonition} Try it
# :class: exercise
# 1. **Does the cache survive the tool call?** Rerun the comparison with a small
#    KV cache (`kv_blocks=4000`, lecture 5) and look at `r.cache`. How many blocks
#    are evicted, what happens to the hit rate, and do the sessions with the
#    longest tool waits lose more than the others?
# 2. **Chunk size for agents.** These runs used the default `chunk_size=512`.
#    Sweep 512, 2048 and 4096 and report TTFT and task time. Why does a chunk size
#    that was bad for chat (lecture 3) look better here, and does prefix caching
#    being on change the answer?
# :::
#
# ## Where a task's time goes
#
# Let's take one task from each system apart. The bar for each turn runs from its
# arrival to its first token (queueing and prefill) and then to its last token
# (decode); the gaps between turns are tool calls and the agent's own bookkeeping.

# %%
def timeline(df, session, title):
    """Run the trace and draw one session's turns on a wall clock."""
    trace = lsg.gaia_trace(df, name=title.split(":")[0].lower())
    r = lsg.simulate(**SYSTEM, trace=trace, num_requests=len(df), qps=0.1,
                     prefix_caching=True, max_tokens=65536)
    q = r.requests.assign(session=lambda d: d["Request Id"] // 1000,
                          turn=lambda d: d["Request Id"] % 1000)
    s = q[q.session == session].sort_values("turn")
    t0 = s.request_arrived_at.min()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for _, x in s.iterrows():
        a = x.request_arrived_at - t0
        ax.barh(x.turn, x.prefill_e2e_time, left=a, color="C3", height=.7)
        ax.barh(x.turn, x.request_e2e_time - x.prefill_e2e_time,
                left=a + x.prefill_e2e_time, color="C0", height=.7)
    ax.barh(0, 0, color="C3", label="queueing + prefill")
    ax.barh(0, 0, color="C0", label="decode")
    ax.set(xlabel="seconds since the task arrived", ylabel="turn", title=title)
    ax.legend(loc="upper right")
    ax.invert_yaxis()
    fig.tight_layout()

    sub = df[df.session == session]
    return pd.Series({
        "task completion time (s)": (s.request_arrived_at + s.request_e2e_time).max() - t0,
        "summed over its turns: decode (s)": (s.request_e2e_time - s.prefill_e2e_time).sum(),
        "summed over its turns: prefill (s)": s.prefill_time_execution_plus_preemption.sum(),
        "summed over its turns: queueing (s)": s.request_scheduling_delay.sum(),
        # Siblings released by one fan-out all waited for the same tool call, so count it once.
        "waiting for tools (s)": sub.groupby(sub.dep.map(tuple)).tool_time.max().sum(),
    }).round(1)

timeline(sessions, 4, "OWL: one task on the GPU; the gaps are tool calls")

# %% [markdown]
# First, you can see that the task is executed as a chain of queries, mostly sequential but
# sometimes parallel. Many agentic systems are mostly sequential, while people are exploring various
# ways to incorporate more parallel structures (for example LATS,
# [Zhou et al., ICML '24](https://arxiv.org/abs/2310.04406)).
# Second, you can see that the blue (decode) parts dominate. This is because the prefix cache hit
# rate is high, and currently there is no other contention to the GPU.
# With a lower prefix cache hit rate and many other requests on the same GPU, the bars will look different
# (because prefill cannot be batched as nicely as decode, as we learned in lecture 2).
# Again, the twelve parallel tasks in the middle are OWL trying to summarize a very long web-scraped
# text while not significantly increasing the context length.
#
# Note that the decode times summed over the turns (364 s) exceed the task's own
# 320 s: that is only possible because some turns ran at the same time. The 120 s of
# queueing is mostly those siblings waiting for each other.

# %%
timeline(miro, miro.groupby("session").size().idxmax(),
         "MiroThinker: one task on the GPU; the gaps are tool calls")

# %% [markdown]
# The single agent looks completely different. There is one request in flight at a
# time, so the bars form a staircase and the summed decode (587 s) is *less* than the
# task's 775 s — the difference is tool time and the agent's own bookkeeping. Almost
# nothing is spent queueing (7 s), because this task never competes with itself.
#
# That is the uncomfortable part for a serving system. For this task, the GPU was
# never the problem: a faster replica would shorten the blue bars and leave the white
# gaps untouched. What would help is the one thing a single serving replica cannot do
# — run more of the task at once. Everything a serving system can do here is about
# *other* tasks: filling the gaps with someone else's work, so the GPU is busy even
# though no single user gets their answer sooner. That is where
# {doc}`08-scheduling-agentic-workloads` goes next.
#
# :::{admonition} Try it
# :class: exercise
# 1. Measure how long OWL's join turn (the one whose `dep` lists twelve turns) waits
#    after the *first* of its dependencies finishes. How much of that is queueing
#    behind its own siblings, and what would a scheduler have to know to shorten it?
# 2. Add up the white gaps in the MiroThinker timeline. If you could serve other
#    tasks during them, how many such tasks would it take to keep the replica busy?
# :::
#
# ## Summary
#
# | | What we saw |
# |---|---|
# | Reusable prompt text | 60% of OWL's prompt tokens, 71% of MiroThinker's; within one agent's conversation it is 87%, across sub-agents and summariser calls far less |
# | Who you measure matters | the same system reports 87% or 71% depending on whether the summariser is in the trace |
# | Prefix caching | 6--10× better TTFT, but only a quarter to a third off the task, because decode and tools are untouched |
# | Queueing | caching does not help a fan-out whose siblings wait for each other |
# | A single-agent task | one request in flight, almost no queueing, and long idle gaps that only *other* tasks can fill |
#
# ## Check your understanding
#
# Click an answer to check it. Wrong answers can be retried.

# %% cellView="form" tags=["remove-input"]
#@title Questions (run this cell)
lsg.quiz([
    {"q": "MiroThinker's main agent reaches an 87% ideal hit rate, but the system as a whole only 71%. Why?",
     "options": ["The block size is too large for the main agent",
                 "A fifth of the prompt tokens are summariser calls, each a different scraped page behind the same short instruction, which share almost nothing",
                 "The main agent's cache is evicted between turns"],
     "answer": 1,
     "explain": "Reuse follows the agent's structure. Papers that only instrument the main loop report the 87% number for the very same system."},
    {"q": "Why is OWL's hit rate (60%) lower than MiroThinker's main agent (87%)?",
     "options": ["OWL's prompts are shorter",
                 "Each OWL worker starts its own conversation with a different system prompt, and the requests of a fan-out share only their instructions",
                 "OWL was recorded with caching disabled"],
     "answer": 1,
     "explain": "Within a worker's stretch of turns almost everything repeats; across workers and across the chunks of a fan-out, much less does."},
    {"q": "Prefix caching cut TTFT by 6× but task completion time by only about a quarter. Why?",
     "options": ["The KV cache hit rate was too low",
                 "Most of a task's time is decode and tool calls, which prefix caching does not touch",
                 "Task time is dominated by queueing"],
     "answer": 1,
     "explain": "The task's critical path is a chain of turns, each spending most of its time generating tokens or waiting for a tool."},
    {"q": "Caching halved MiroThinker's p99 TTFT but left OWL's unchanged. What is OWL's tail made of?",
     "options": ["Very long prompts that miss in the cache",
                 "A fan-out of a dozen requests released at once, the last of which queues behind its siblings",
                 "Tool calls that time out"],
     "answer": 1,
     "explain": "That wait is queueing, not prefill, so making prefill cheaper does not shorten it. It needs a scheduler, which is the next lecture."},
    {"q": "A session waits 22 s for a tool. What is the dilemma for the replica holding its KV cache?",
     "options": ["Whether to keep the blocks (idle memory) or evict them (a full re-prefill when the turn returns)",
                 "Whether to decode ahead speculatively",
                 "Whether to switch the session to another model"],
     "answer": 0,
     "explain": "Tool gaps make the reuse distance long. This is the agentic version of lecture 5's eviction trade-off, and it is why cache capacity and offload matter for agents."},
])
