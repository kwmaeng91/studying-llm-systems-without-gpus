# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 6. Agentic Workloads
#
# By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖
#
# Every workload so far has been a *request*: a prompt arrives, an answer streams
# back, the request is gone. An agent does not work that way. Given one task — "find
# which main course disappeared from this restaurant's menu between two dates" — it
# issues dozens of LLM requests: it plans, picks a worker, calls a search tool,
# reads the result, calls another tool, writes code, runs it, reads the error,
# tries again, and finally answers. The user waits for the whole chain.
#
# That changes the serving problem in three ways:
#
# - **The requests are dependent.** Turn *k+1* cannot start until turn *k* has
#   finished *and* its tool call has returned. The system cannot decide to run
#   them in parallel; the agent decides.
# - **The prompts are enormous and repetitive.** Each turn resends the whole
#   conversation plus the new tool output, so prompt tokens outnumber generated
#   tokens by an order of magnitude — and almost all of them were prefilled a
#   moment ago (lecture 5).
# - **The metric users care about is the task, not the request.** A p99 TTFT of
#   200 ms means nothing if the task takes six minutes.
#
# This lecture uses **real traces**: every LLM request that the
# [OWL](https://github.com/camel-ai/owl) multi-agent system issued while solving
# tasks from the [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)
# benchmark, with the prompt and output token ids, the dependencies between turns,
# and separately measured tool latencies. They were recorded as
# [GAIATrace](https://github.com/psu-paws/Vidur-Agent) (Kim et al., IISWC 2026) and
# ship with the simulator, so they are already on your disk after `lsg.setup()`.
#
# :::{admonition} Learning goals
# - Describe the structure of an agent session: turns, roles, dependencies,
#   fan-out, and tool time.
# - Explain why agent traffic is prompt-heavy in tokens but decode-heavy in time.
# - Measure prefix-cache hit rates on a real agent trace and explain where the
#   hits come from.
# - Reason about task-level latency: what the critical path is, and which
#   serving-system knobs can and cannot shorten it.
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
# `lsg.gaia_sessions` loads recorded sessions, one row per LLM request. We take 24
# of them. (The recorded tasks include requests of over 100,000 tokens; the
# course's runtime predictor is fitted up to a smaller context, so sessions whose
# longest request does not fit are skipped. The traces are a sample of real
# behaviour, not a benchmark to compare systems with.)

# %%
sessions = lsg.gaia_sessions(num_sessions=24, seed=0)
print(f"{sessions.session.nunique()} sessions, {len(sessions)} LLM requests")
sessions.drop(columns="token_ids").head(8)

# %% [markdown]
# Each row is one request to the model:
#
# - `session` / `turn`: which task, and the position in it.
# - `role`: which agent inside the system issued it. OWL is a *multi-agent* system:
#   a planner splits the task, a coordinator assigns each subtask to a worker, and
#   the workers (web search, code, browser, document reader) do it.
# - `dep`: the turns that must finish before this one is released.
# - `tool_time`: how long the tool call that this turn waited for actually took,
#   measured by re-running the tool outside the agent.
#
# :::{note}
# The traces are released under CC-BY-4.0 as part of GAIATrace. The tasks
# themselves come from the GAIA benchmark, whose
# [dataset terms](https://huggingface.co/datasets/gaia-benchmark/GAIA) govern its
# questions; what follows are short excerpts from one recorded run, shown so you
# can see what a real agent prompt is made of.
# :::
#
# ## What the model actually sees
#
# The traces carry token ids, so we can read the requests back. Let's look at the
# first turns of one session. (`lsg.decode` uses the same tokenizer the traces were
# built with; it downloads a small vocabulary file the first time.)

# %%
def show(i, head=0, tail=0, out=400):
    """Print a trimmed view of request i's prompt and of what the model generated."""
    r = sessions.iloc[i]
    ids = json.loads(r.token_ids)
    prompt, output = lsg.decode(ids[:r.num_prefill_tokens]), lsg.decode(ids[r.num_prefill_tokens:])
    print(f"=== session {r.session} turn {r.turn} | role: {r.role} | "
          f"{r.num_prefill_tokens:,} prompt + {r.num_decode_tokens} output tokens "
          f"| tool wait before it: {r.tool_time:.1f} s")
    if head:
        print("--- prompt, first lines " + "-" * 40); print(prompt[:head].strip())
    if tail:
        print("--- prompt, last lines " + "-" * 41); print(prompt[-tail:].strip())
    print("--- generated " + "-" * 50); print(output[:out].strip(), "\n")

show(0, head=620, out=560)

# %% [markdown]
# That is the planner: it receives the user's task and a description of the
# available workers, and emits a list of subtasks. Everything is marked up with
# the model's chat template (`<|start|>`, `<|message|>`, `<|end|>`); the
# `<|channel|>final` and `<|channel|>analysis` markers separate the answer from the
# model's own reasoning.
#
# Two turns later, a web-search worker decides to call a tool. Its entire output is
# 36 tokens of JSON — the call itself:

# %%
show(2, out=300)

# %% [markdown]
# The agent executes that call, which takes 1.2 s, and sends the *result back as
# part of the next prompt*. Here is the end of the next request's prompt, which is
# exactly the previous prompt, plus the previous output, plus the tool result:

# %%
show(3, tail=430, out=330)

# %% [markdown]
# This is the mechanism behind everything in this lecture. Nothing is remembered
# between requests, so each turn re-sends the whole conversation. The prompt grows
# monotonically along a branch, and almost every prompt is a previous prompt with
# something appended — which is to say, almost every prompt is a prefix-cache hit
# waiting to happen.
#
# ## The shape of an agent session
#
# Let's look at one whole session: how the prompt grows, which role issues each
# turn, and where the tool time goes.

# %%
one = sessions[sessions.session == 4]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax2 = ax.twinx()
ax2.bar(one["turn"], one["tool_time"], color="0.85", zorder=0, width=.8)
ax2.set_ylabel("tool time (s)", color="0.5")
ax2.grid(False)
for role, g in one.groupby("role"):
    ax.scatter(g["turn"], g["num_prefill_tokens"], label=role, s=30, zorder=3)
ax.set(xlabel="turn", ylabel="prompt tokens", title=f"One task: {len(one)} LLM requests")
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.legend(fontsize=8, ncol=3, loc="upper left")
fig.tight_layout()

# %% [markdown]
# The prompt grows inside a stretch of turns and then drops: each time the
# coordinator hands a subtask to a fresh worker, that worker starts a new
# conversation with its own system prompt. Within a worker's stretch, every turn
# adds a tool result to the prompt.
#
# The flat band of twelve large turns in the middle is not a conversation at all.
# It is a **fan-out**: the web worker retrieved a long page, split it into twelve
# pieces, and asked the model about each piece *in parallel*. They all depend on
# the same turn and a later turn joins them:

# %%
one[one["role"] == "web summarize"].drop(columns="token_ids").head(4)

# %% [markdown]
# All twelve are released at the same instant, each with about 9,000 prompt
# tokens: 100,000 tokens of prefill arriving at one replica at once, after a 22 s
# tool call during which that replica had nothing from this session to do. Agent
# traffic is **bursty by construction**.
#
# ## Tokens, time, and tools
#
# Now the whole sample. Three numbers decide what this workload does to a serving
# system:

# %%
tokens = pd.Series({
    "requests": len(sessions),
    "requests per session (median)": sessions.groupby("session").size().median(),
    "prompt tokens (total)": sessions.num_prefill_tokens.sum(),
    "output tokens (total)": sessions.num_decode_tokens.sum(),
    "prompt : output": sessions.num_prefill_tokens.sum() / sessions.num_decode_tokens.sum(),
    "median prompt tokens": sessions.num_prefill_tokens.median(),
    "median output tokens": sessions.num_decode_tokens.median(),
    "turns waiting on a tool": (sessions.tool_time > 0).mean(),
    "tool time, median of those (s)": sessions.loc[sessions.tool_time > 0, "tool_time"].median(),
})
tokens.round(2)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for role, g in sessions.groupby("role"):
    ax[0].scatter(g["num_prefill_tokens"], g["num_decode_tokens"], s=10, alpha=.6, label=role)
ax[0].set(xlabel="prompt tokens", ylabel="output tokens", xscale="log", yscale="log",
          title="Every request, by role")
ax[0].legend(fontsize=7, ncol=2)
tool = np.sort(sessions.loc[sessions.tool_time > 0, "tool_time"])
ax[1].plot(tool, np.arange(1, len(tool) + 1) / len(tool))
ax[1].set(xlabel="tool time (s)", ylabel="fraction of tool calls", xscale="log",
          title="How long tools take")
fig.tight_layout()

# %% [markdown]
# Prompts outnumber outputs by about 11:1 in tokens, and many turns generate only
# a few dozen tokens — a tool call, a worker id, a yes/no. Compared with the chat
# workloads of lectures 1–4 (256 in, 256 out), this is a different animal, and it
# looks like the ideal case for everything lecture 4 said about prefill-heavy
# traffic.
#
# But tokens are not time. A prompt token costs roughly 0.25 ms of GPU time here,
# while an output token costs about 45 ms — nearly 200× more — because decode is
# memory-bound (lecture 1). A turn with 9,000 prompt tokens and 180 output tokens
# spends a couple of seconds on its prompt and eight on its answer.
# **Agent traffic is prompt-heavy in tokens and decode-heavy in time.** Keep both
# halves of that sentence; each explains a different result below.
#
# ## How much of this is reusable?
#
# Before simulating anything, we can compute what a perfect, infinitely large
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

print(f"ideal prefix-cache hit rate: {ideal_hit_rate(sessions):.1%}")

# %% [markdown]
# About 60% of all prompt tokens in this workload have been computed before. The
# number is not higher because of the structure we just saw: every time the
# coordinator starts a new worker, that worker's prompt begins with a different
# system prompt, and the twelve fan-out requests share only their instructions, not
# the page chunk each one carries.
#
# ## Serving the trace
#
# Now give the trace to the simulator. `lsg.gaia_trace` writes it in the
# simulator's format, including the token ids and the dependency graph, so turn
# *k+1* is released only after turn *k* finishes and its tool call returns. We send
# one new task every ten seconds to a single replica.
#
# Because a user waits for the whole task, we measure **task completion time**: from
# the moment a session's first request arrives to the moment its last one finishes,
# tool time included.

# %%
trace = lsg.gaia_trace(sessions)

def task_times(result):
    """Wall-clock time of each session, from its first arrival to its last completion."""
    q = result.requests.assign(session=lambda d: d["Request Id"] // 1000,
                               done=lambda d: d["request_arrived_at"] + d["request_e2e_time"])
    return q.groupby("session").apply(
        lambda d: d["done"].max() - d["request_arrived_at"].min(), include_groups=False)

agent = {}
for pc in (False, True):
    r = lsg.simulate(**SYSTEM, trace=trace, num_requests=len(sessions), qps=0.1,
                     prefix_caching=pc, max_tokens=65536)
    t = task_times(r)
    agent[f"prefix caching {'on' if pc else 'off'}"] = {
        "hit rate": r.cache_hit_rate,
        "task time p50 (s)": t.median(), "task time p90 (s)": t.quantile(0.9),
        **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT p50 (ms)", "throughput (tok/s)"]]}
pd.DataFrame(agent).round(2)

# %% [markdown]
# The measured hit rate, 60%, lands within a percent of the hand-computed ideal.
# Blocks *are* evicted during the run (`r.cache` reports tens of thousands), but
# the replica is large enough that the evicted ones are mostly blocks nobody comes
# back to. Prefill work drops by 60%, TTFT by 6×, and the median task finishes
# about a minute and a half sooner.
#
# Note how much smaller the task-level gain is than the request-level one. A 6×
# better TTFT buys a quarter off the task. The reason is the second half of the sentence
# above: the task's time is mostly spent generating tokens and waiting for tools,
# and prefix caching touches neither.
#
# ## Where a task's time goes
#
# Let's take the session apart. The bar for each turn runs from its arrival to its
# first token (queueing and prefill) and then to its last token (decode); the gaps
# between turns are tool calls and the agent's own bookkeeping.

# %%
r = lsg.simulate(**SYSTEM, trace=trace, num_requests=len(sessions), qps=0.1,
                 prefix_caching=True, max_tokens=65536)
q = r.requests.assign(session=lambda d: d["Request Id"] // 1000,
                      turn=lambda d: d["Request Id"] % 1000)
s4 = q[q.session == 4].sort_values("turn")
t0 = s4.request_arrived_at.min()

fig, ax = plt.subplots(figsize=(9, 4.2))
for _, x in s4.iterrows():
    a = x.request_arrived_at - t0
    ax.barh(x.turn, x.prefill_e2e_time, left=a, color="C3", height=.7)
    ax.barh(x.turn, x.request_e2e_time - x.prefill_e2e_time,
            left=a + x.prefill_e2e_time, color="C0", height=.7)
ax.barh(0, 0, color="C3", label="queueing + prefill")
ax.barh(0, 0, color="C0", label="decode")
ax.set(xlabel="seconds since the task arrived", ylabel="turn",
       title="One task on the GPU; the gaps are tool calls")
ax.legend(loc="upper right")
ax.invert_yaxis()
fig.tight_layout()

# %%
tool4 = sessions[sessions.session == 4]
breakdown = pd.Series({
    "task completion time (s)": (s4.request_arrived_at + s4.request_e2e_time).max() - t0,
    "summed over its turns: decode (s)": (s4.request_e2e_time - s4.prefill_e2e_time).sum(),
    "summed over its turns: prefill (s)": s4.prefill_time_execution_plus_preemption.sum(),
    "summed over its turns: queueing (s)": s4.request_scheduling_delay.sum(),
    # Siblings released by one fan-out all waited for the same tool call, so count it once.
    "waiting for tools (s)": tool4.groupby(tool4.dep.map(tuple)).tool_time.max().sum(),
})
breakdown.round(1)

# %% [markdown]
# Three things to read off the picture:
#
# 1. **The task is a chain.** Most of the time exactly one turn of this task is
#    running. Making the GPU twice as fast would shorten each bar, not remove the
#    serialization.
# 2. **Decode dominates the bars.** Summed over the task's turns, decode takes
#    about 6 minutes against 37 s of actual prefill, even though the prompts are
#    10–50× longer than the answers. Tool calls add about a minute.
# 3. **The twelve-way fan-out is the exception,** and it is visible: a dozen bars
#    start together, and the next turn waits for the last of them. There the
#    serving system *does* control the latency — those twelve prefills are
#    serialized by the replica, so the join waits for all of them.
#
# The sums in the table exceed the task's wall-clock time precisely because of that
# fan-out: twelve turns run at once, and most of the 2 minutes of queueing is those
# twelve siblings waiting for each other.
#
# ## A busier cluster
#
# One task at a time is not a serving problem. Let's load 100 recorded tasks and
# run them against four replicas (eight GPUs), one new task per second.

# %%
busy_sessions = lsg.gaia_sessions(num_sessions=100, seed=0)
busy_trace = lsg.gaia_trace(busy_sessions)
print(f"{len(busy_sessions)} requests, "
      f"{busy_sessions.num_prefill_tokens.sum() / 1e6:.1f}M prompt tokens")

# %% [markdown]
# ### Routing: cache affinity against load balance
#
# The cache is per replica (lecture 5), and an agent session is a long chain of
# requests that all want the same prefix. The router's choice therefore decides
# the hit rate.

# %%
routing = {}
for policy in ("round_robin", "lor", "sticky_lor", "dynamo_kv"):
    r = lsg.simulate(**SYSTEM, trace=busy_trace, num_requests=len(busy_sessions), qps=1.0,
                     num_replicas=4, prefix_caching=True, global_scheduler=policy,
                     max_tokens=65536)
    t = task_times(r)
    routing[policy] = {"hit rate": r.cache_hit_rate, "task time p50 (s)": t.median(),
                       "task time p90 (s)": t.quantile(0.9),
                       **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)"]]}
pd.DataFrame(routing).T.round(2)

# %% [markdown]
# `round_robin` ignores history and loses a third of the available hits. `lor`
# (least outstanding requests) recovers some of them without trying, because an
# idle replica is often the one that just finished this session's previous turn.
# `sticky_lor`, which pins a session to one replica for its whole life, gets
# closest to the single-replica hit rate and wins on median task time and median
# TTFT.
#
# But look at the tails: both cache-aware policies are two to three times worse at
# p99 than plain round-robin. Affinity means staying on a replica even when it has
# just been handed a twelve-way fan-out, and the siblings of that burst pay for it.
# Affinity and balance pull in opposite directions, and which one to favour depends
# on whether you are judged on the median or the tail. Lecture 8 looks at routing
# on its own.
#
# ### Scheduling: whose turn goes first
#
# Inside a replica, the waiting queue is FCFS by default. Two alternatives in the
# simulator are aimed squarely at agents:
#
# - **session FCFS** (`session_priority`): order waiting requests by the age of
#   their *session*, so that a task that started long ago is not overtaken by a
#   brand-new one.
# - **shortest job first** (`sjf_priority`): order by how many prompt tokens are
#   actually left to compute, after subtracting the prefix-cache hit. In this
#   workload many turns have almost nothing to prefill, and they are being held up
#   behind 9,000-token newcomers.

# %%
policies = {"FCFS": {},
            "session FCFS": {"--vllm_v1_scheduler_config_session_priority": True},
            "shortest job first": {"--vllm_v1_scheduler_config_sjf_priority": True}}
sched = {}
for name, extra in policies.items():
    r = lsg.simulate(**SYSTEM, trace=busy_trace, num_requests=len(busy_sessions), qps=1.0,
                     prefix_caching=True, max_tokens=65536, extra=extra)
    t = task_times(r)
    sched[name] = {"task time p50 (s)": t.median(), "task time p90 (s)": t.quantile(0.9),
                   **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)", "makespan (s)"]]}
pd.DataFrame(sched).T.round(2)

# %% [markdown]
# On a single replica carrying 100 tasks, shortest-job-first shortens the median
# task by about 13% and the median TTFT by 4×: letting the cheap turns through
# first keeps many sessions moving instead of parking them behind one long prefill.
# Ordering by session age does the opposite at the tail — it protects old sessions
# by making new ones wait.
#
# Note what did *not* change: the makespan. No queue order creates GPU capacity.
# Scheduling moves latency between tasks; it does not reduce the total work. And
# back on the lightly loaded replica of the previous section, the same policies
# change the task time by a fraction of a percent, because there is almost nothing
# in the queue to reorder — the critical path is the session's own chain of
# decodes and tool calls.
#
# ## What this means for a serving system
#
# | Observation | Consequence |
# |---|---|
# | Turns are dependent; a task is a chain | Task latency ≈ sum over turns of (queue + prefill + decode + tool). Throughput optimisations that lengthen any turn hurt the user directly. |
# | Prompts are ~11× the output in tokens | Prefill-heavy: large chunk sizes, prefill-side capacity, and prefix caching all matter (lectures 3–5). |
# | Output tokens cost ~200× more each | But most *time* is decode. Expect the decode pool, not the prefill pool, to size the cluster (lecture 4). |
# | ~60% of prompt tokens are a repeat | Prefix caching is not an optimisation here, it is a requirement; and the router has to preserve it. |
# | Fan-out bursts of a dozen requests | Admission and scheduling decide when the join unblocks; p99 TTFT of a *batch of siblings* is what matters, not p99 over all requests. |
# | Idle gaps while tools run | A session holds KV cache it is not using. Evicting it costs a re-prefill; keeping it costs memory. |
#
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
#
# 1. **Does the cache survive the tool call?** Rerun the caching comparison with a
#    small KV cache (`kv_blocks=4000`, lecture 5) and look at `r.cache`. How many
#    blocks are evicted, and what happens to the hit rate? Which sessions lose?
# 2. **Chunk size for agents.** The runs above used the default `chunk_size=512`.
#    Sweep 512, 2048, 4096 on the single-replica run and report TTFT and task time.
#    Why does a chunk size that was bad for chat (lecture 3) look better here?
# 3. **Disaggregate.** Build a PD cluster with lecture 4's `pd_cluster` helper
#    (four replicas, splits 1:3, 2:2, 3:1) and run the busy trace. Which split wins,
#    and does prefix caching change the answer? (Hint: what is the prefill pool's
#    hit rate if a session's turns go to different prefill replicas?)
# 4. **The cost of a fan-out.** In the single-session timeline, measure how long
#    the join turn (`dep` with twelve entries) waits after the first of its
#    dependencies finishes. How much of that is queueing behind its own siblings?
# 5. **A different sample.** Rerun the characterisation with `seed=1` and
#    `num_sessions=48`. How stable are the prompt:output ratio and the ideal hit
#    rate? What does that say about tuning a system to a trace?
# :::
#
# ## Check your understanding
#
# Click an answer to check it. Wrong answers can be retried.

# %% cellView="form" tags=["remove-input"]
#@title Questions (run this cell)
lsg.quiz([
    {"q": "Why does an agent's prompt grow from turn to turn?",
     "options": ["The model remembers more each time",
                 "The model has no memory between requests, so every turn re-sends the conversation plus the new tool output",
                 "The system pads prompts to a block boundary"],
     "answer": 1,
     "explain": "Each request is independent. Continuity is created by resending everything, which is exactly why the prefix cache works so well here."},
    {"q": "Agent traffic has about 11 prompt tokens per output token. Where does most of the GPU <i>time</i> go?",
     "options": ["Prefill, because there are far more prompt tokens",
                 "Decode, because each output token costs roughly 200× more than a prompt token"],
     "answer": 1,
     "explain": "Prefill is compute-bound and processes thousands of tokens per step; decode is memory-bound and produces one token per step per request."},
    {"q": "Prefix caching cut TTFT by 6× but task completion time by only about a quarter. Why?",
     "options": ["The cache hit rate was too low",
                 "Most of a task's time is decode and tool calls, which prefix caching does not touch",
                 "Task time is dominated by queueing"],
     "answer": 1,
     "explain": "The task's critical path is a chain of turns, each spending most of its time generating tokens or waiting for a tool."},
    {"q": "Twelve requests are released at once by a fan-out, and a later turn depends on all twelve. Which metric predicts the task's progress?",
     "options": ["The median TTFT of the twelve", "The maximum completion time among the twelve",
                 "The throughput of the replica"],
     "answer": 1,
     "explain": "A join waits for its slowest dependency, which is why tail latency within a sibling group matters more than averages."},
    {"q": "Why does sticky routing give the best median but the worst tail in the four-replica run?",
     "options": ["Sticky routing disables prefix caching for new sessions",
                 "It keeps a session on the replica that holds its prefix even when that replica is temporarily overloaded",
                 "It sends every session to replica 0"],
     "answer": 1,
     "explain": "Affinity maximises hits but gives up the freedom to move work away from a busy replica, e.g. one that has just received a fan-out burst."},
    {"q": "Shortest-job-first improved the median task time but left the makespan unchanged. What does that tell you?",
     "options": ["The scheduler is broken",
                 "Queue order redistributes waiting time between tasks; it does not create GPU capacity",
                 "The workload was not actually loaded"],
     "answer": 1,
     "explain": "Total work is fixed. Scheduling decides who waits; only more hardware, less work (caching), or cheaper work changes the makespan."},
    {"q": "A session waits 22 s for a tool. What is the dilemma for the replica holding its KV cache?",
     "options": ["Whether to keep the blocks (idle memory) or evict them (a full re-prefill when the turn returns)",
                 "Whether to decode ahead speculatively",
                 "Whether to switch the session to another model"],
     "answer": 0,
     "explain": "Tool gaps make the reuse distance long. This is the agentic version of lecture 5's eviction trade-off, and it is why cache capacity and offload matter for agents."},
])
