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
# back, the request is gone. An agent works differently. Given one task — "find
# which main course disappeared from this restaurant's menu between two dates" — it
# issues dozens of LLM requests: it plans, picks a worker, calls a search tool,
# reads the result, calls another tool, writes code, runs it, reads the error,
# tries again, and finally answers. The user waits for the whole chain.
#
# Three things differ from what we saw before. First, the requests are
# *dependent*: turn *k+1* may not be able to start until turn *k* has finished and its tool
# call has returned. Second, the prompts are long and repetitive, because each turn
# resends the whole conversation plus the new tool output, which is lecture 5's
# problem in its most extreme form. Third, the thing a user waits for is the task,
# not the request: a p99 TTFT of 200 ms says little if the task takes six minutes.
#
# The lecture works from recorded traces rather than a synthetic workload. They come
# from [GAIATrace](https://github.com/psu-paws/Vidur-Agent)
# ([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725)), which recorded every
# LLM request two agent systems issued while solving tasks from GAIA
# ([Mialon et al., ICLR '24](https://openreview.net/forum?id=fibxvahvs3)), a
# benchmark of questions that need browsing, file handling and code to answer. The
# corpus covers [OWL](https://github.com/camel-ai/owl)
# ([Hu et al., NeurIPS '25](https://arxiv.org/abs/2505.23885)), a multi-agent system,
# and [MiroThinker](https://github.com/MiroMindAI/MiroThinker), a single agent with a
# summariser; we use OWL here. Each request carries its prompt and output token ids,
# the turns it depended on, and a separately measured tool latency. The traces ship
# with the simulator, so they are already on your disk after `lsg.setup()`.
# We are also in the process of extending GAIATrace to more setups and datasets!
#
# :::{admonition} Learning goals
# - Describe the structure of an agent session: turns, roles, dependencies,
#   fan-out, and tool time.
# - Explain why agent traffic is prompt-heavy in tokens but decode-heavy in time.
# - Measure KV cache hit rates on a real agent trace and explain where the hits
#   come from.
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
# of them out of the 165 tasks OWL ran. Sessions containing a request longer than
# the course's runtime predictor was fitted for are skipped, which removes the
# largest tasks (again, this course uses a downsized simulator for simplicity).

# %%
sessions = lsg.gaia_sessions(num_sessions=24, seed=0)
print(f"{sessions.session.nunique()} sessions, {len(sessions)} LLM requests")
sessions.drop(columns="token_ids").head(8)

# %% [markdown]
# The table gives a peek at how OWL solves a particular question.
# Each row is one request to the model, which does the planning, coordination, web search, and so on.
# Note that two models are used here: gpt-4o for tasks that do not require heavy thinking, and gpt-oss-120b for
# tasks that require thinking. This is just how the authors of GAIATrace (that's me and my students!) decided to do it.
#
# - `session` / `turn`: which task, and the position in it.
# - `role`: which agent inside the system issued it. OWL is a *multi-agent* system:
#   a planner splits the task, a coordinator assigns each subtask to a worker, and
#   the workers (web search, code, browser, document reader) do it.
# - `model`: which model served the request in the recording. OWL uses two, which
#   GAIATrace calls the *main* and *sub* models, and gives each a different set of
#   roles.
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
# That is the planner: it receives the user's task ("I went to Virtue restaurant & bar ...") and a description of the
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
# between requests, so each turn re-sends the whole conversation, and the prompt
# grows along a branch until that worker is done. Almost every prompt is some
# earlier prompt with text appended to it, which is the exact condition lecture 5
# said a prefix cache needs.
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
# Each dot represents a request from a sub-agent (plan, coordinate, web search, ...),
# and when the agent uses a tool, the bar shows the tool execution time.
# The prompt length (y-axis of each dot) grows inside a stretch of turns and then drops: each time the
# coordinator hands a subtask to a fresh worker, that worker starts a new
# conversation with its own system prompt. Within a worker's stretch, every turn
# adds a tool result to the prompt. Again, this is how OWL was designed and not necessarily universal.
#
# The flat band of twelve large turns in the middle is the web worker retrieving a long page, splitting it into twelve
# pieces, and asking the model to summarise each piece in parallel. This is again a specific design
# made by the OWL authors to keep the prompt (prefill) length small and is not fundamental.
# You don't have to understand every piece that is happening in this plot, and many things are specific to the
# particular design of this agent; having a sense
# that multi-agent systems are complex is probably enough.
#
# ## Tokens, time, and tools
#
# Now, let's look at prompt (input) and output tokens for each sub-agent:

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
# You can see that different sub-agents, depending on their role, show different behaviors:
# the coordinator has a distinct group of different prompt but similar output lengths;
# the web search agent has varying prompt length but similar output length; and the web summariser agent
# has high prompt length and relatively shorter output length.
# These make sense once you think of what each sub-agent does.
# In general, prompt (input) is longer than output by about 11:1 in tokens,
# so this is prefill-heavy traffic.
#
# Prompt and output tokens cost differently, though. A prompt token costs roughly 0.25 ms of GPU time on
# our simulated replica, while an output token costs about 45 ms — nearly 200× more —
# because decode is memory-bound (lecture 1). A turn with 9,000 prompt tokens and
# 180 output tokens spends a couple of seconds on its prompt and eight on its
# answer. So being prefill-heavy does not always mean prefill is going to be the bottleneck (it may or may not).
# Another thing to consider is the prefix cache (lecture 5) — prefill gets much cheaper with a high prefix cache hit rate.
#
# ## Revisiting prefix caching for agentic workloads
#
# Now, let's see how many prefix cache hits this trace gets.
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

print(f"ideal prefix-cache hit rate: {ideal_hit_rate(sessions):.1%}")

# %% [markdown]
# About 60% of all prompt tokens in this workload have been computed before and can benefit from prefix caching, and this number roughly matches the
# GAIATrace paper ([Kim et al., IISWC '26](https://arxiv.org/abs/2606.01725)).
# This number is high, because the system prompt is long, and sub-agents often look at their past conversation history, similar
# to the multi-turn chat behavior from lecture 5.
# However, this is much lower than what some other papers (like Agentic AI Workload Characteristics,
# [Yuan et al., IISWC '26](https://arxiv.org/abs/2605.26297)) reported,
# where the reported numbers were more like 87--99%.
# This is because of how OWL is designed: as a multi-agent system, OWL runs multiple sub-agents, and there is limited sharing of prompts between different agents.
#
# Still, the number is (slightly) higher than what Mooncake
# ([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079)) reported, which was about 59% for tool- and agent-style
# traffic reaching the Kimi chatbot, and about 40% on ordinary conversation.
# Again, how you designed the agentic system significantly affects the prefix cache hit rate.
# Since we still do not have a consensus on what the right design for an agentic system is, this number will probably fluctuate in the future until we converge to a decision.
#
# ## Serving the trace
#
# Now, let's try giving the trace to the simulator.
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
        "KV cache hit rate": r.cache_hit_rate,
        "task time p50 (s)": t.median(), "task time p90 (s)": t.quantile(0.9),
        **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT p50 (ms)", "throughput (tok/s)"]]}
pd.DataFrame(agent).round(2)

# %% [markdown]
# The measured KV cache hit rate, 60%, lands within a percent of the hand-computed
# ideal. Blocks are evicted during the run (`r.cache` reports tens of thousands),
# but this replica is large enough that the ones it loses are mostly blocks nobody
# comes back to. Prefill work drops by 60%, TTFT by 6×, and the median task
# finishes about a minute and a half sooner.
#
# The task-level gain is much smaller than the request-level one: a 6× better TTFT
# buys a quarter off the task. That is the second half of the earlier sentence at
# work — a task spends most of its time generating tokens and waiting for tools,
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
# ## Summary
#
# | | What we saw |
# |---|---|
# | Shape of a task | an agentic task is mostly a series of sequential requests but sometimes parallel |
# | Tokens | usually more input tokens than output tokens, but it depends on the agent's role |
# | Time | decode still dominates (especially when the batch size is low and prefix cache hit rate is high) |
# | Prefix caching | hit rate is around 60% and improves TTFT by 6×, but the median task time by only about a quarter |
#
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
#
# 1. **Does the cache survive the tool call?** Rerun the caching comparison with a
#    small KV cache (`kv_blocks=4000`, lecture 5) and look at `r.cache`. How many
#    blocks are evicted, what happens to the hit rate, and do the sessions with the
#    longest tool waits lose more than the others?
# 2. **Chunk size for agents.** The runs above used the default `chunk_size=512`.
#    Sweep 512, 2048, 4096 on the single-replica run and report TTFT and task time.
#    Why does a chunk size that was bad for chat (lecture 3) look better here, and
#    does prefix caching being on change the answer?
# 3. **The cost of a fan-out.** In the single-session timeline, measure how long
#    the join turn (`dep` with twelve entries) waits after the *first* of its
#    dependencies finishes. How much of that is queueing behind its own siblings,
#    and what would a scheduler have to know to shorten it?
# 4. **Size the two pools.** Using the `model` column, work out how much GPU time
#    each of the two models would need for this workload: count prompt and output
#    tokens per model and price them at the ~0.25 ms and ~45 ms per token measured
#    above. If you had eight GPUs, how would you split them, and which pool would
#    saturate first? Then check your reasoning against what `lsg.gaia_sessions`
#    shows about which roles sit on the critical path.
# 5. **A different sample.** Rerun the characterisation with `seed=1` and
#    `num_sessions=48`. How stable are the prompt:output ratio and the ideal hit
#    rate? What does that say about tuning a system to one trace?
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
     "options": ["The KV cache hit rate was too low",
                 "Most of a task's time is decode and tool calls, which prefix caching does not touch",
                 "Task time is dominated by queueing"],
     "answer": 1,
     "explain": "The task's critical path is a chain of turns, each spending most of its time generating tokens or waiting for a tool."},
    {"q": "A session waits 22 s for a tool. What is the dilemma for the replica holding its KV cache?",
     "options": ["Whether to keep the blocks (idle memory) or evict them (a full re-prefill when the turn returns)",
                 "Whether to decode ahead speculatively",
                 "Whether to switch the session to another model"],
     "answer": 0,
     "explain": "Tool gaps make the reuse distance long. This is the agentic version of lecture 5's eviction trade-off, and it is why cache capacity and offload matter for agents."},
    {"q": "The ideal KV cache hit rate on this trace is 60% rather than 95%. What limits it?",
     "options": ["The 16-token block size",
                 "Each new worker starts its own conversation with a different system prompt, and the fan-out requests share only their instructions",
                 "The traces were recorded with caching disabled"],
     "answer": 1,
     "explain": "Reuse follows the agent's structure. Within a worker's stretch of turns almost everything repeats; across workers and across the chunks of a fan-out, much less does."},
])
