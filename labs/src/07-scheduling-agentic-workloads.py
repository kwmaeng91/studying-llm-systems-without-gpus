# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [TEMP] 7. Scheduling Agentic Workloads
#
# By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖
#
# {doc}`06-agentic-workloads` followed a single agent task through one replica. A
# serving system does not get to do that: it carries many tasks at once, and it has
# to decide, for every turn of every task, *which replica* should take it and
# *which waiting turn* a replica should serve next. This lecture is about those two
# decisions on the same recorded GAIA traces.
#
# :::{admonition} Status: draft
# :class: warning
# This lecture is still being written. The routing and scheduling experiments below
# are real, but the lecture is missing the comparison it needs most: chunked prefill
# ({doc}`03-chunked-prefill`) against prefill–decode disaggregation
# ({doc}`04-pd-disaggregation`) on agentic traffic, where prompts are long, mostly
# cached, and arrive in bursts. That is coming later.
# :::

# %% [markdown]
# ## Setup
#
# Same system as lectures 3–6: Qwen2.5-32B-Instruct, with each replica running on
# two A100s.

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

import pandas as pd

SYSTEM = dict(model="Qwen/Qwen2.5-32B-Instruct", device="a100", tensor_parallel=2)

def task_times(result):
    """Wall-clock time of each session, from its first arrival to its last completion."""
    q = result.requests.assign(session=lambda d: d["Request Id"] // 1000,
                               done=lambda d: d["request_arrived_at"] + d["request_e2e_time"])
    return q.groupby("session").apply(
        lambda d: d["done"].max() - d["request_arrived_at"].min(), include_groups=False)

# %% [markdown]
# ## A busier cluster
#
# One task at a time is not a serving problem. Let's load 100 recorded tasks and
# run them against four replicas (eight GPUs), one new task per second. Note what
# "load" means here: the arrival rate matters much less than it did in lecture 2,
# because a session only ever has a turn or two in flight. Adding load means adding
# concurrent *tasks*, not sending requests faster.

# %%
busy_sessions = lsg.gaia_sessions(num_sessions=100, seed=0)
busy_trace = lsg.gaia_trace(busy_sessions)
print(f"{len(busy_sessions)} requests, "
      f"{busy_sessions.num_prefill_tokens.sum() / 1e6:.1f}M prompt tokens")

# %% [markdown]
# ## Routing: cache affinity against load balance
#
# Each replica keeps its own KV cache, so a request only hits what the replica it
# lands on happens to hold. An agent session is a long chain of requests that all
# want the same prefix, so the router decides the hit rate. This is why production
# agent stacks care about routing at all: NVIDIA's
# [Dynamo](https://github.com/ai-dynamo/dynamo) scores replicas by the prompt
# tokens they would still have to compute as well as by load, and Mooncake
# ([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079)) routes around a
# cluster-wide KV store instead. We compare three of the simulator's policies, plus
# plain least-outstanding-requests as a cache-blind baseline.

# %%
routing = {}
for policy in ("round_robin", "lor", "sticky_lor", "dynamo_kv"):
    r = lsg.simulate(**SYSTEM, trace=busy_trace, num_requests=len(busy_sessions), qps=1.0,
                     num_replicas=4, prefix_caching=True, global_scheduler=policy,
                     max_tokens=65536)
    t = task_times(r)
    routing[policy] = {"KV cache hit rate": r.cache_hit_rate, "task time p50 (s)": t.median(),
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
# The tails go the other way: both cache-aware policies are two to three times
# worse at p99 than plain round-robin. Affinity means staying on a replica even
# when it has just been handed a twelve-way fan-out, and the siblings of that burst
# pay for it. Which side of that trade to take depends on whether the deployment is
# judged on the median or the tail.
#
# ## Scheduling: whose turn goes first
#
# Inside a replica, the waiting queue is FCFS by default. The simulator offers two
# alternatives aimed at agent workloads: `session_priority` orders waiting requests
# by the age of their *session*, so a task that started long ago is not overtaken
# by a brand-new one; and `sjf_priority` orders by the prompt tokens actually left
# to compute, after subtracting the prefix-cache hit. The second is interesting
# here because many turns have almost nothing to prefill — a cached 9,000-token
# prompt plus 100 new tokens — and under FCFS they wait behind newcomers that have
# all 9,000 still to do.

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
                   **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)",
                                  "total execution time (s)"]]}
pd.DataFrame(sched).T.round(2)

# %% [markdown]
# On a single replica carrying 100 tasks, shortest-job-first shortens the median
# task by about 13% and the median TTFT by 4×: letting the cheap turns through
# first keeps many sessions moving instead of parking them behind one long prefill.
# Ordering by session age does the opposite at the tail, protecting old sessions by
# making new ones wait.
#
# What none of them changes is the total execution time, because queue order
# redistributes waiting rather than creating GPU capacity. And on the lightly
# loaded replica of {doc}`06-agentic-workloads` the same policies change the task
# time by a fraction of a percent: there is almost nothing in the queue to reorder,
# and the critical path is the session's own chain of decodes and tool calls.
#
# ## Summary
#
# | | What we saw |
# |---|---|
# | Routing decides the hit rate | a session's turns want the same prefix, and only the replica that holds it can give them one |
# | Affinity against balance | sticky routing wins the median, round-robin wins the tail; a KV-aware router sits between them |
# | Queue order moves latency | shortest-job-first cuts the median task time and TTFT by letting cheap, mostly-cached turns through |
# | Queue order creates nothing | the total execution time is the same under every policy |
# | Load means tasks | at a lightly loaded replica none of this matters, because there is no queue to reorder |
#
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
#
# 1. **Disaggregate.** Build a PD cluster with lecture 4's `pd_cluster` helper
#    (four replicas, splits 1:3, 2:2, 3:1) and run the busy trace. Which split wins,
#    and does prefix caching change the answer? (Hint: what is the prefill pool's
#    hit rate if a session's turns go to different prefill replicas?)
# 2. **Routing under a tighter replica.** Rerun the routing comparison with
#    `batch_size_cap=32`, so each replica can only run 32 requests at once. Which
#    policy gains the most, and which loses?
# 3. **Starvation.** `sjf_priority` can leave a long prefill waiting forever. Add
#    `--vllm_v1_scheduler_config_sjf_starvation_timeout` (in seconds) and find the
#    value that keeps most of the median gain without the p99 of the whole run
#    getting worse.
# :::
#
# ## Check your understanding
#
# Click an answer to check it. Wrong answers can be retried.

# %% cellView="form" tags=["remove-input"]
#@title Questions (run this cell)
lsg.quiz([
    {"q": "Why does sticky routing give the best median task time but a much worse tail in the four-replica run?",
     "options": ["Sticky routing disables prefix caching for new sessions",
                 "It keeps a session on the replica that holds its prefix even when that replica is temporarily overloaded",
                 "It sends every session to replica 0"],
     "answer": 1,
     "explain": "Affinity maximises hits but gives up the freedom to move work away from a busy replica, e.g. one that has just received a fan-out burst."},
    {"q": "Shortest-job-first improved the median task time but left the total execution time of the whole run unchanged. What does that tell you?",
     "options": ["The scheduler is broken",
                 "Queue order redistributes waiting time between tasks; it does not create GPU capacity",
                 "The workload was not actually loaded"],
     "answer": 1,
     "explain": "Total work is fixed. Scheduling decides who waits; only more hardware, less work (caching), or cheaper work shortens the run itself."},
    {"q": "Why does shortest-job-first help this workload in particular?",
     "options": ["Agent turns are short to generate",
                 "Most turns have almost nothing left to prefill once the prefix cache is counted, so they are cheap to let through",
                 "It raises the KV cache hit rate"],
     "answer": 1,
     "explain": "The ordering uses prompt tokens remaining after the cache hit. A returning turn with a 9,000-token cached prompt and 100 new tokens is nearly free, and under FCFS it would wait behind a newcomer with all 9,000 to do."},
])
