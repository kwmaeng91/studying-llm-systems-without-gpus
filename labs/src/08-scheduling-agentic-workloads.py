# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [TEMP] 8. Scheduling Agentic Workloads
#
# By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖
#
# {doc}`07-agentic-workloads-2` followed a single agent task through one replica. A
# serving system does not get to do that: it carries many tasks at once, and every
# knob we met in lectures 2–5 has to be set for traffic that looks nothing like
# chat. This lecture puts the same recorded GAIA traces on a cluster and turns
# those knobs one at a time — how fast tasks arrive, how the GPUs are carved into
# replicas, chunked prefill against prefill–decode disaggregation, where the router
# sends each turn, and which waiting turn a replica serves next.
#
# :::{admonition} Status: draft
# :class: warning
# This lecture is still being written. The text and experiments are not in the right shape yet.
# :::
#
# :::{note}
# The runs here are larger than in earlier labs: 100 recorded tasks against up to
# eight GPUs, with a wider runtime-predictor grid (`max_tokens=65536`, shared with
# lecture 7). Expect the whole notebook to take several minutes, and the first run
# of a new (model, GPU, parallelism) combination to spend about a minute fitting
# its predictor.
# :::
#
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

def run(kv=False, **kwargs):
    """One simulation of the busy trace, reported per task as well as per request."""
    r = lsg.simulate(**{**SYSTEM, **kwargs}, trace=busy_trace, num_requests=len(busy_sessions),
                     prefix_caching=True, max_tokens=65536)
    t = task_times(r)
    out = {"KV cache hit rate": r.cache_hit_rate,
           "task time p50 (s)": t.median(), "task time p90 (s)": t.quantile(0.9),
           **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)", "total execution time (s)"]]}
    return {"KV tokens per replica": r.kv_cache_tokens, **out} if kv else out

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
# ## How much does the arrival rate matter?
#
# In lecture 2, the arrival rate was *the* knob: past the knee, the batch could not
# grow, requests queued, and TTFT climbed. Agent traffic does not behave that way,
# because a task is a chain. However many tasks are in flight, each of them only
# has a turn or two running at any moment, and the rest of its time is spent
# waiting for a tool or for its own previous turn. The cluster is throttled by the
# workload's dependencies rather than by the rate at which we hand it work.
#
# Let's see how much it matters, from one new task every four seconds to four new
# tasks a second — a 16× range.

# %%
arrival = {q: run(qps=q, num_replicas=4, global_scheduler="sticky_lor")
           for q in (0.25, 0.5, 1.0, 2.0, 4.0)}
pd.DataFrame(arrival).T.rename_axis("new tasks per second").round(2)

# %% [markdown]
# Sixteen times the arrival rate costs 12% of the median task time, and the time to
# get through all 100 tasks barely moves. Compare that with lecture 2, where
# crossing the knee was enough to flatten throughput and send TTFT climbing.
#
# This is worth remembering when sizing an agent deployment: the useful unit of
# load is *concurrent tasks*, not requests per second, and the way to overload this
# cluster is to give it more tasks at once, not to hand it the same tasks faster.
#
# ## Carving the same GPUs differently
#
# Eight A100s can be one replica of eight GPUs, two of four, or four of two. The
# model is the same, the hardware is the same, and the three options differ in what
# a replica can do:
#
# - **More replicas** run more turns in parallel and keep each all-reduce small,
#   but each replica has its own KV cache, so the same prefixes end up stored
#   several times and a session only hits on the replica it was sent to.
# - **Fewer, wider replicas** put all the memory behind one cache — splitting the
#   weights across eight GPUs leaves far more room for KV than splitting them
#   across two — but every turn queues behind every other turn.

# %%
carving = {f"{n} x TP{tp}": run(kv=True, tensor_parallel=tp, num_replicas=n, qps=1.0,
                                global_scheduler="sticky_lor")
           for n, tp in ((4, 2), (2, 4), (1, 8))}
pd.DataFrame(carving).T.round(2)

# %% [markdown]
# `2 x TP4` wins here, and the two ends lose for opposite reasons. Four narrow
# replicas spread the cache thinnest (the lowest hit rate of the three) and give
# the worst tail TTFT, because a session is stuck with whichever replica it was
# pinned to. One wide replica has the best hit rate — every prefix is in the one
# cache — and six times the KV capacity of a TP2 replica, because the weights are
# spread over eight GPUs instead of two and whatever is left over becomes cache.
# Its median TTFT is still twice as bad as the other two, because every one of the
# hundred tasks queues in the same place.
#
# The middle option keeps enough parallelism to absorb a fan-out while keeping the
# cache concentrated. That balance is specific to this workload; a trace with
# shorter prompts and less sharing would push the answer towards more replicas.
#
# ## Chunked prefill or disaggregation?
#
# Lectures 3 and 4 gave two ways to stop long prefills from disturbing decodes:
# chop the prefill into chunks and interleave it, or move prefill and decode onto
# separate GPUs. Agent traffic is the hard case for both — prompts are long, most
# of them are already cached, and a fan-out can drop a dozen of them on a replica
# at once.
#
# We compare the chunked cluster we have been using with two disaggregated ones
# built from the same eight GPUs, using lecture 4's helper. One difference is
# forced on us: a disaggregated cluster needs the PD-aware router
# (`load_aware`), so these rows do not get the session affinity that `sticky_lor`
# gives the chunked row.

# %%
def pd_cluster(n_prefill, n_decode, prefill_chunk_size=4096):
    """n_prefill prefill replicas and n_decode decode replicas, each on two A100s."""
    def group(role, n, chunk_size):
        return {"role": role, "num_replicas": n,
                "replica_config": {"model_name": SYSTEM["model"], "device": SYSTEM["device"],
                                   "tensor_parallel_size": SYSTEM["tensor_parallel"],
                                   "network_device": "a100_dgx", "pd_disaggregation": 1},
                "replica_scheduler_config": {"type": "vllm_v1", "batch_size_cap": 128,
                                             "chunk_size": chunk_size}}
    return {"replica_groups": [group("prefill", n_prefill, prefill_chunk_size),
                               group("decode", n_decode, 512)],
            "replica_groups_pools": [{"prefill": list(range(n_prefill)),
                                      "decode": list(range(n_prefill, n_prefill + n_decode)),
                                      "cross_node": False}]}

split = {"chunked, 4 replicas": run(qps=1.0, num_replicas=4, global_scheduler="sticky_lor")}
for n_prefill, n_decode in ((2, 2), (3, 1)):
    split[f"PD {n_prefill}:{n_decode}"] = run(qps=1.0, global_scheduler="load_aware",
                                              replica_groups=pd_cluster(n_prefill, n_decode))
pd.DataFrame(split).T.round(2)

# %% [markdown]
# Disaggregation does well here, and PD 3:1 — three quarters of the cluster
# prefilling — does best: a slightly better median task time than the chunked
# cluster, with a tail TTFT about two and a half times lower. That split looks lopsided until you
# remember what the trace is made of. Prompts are enormous, answers are short, and
# the bursts that hurt are bursts of *prefill*; a large prefill pool absorbs a
# twelve-way fan-out without any of it landing on a GPU that is streaming someone
# else's answer.
#
# What disaggregation gives up is some of the cache. Both PD rows hit a few points
# lower than the chunked row, because a session's turns are spread over the prefill
# pool instead of returning to one replica. Lecture 5's rule still applies, and it
# is the reason a production disaggregated stack also routes by prefix.

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
routing = {policy: run(qps=1.0, num_replicas=4, global_scheduler=policy)
           for policy in ("round_robin", "lor", "sticky_lor", "dynamo_kv")}
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
sched = {name: run(qps=1.0, extra=extra) for name, extra in policies.items()}
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
# loaded replica of {doc}`07-agentic-workloads-2` the same policies change the task
# time by a fraction of a percent: there is almost nothing in the queue to reorder,
# and the critical path is the session's own chain of decodes and tool calls.
#
# ## Summary
#
# | Knob | What we saw |
# |---|---|
# | Arrival rate | 16× the arrival rate costs 12% of the median task: a chain of dependent turns throttles itself, so load means concurrent *tasks*, not requests per second |
# | Carving the GPUs | the middle option won; more replicas split the cache and the tail suffers, one wide replica holds every prefix but queues everything in one place |
# | Chunked vs disaggregated | a prefill-heavy split (PD 3:1) matched the chunked cluster's median task time with a far better tail, at the cost of a few points of hit rate |
# | Routing | sticky affinity wins the median and the hit rate, round-robin wins the tail, a KV-aware router sits between them |
# | Queue order | shortest-job-first cuts the median task time and TTFT by letting cheap, mostly-cached turns through, and changes the total execution time not at all |
#
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
#
# 1. **The other split.** Add `PD 1:3` to the disaggregation table. Lecture 4 found
#    that an undersized prefill pool makes TTFT explode; does it here, and does the
#    prefix cache hide any of it?
# 2. **More tasks, not faster tasks.** The arrival sweep barely moved anything.
#    Load the trace with `num_sessions=48` and with `num_sessions=100` at the same
#    `qps` and compare. Which of the two kinds of "more load" actually hurts?
# 3. **Does the carving answer survive?** Repeat the three carvings with prefix
#    caching off. Does `2 x TP4` still win, and what does that tell you about why
#    it won?
# 4. **Routing under a tighter replica.** Rerun the routing comparison with
#    `batch_size_cap=32`, so each replica can only run 32 requests at once. Which
#    policy gains the most, and which loses?
# 5. **Starvation.** `sjf_priority` can leave a long prefill waiting forever. Add
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
    {"q": "Raising the arrival rate 16× cost only 12% of the median task time. Why does agent traffic behave so differently from lecture 2's chat workload?",
     "options": ["The replicas were oversized",
                 "A task is a chain of dependent turns, so however many tasks are in flight, each has only a turn or two running at a time",
                 "The prefix cache absorbed the extra load"],
     "answer": 1,
     "explain": "The workload throttles itself. To load this cluster you add concurrent tasks, not requests per second."},
    {"q": "Eight GPUs as one TP8 replica had the best KV cache hit rate but the worst median TTFT. What explains both?",
     "options": ["TP8 is slower per token",
                 "One replica means one cache that holds every prefix, and one queue that every task has to pass through",
                 "Prefix caching does not work above TP4"],
     "answer": 1,
     "explain": "Fewer, wider replicas concentrate the cache and the memory behind it, and concentrate the queueing too."},
    {"q": "Why does a prefill-heavy disaggregated split (PD 3:1) suit this trace?",
     "options": ["Agent answers are long, so decode needs little hardware",
                 "Prompts are long and answers short, and the bursts that hurt are bursts of prefill, which a large prefill pool can absorb away from the decoding turns",
                 "Disaggregation raises the cache hit rate"],
     "answer": 1,
     "explain": "A twelve-way fan-out is 100,000 tokens of prefill at once. Keeping it off the GPUs that are streaming answers is exactly what lecture 4's separation buys."},
    {"q": "Why does shortest-job-first help this workload in particular?",
     "options": ["Agent turns are short to generate",
                 "Most turns have almost nothing left to prefill once the prefix cache is counted, so they are cheap to let through",
                 "It raises the KV cache hit rate"],
     "answer": 1,
     "explain": "The ordering uses prompt tokens remaining after the cache hit. A returning turn with a 9,000-token cached prompt and 100 new tokens is nearly free, and under FCFS it would wait behind a newcomer with all 9,000 to do."},
])
