# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 5. Prefix Caching
#
# By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖
#
# The last three lectures treated prefill as work that has to be done: it is
# compute-bound (lecture 1), it interferes with decoding (lecture 3), and we can
# either chunk it or move it to its own GPUs (lecture 4). This lecture is about
# how we can avoid some prefill.
#
# ## Why the same tokens arrive over and over
#
# An LLM is a pure function of the tokens you hand it. The server keeps nothing
# between requests, at least in its simplest form. So in a chatbot, where the model
# must be aware of the conversation so far, the past conversation is simply
# prepended to the new question: the model reads the whole history and the question
# together, every time.
#
# So a two-message chat is not two small requests. It is one small request and one
# large one, and what actually goes over the wire looks like this:
#
# ```text
# request 1   [system prompt, ~1,000 tokens]
#             [user: "How do I reverse a list in Python?"]
#
# response 1  [assistant: "Use list.reverse(), or ..."]
#
# request 2   [system prompt, ~1,000 tokens]            <- identical
#             [user: "How do I reverse a list in Python?"]   <- identical
#             [assistant: "Use list.reverse(), or ..."]      <- identical
#             [user: "What about a tuple?"]                  <- new
# ```
#
# The **system prompt** is the block of text the application puts in front of every
# conversation and never shows you: who the assistant is supposed to be, what it
# must refuse to do, today's date, the JSON schema of every tool it may call, and
# often a handful of examples of good answers. In a serious application it is not
# short — several hundred to a few thousand tokens, and for an agent carrying a
# dozen tool definitions, much more (lecture 6).
#
# The first request prefills the system prompt plus the user's question. The second
# prefills the system prompt, the first question, the first answer, *and* the second
# question. It has to carry the first exchange because the model cannot see it
# otherwise: "What about a tuple?" is meaningless without the question and the
# answer above it. The prompt keeps growing like this for the rest of the
# conversation — and almost all of it is text the model has already processed.
#
# ### The same text also repeats across users
#
# Repetition is not confined to one conversation. *Every* request an LLM-based
# application makes is prefixed with the same system prompt, so the opening
# thousand tokens of every request are token-for-token identical, even though the
# users who sent them have nothing to do with each other.
#
# It helps to see what that text actually is. The excerpts below are decoded from
# the recorded traces of [OWL](https://github.com/camel-ai/owl)
# ([Hu et al., NeurIPS '25](https://arxiv.org/abs/2505.23885)), the open-source
# multi-agent system we take apart in lecture 6. This is verbatim what it sent to
# the model, elided where marked. Every request its web-search worker makes opens
# with the following **system prompt**:
#
# ```text
# You are a helpful assistant that can search the web, extract webpage content,
# simulate browser actions, and provide relevant information to solve the given task.
# Keep in mind that:
# - Do not be overly confident in your own knowledge. Searching can provide a
#   broader perspective and help validate existing knowledge.
# - If the search snippet is unhelpful but the URL comes from an authoritative
#   source, try visit the website for more details.
# - When looking for specific numerical values (e.g., dollar amounts), prioritize
#   reliable sources and avoid relying only on search snippets.
# - When solving tasks that require web searches, check Wikipedia first before
#   exploring other websites.
# [...]
# ```
#
# Directly below it come the **tool definitions** — the name, description and
# argument schema of each of the eight tools this worker may call. Two of them:
#
# ```text
# namespace functions {
#
# // Use Google search engine to search information for the given query.
# type search_google = (_: {
# // The query to be searched.
# query: string,
# // The number of result pages to retrieve.
# num_result_pages: number,
# }) => any;
#
# // Search the entity in WikiPedia and return the summary of the required page,
# // containing factual information about the given entity.
# type search_wiki = (_: {
# // The entity to be searched.
# entity: string,
# }) => any;
#
# [... six more ...]
# }
# ```
#
# A different role, OWL's planner, instead carries a page of **worked examples** of
# the judgement it is being asked to make — the agent's version of the few-shot
# examples a classifier puts in front of every input:
#
# ```text
# Here are some scenarios where using code is the preferred approach:
# 1. Tasks requiring access to a large number of webpages. Example: "How many times
#    was a Twitter/X post cited as a reference on English Wikipedia pages for each
#    day of August in the last June 2023 versions of the pages?" Reason: Manually
#    checking each Wikipedia page would be highly inefficient, while Python code can
#    systematically fetch and process the required data.
# 2. Data processing involving complex filtering or calculations. Example: "Analyze
#    all article titles on Hacker News in March 2024 and find the top 10 most
#    frequently occurring keywords." Reason: This task requires processing a large
#    amount of text data, which is best handled programmatically.
# [...]
# ```
#
# Whenever the agent needs a web search, the request it sends to the web-search
# worker opens with about 1,500 tokens of the text shown above, before the actual
# task is even mentioned.
#
# Common prefixes also appear when many people ask about the same document — the
# same manual, contract or repository file pasted above a different question each
# time. [vLLM's documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
# gives exactly that as the canonical case for turning prefix caching on.
#
# Mooncake ([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079)), the serving system behind the Kimi
# chatbot, reports from its production traces that roughly half of all prompt tokens
# are reusable when the cache has room — about 40% on conversational traffic, where
# most of the reuse is a user's own history, and about 59% on tool- and agent-style
# traffic, where those long, repetitive system prompts are reused across users. The
# same deployment's hit rate falls below 20% at peak hours, when the cache does not
# have room; we will reproduce that effect later in the lecture.
#
# ## The idea
#
# **Prefix caching** keeps that KV cache — the per-token keys and values every
# later token attends to (lecture 2) — around after a request finishes, and reuses
# it. If a new prompt starts with a set of tokens whose KV is still around, those
# tokens are not prefilled again; the model reuses the KV from the past and computes
# KV only for the tokens it has never seen. It costs nothing in accuracy: the KV of a token depends only on
# the tokens before it, so a reused block holds exactly the values a fresh prefill
# would have produced.
#
# :::{admonition} Learning goals
# - Explain how a KV cache is addressed by block hashes, and why reuse works only
#   on a *prefix*.
# - Measure the effect of prefix caching on prefill work and TTFT, both within one
#   conversation and across users who share a long preamble.
# - Explain the two things that destroy a hit: eviction under memory pressure, and
#   a request landing on a replica that does not hold the prefix.
# - Lay out a prompt so that it is cache-friendly, and say what cross-user sharing
#   costs as well as what it buys.
# - Say where cached blocks actually live in a production deployment, and where
#   these labs simplify.
# :::

# %% [markdown]
# ## Setup
#
# Same system as lectures 3 and 4: Qwen2.5-32B-Instruct, one replica on two A100s.

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
# ## How a prefix cache works
#
# ### Step 1: the KV cache is already paged
#
# We have not yet looked at how the KV cache is laid out in memory. It is not one
# contiguous array per request: it is stored at the granularity of **blocks** (also
# called pages), which you can think of as pages in virtual memory. The idea of
# storing the KV cache in blocks was introduced by **PagedAttention**
# ([Kwon et al., SOSP '23](https://arxiv.org/abs/2309.06180)), the paper vLLM was built around.
#
# Block sizes can be configured: vLLM defaults to 16 tokens (`--block-size`),
# TensorRT-LLM to 32, and SGLang to 1 (`--page-size`). Larger blocks mean less
# bookkeeping per token and more efficient transfers when blocks move between
# memories; smaller blocks waste less space in the last, partial block and — as the
# next step shows — match prefixes more precisely. Our labs use 16.
#
# ### Step 2: give every block a name
#
# Prefix caching adds one idea on top: **give every full block a name, and keep a
# table from names to blocks.** The name is a hash of
#
# ```text
# hash(block) = H( hash(previous block), the 16 token ids in this block )
# ```
#
# so a block's name depends on its own tokens *and* on the entire history before
# it. Two requests get the same name for their 5th block only if their first 80 (16$\times$5)
# tokens are identical.
#
# When a request arrives, the scheduler hashes its prompt block by block and looks
# each name up, stopping at the first miss. Everything before the miss is a **hit**:
# those blocks are reused (their reference count goes up, so they cannot be
# evicted while in use) and the request is prefilled starting from the first
# uncached token.
#
# ```text
#   turn 1 prompt   [sys ][sys ][user1]                 3 blocks computed,
#                                      [ans1]           1 more filled while decoding
#   turn 2 prompt   [sys ][sys ][user1][ans1][user2]    4 hits, 1 block computed
# ```
#
# Note that the answer's blocks are reusable too: they were written during turn 1's
# decode and named like any other, which is why turn 2 reuses the previous answer
# as well as the previous prompt.
#
# Three properties follow, and all three matter in practice:
#
# 1. **Matching is on a prefix, not on a substring.** The chained hash means one
#    different token early in the prompt gives every later block a different name,
#    even if the rest of the text is identical. Only the run of tokens that is
#    identical from the very first one counts as a hit.
# 2. **The last, partial block is not named.** Only full blocks are hashed, so
#    hits are rounded down to a multiple of the block size. This is a small artefact
#    that we will mostly ignore; it only matters when the block size is large, which
#    it usually is not.
# 3. **Cached blocks compete for memory with running requests.** Cached prefixes and
#    the KV cache of running requests sit in the same pool, so when a request needs
#    more space, the server has to evict something — and the natural candidate is a
#    block no running request is using, which is exactly what a cached prefix is. An
#    evicted prefix cannot be reused, even when a request that matches it arrives
#    later.
#
# ### The same idea, a different data structure
#
# A flat hash table is not the only way to answer "which prefixes do I already
# have". SGLang keeps them in a **radix tree** over token ids. If you are interested, read the *RadixAttention* paper
# ([Zheng et al., NeurIPS '24](https://arxiv.org/abs/2312.07104)).
# Our simulator implements the hashed-block scheme from vLLM.
#
# ### Where the cached blocks live
#
# Our simulator simulates the simplest design, where the KV and prefix caches are in GPU memory.
# When the GPU memory is full, something needs to be evicted.
# Production systems can be much more complex. vLLM
# can offload blocks to CPU memory; [LMCache](https://github.com/LMCache/LMCache)
# adds a reusable KV store with local and remote backends; and Mooncake
# ([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079)), which serves the Kimi chatbot, goes furthest — the KV cache
# may live in CPU DRAM, SSDs, or even remote nodes.
# In such a complex setup, requests must be routed to the node where relevant KV lives, or alternatively,
# the KV must be pulled from CPU DRAM, SSDs, or remote nodes over the network, which adds extra overhead.
# Sometimes, doing so is still beneficial; at other times, just redoing the prefill can be better.
#

# %% [markdown]
# ## A multi-turn workload
#
# Let's build a trace of conversations. Each session is a user and an assistant
# taking turns: a 200-token user message, a 200-token answer, then another user
# message appended to everything that came before, and so on. Turn *k+1* is
# released only after turn *k* finishes (`dep`), plus a **think time** standing in
# for the human reading the answer.
#
# To simulate how prefix cache works, we need the *token ids*. (Remember, only the same sequence of token ids causes a prefix hit!)
# So, the traces in this lecture carry a `token_ids` column,
# instead of only specifying the lengths as in lectures 1–4.
# Prefix caching is off by default in `lsg.simulate`
# and switched on with `prefix_caching=True`.
# We generate requests with random token ids, so that prefix cache hits occur only where we intend them to.
#
# For simplicity, we do not model the system prompt. Turn 0's prompt
# is just the user's question, so the only thing two requests can share is the
# history of one conversation. A shared preamble (system prompt) comes in the next experiment.
#
# This is also a deliberately clean workload — fixed-length turns, every turn
# appending to the one before, nothing matching by accident. Real conversations are
# messier: turns vary in length, users edit or regenerate a message (which rewrites
# the history and throws the prefix away), and clients trim old turns to fit a
# context limit (which changes the *start* of the prompt, the worst place to change
# anything). Read the KV cache hit rates below as the shape of the effect rather than a
# number to expect; lecture 6 measures 60% on a recorded agent workload.

# %%
rng = np.random.default_rng(0)
def toks(n):
    """n token ids that appear nowhere else (so nothing matches by accident)."""
    return rng.integers(10, 150_000, n).tolist()

def chat_trace(n_sessions=24, n_turns=6, question=200, answer=200, preamble=0,
               preamble_first=True, think=5.0, long_answer=None, long_every=4, name=None):
    """Multi-turn conversations, optionally sharing a `preamble`-token system prompt.

    With `long_answer`, every `long_every`-th session asks for that many output
    tokens instead of `answer`, so the sessions do not all cost the same.
    """
    shared = toks(preamble)
    rows = []
    for s in range(n_sessions):
        context = []
        reply_len = long_answer if (long_answer and s % long_every == 0) else answer
        for t in range(n_turns):
            user = toks(question)
            # Turn 0 starts the conversation; later turns append to the context.
            context = (shared + user if preamble_first else user + shared) if t == 0 \
                      else context + user
            reply = toks(reply_len)
            rows.append(dict(prefill=len(context), decode=reply_len, ids=context + reply,
                             session=s, turn=t, dep=[] if t == 0 else [t - 1],
                             think=0.0 if t == 0 else think, rid=1000 * s + t))
            context = context + reply    # the next turn sees this answer too
    df = pd.DataFrame(rows)
    path = lsg.make_trace(df.prefill, df.decode, len(df), name=name, token_ids=df.ids,
                          session_id=df.session, turn_id=df.turn, dep=df.dep,
                          think_time=df.think, request_id=df.rid)
    return path, df

chat, chat_df = chat_trace()
chat_df[["session", "turn", "prefill", "decode", "think"]].head(7)

# %% [markdown]
# Turn 0 prefills 200 tokens, and every later turn 400 more than the one before:
# the user's new message plus the assistant's previous answer. Only those 400
# tokens are new; everything else was computed on this replica seconds ago.
#
# ## Does it help?
#
# Run the same trace with prefix caching off and on.

# %%
def run(trace, df, **kwargs):
    return lsg.simulate(**SYSTEM, trace=trace, num_requests=len(df), **kwargs)

cols = ["TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT p50 (ms)", "throughput (tok/s)",
        "total execution time (s)"]
runs = {f"prefix caching {'on' if pc else 'off'}": run(chat, chat_df, qps=1.0, prefix_caching=pc)
        for pc in (False, True)}
pd.DataFrame({k: {**r.summary()[cols], "prompt tokens computed": r.requests["request_num_prefill_tokens"].sum()
                                       - r.requests["request_num_prefill_tokens_cached"].sum(),
                  "KV cache hit rate": r.cache_hit_rate}
              for k, r in runs.items()}).round(2)

# %% [markdown]
# 83% of all prompt tokens do not need prefilling at all: their KV cache is simply
# reused from an earlier turn. The median TTFT falls from about
# 350 ms to 85 ms, and the tail by a factor of sixteen: a cache hit removes most
# of the queueing too, because the prefill it skips would otherwise have delayed
# every request behind it.
#
# TPOT improves as well, for the reason lecture 3 gave: shorter prefills mean
# fewer steps that carry a big prefill chunk alongside the decodes.
#
# The gain is not spread evenly over the conversation. Let's look per turn.

# %%
hit = runs["prefix caching on"].requests.copy()
hit["turn"] = hit["Request Id"] % 1000
per_turn = hit.groupby("turn").agg(prompt=("request_num_prefill_tokens", "mean"),
                                   cached=("request_num_prefill_tokens_cached", "mean"),
                                   ttft=("prefill_e2e_time", "mean"))
miss = runs["prefix caching off"].requests.copy()
miss["turn"] = miss["Request Id"] % 1000
per_turn["ttft_off"] = miss.groupby("turn")["prefill_e2e_time"].mean()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].bar(per_turn.index, per_turn["cached"], label="reused from cache", color="C0")
ax[0].bar(per_turn.index, per_turn["prompt"] - per_turn["cached"],
          bottom=per_turn["cached"], label="computed", color="C3")
ax[0].set(xlabel="turn", ylabel="prompt tokens", title="Where the prompt comes from")
ax[1].plot(per_turn.index, 1e3 * per_turn["ttft_off"], "o-", color="C3", label="caching off")
ax[1].plot(per_turn.index, 1e3 * per_turn["ttft"], "o-", color="C0", label="caching on")
ax[1].set(xlabel="turn", ylabel="mean TTFT (ms)", title="Time to first token", ylim=(0, None))
for a in ax:
    a.legend(fontsize=9)
fig.tight_layout()

# %% [markdown]
# Without caching, a conversation gets more expensive with every turn: the prompt
# grows, so TTFT grows. With caching, the *computed* part of the prompt is flat —
# always the 400 new tokens — and so is TTFT.
#
# (Turn 0 is the exception. Its prompt is new, so it is a full miss, and it is the
# only turn whose TTFT the cache cannot improve.)

# %% [markdown]
# ## Prompt layout: where the shared text goes
#
# Now the cross-user case from the introduction. These 60 requests belong to 60
# different users with nothing in common except the application they are talking
# to — no conversation history to reuse, and the first user to arrive gets no help
# from the cache at all. What they share is the preamble the application prepends:
# system prompt, tool definitions, few-shot examples.
#
# We give them a shared preamble of 0, 512, or 2,048 tokens, placed either before
# the user's question or after it. The text is the same in both cases; only the
# order differs.

# %%
layout = {}
for preamble in (0, 512, 2048):
    for first in (True, False):
        if preamble == 0 and not first:
            continue
        tr, df = chat_trace(n_sessions=60, n_turns=1, question=200, answer=100,
                            preamble=preamble, preamble_first=first)
        r = run(tr, df, qps=2.0, prefix_caching=True)
        label = f"{preamble}-token preamble" + ("" if first else ", after the question")
        layout[label] = {"prompt tokens": int(r.requests["request_num_prefill_tokens"].mean()),
                         "KV cache hit rate": r.cache_hit_rate,
                         **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)"]]}
pd.DataFrame(layout).T.round(2)

# %% [markdown]
# With the preamble first, 90% of the prompt tokens are shared and the median TTFT
# barely moves as the preamble grows from nothing to 2,048 tokens: one request pays
# for the preamble and the other 59 read its KV cache.
#
# If you move the identical text behind the user's question, the KV cache hit rate collapses to
# zero. The first block already differs between users, so the chained hash gives
# every later block a different name too, and a 2,048-token preamble now costs a
# 17× higher median TTFT than the same preamble placed first.
#
# :::{note}
# This is the practical rule for anyone *writing* prompts for a served
# application: **most stable first, most variable last**. System prompt, tool
# definitions, and retrieved documents that change rarely go at the top; the
# user's turn goes at the bottom. A timestamp or a session id at the start of a
# system prompt is enough to disable sharing between users entirely.
# :::
#
# Hosted APIs make the same rule explicit, and their details are worth knowing
# because that is the deployment most readers meet first.
# [Anthropic's prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
# for instance, is opt-in and manual: you place up to four `cache_control`
# breakpoints, everything before a breakpoint is cached for 5 minutes by default
# (1 hour as an option), a cache *write* costs more than an ordinary input token
# while a *read* costs far less, and a prefix shorter than a model-dependent minimum
# (512 to 4,096 tokens) is silently not cached at all. The request is rendered in a
# fixed order — tool definitions, then system prompt, then the conversation — so a
# frozen tool list and a stable system prompt fall naturally at the front. Other
# providers match prefixes automatically, with no markup. Either way, what the
# application controls is the same thing we are measuring here: what goes first.

# %% [markdown]
# ## The cache is finite
#
# Cached blocks live in the same pool as the KV cache of the *running* requests.
# When a system needs a block and none is free, it evicts the least recently used
# cached block — a prefix someone might have come back to.
#
# Our TP-2 A100 GPU setup holds about 350,000 tokens of KV cache, far more than this small
# workload needs, which is why nothing was evicted above. Real replicas are not so
# comfortable: they run hundreds of concurrent requests with long contexts. We can
# emulate that pressure by shrinking the pool with `kv_blocks` (the number of
# 16-token blocks) and watching a 40-session workload, whose live context is about
# 90,000 tokens, fit in less and less memory.

# %%
chat40, chat40_df = chat_trace(n_sessions=40, n_turns=6, think=10.0, name="chat40")

def cache_row(r):
    return {"KV cache hit rate": r.cache_hit_rate,
            "evictions": int(r.cache.loc[0, "evictions"]),
            **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)"]]}

# The first row is the same workload with prefix caching switched off, for reference.
rows = {"no prefix cache": cache_row(run(chat40, chat40_df, qps=2.0, prefix_caching=False))}
sizes = [3000, 4000, 5000, 6000, 7000, 8000]
sweep_rows = [cache_row(run(chat40, chat40_df, qps=2.0, prefix_caching=True, kv_blocks=b))
              for b in sizes]
rows.update({f"{b * 16:,}": row for b, row in zip(sizes, sweep_rows)})
cache_sweep = pd.DataFrame(sweep_rows, index=[b * 16 for b in sizes])
pd.DataFrame(rows).T.rename_axis("KV cache size (tokens)").round(2)

# %%
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(cache_sweep.index, cache_sweep["KV cache hit rate"], "o-", color="C0")
ax.set(xlabel="KV cache size (tokens)", ylabel="KV cache hit rate", ylim=(0, 1))
ax2 = ax.twinx()
ax2.plot(cache_sweep.index, cache_sweep["evictions"], "s--", color="C3")
ax2.set_ylabel("blocks evicted", color="C3")
ax2.grid(False)
ax.set_title("Hit rate and evictions against KV cache size")
fig.tight_layout()

# %% [markdown]
# Above about 96,000 tokens of KV cache — enough to hold every live session's
# history — eviction all but stops and the KV cache hit rate is the full 83%. Below
# it, blocks are thrown away between turns and have to be recomputed: at 48,000
# tokens the hit rate has fallen to 37%, and the tail TTFT is more than thirty times
# worse than at 96,000, because evicted sessions re-prefill their whole history and
# preempt each other while doing it. A cramped cache is still better than none —
# the median TTFT is four times lower than the no-cache row — but the tail is close
# to it.
#
# Again, unlike our simulator, which simply **drops** the least recently used
# cached block, many production systems **spill it to CPU memory or SSD**
# (vLLM's offloading, LMCache, Mooncake's cluster-wide pool) and fetch it back when
# the session returns. This field still has a lot of interesting research questions.

# %% [markdown]
# ## Prefix-cache-aware routing
#
# When each replica (e.g., a group of TP-2 GPU servers running inference) keeps its cached blocks to itself,
# requests can benefit from prefix caching only if the replica they are allocated to holds the prefix blocks that they need.
# If turn 2 of a conversation is handled by a
# different replica than turn 1, the prefix is on the wrong GPU and the request is a
# full miss.
# To make sure that requests are mostly routed to the replica that holds useful prefix blocks for them,
# real clusters implement KV-aware routers. NVIDIA's
# [Dynamo](https://github.com/ai-dynamo/dynamo) has a **KV-aware router**, scoring each replica by how much of
# the incoming prompt it already holds and weighing that against how loaded it is;
# the `dynamo_kv` policy below is modelled on exactly that scoring. Mooncake
# ([Qin et al., FAST '25](https://arxiv.org/abs/2407.00079))
# implements a similar KV-aware scheduling policy, but it can additionally fetch blocks from a remote node
# when doing so is worth it, which gives it a larger decision space.
#
# For our experiment below, let's simply assume that KV blocks cannot move around, and it will be a full miss if
# requests are routed to a wrong replica.
#
# The conversations we have used so far are too well behaved, with all the users asking the exact same length questions
# and getting the same length answer. Here, we introduce a messier workload.
# There are 160 users. Every user shares the same 4,096-token system prompt,
# followed by a short conversation of three turns.
# The new part is that the users are no longer equally expensive: one user in four
# asks something that takes a 2,000-token answer, while the other three get 200
# tokens. That is enough to make the replicas unequal too. A replica that happens to
# be decoding three long answers at the same time is busy for a while; a replica
# that only has short ones is free again in a second.
#
# This gives a router two things to think about at once, and they do not always
# agree.
#
# The first is the cache. A replica is a good place to send a request if it already
# holds the tokens of that request. There are now two ways that can happen. The
# replica may hold *this user's* earlier turns, which is the kind of reuse we have
# been measuring so far. Or it may simply hold the 4,096-token system prompt, which
# every user sends and which therefore ends up on every replica within the first few
# requests. The second kind of hit is much easier to come by, and it means that
# moving a user to a different replica is no longer a disaster: the new replica
# still has 4,096 of the 4,200 to 4,800 tokens in the prompt.
#
# The second is the load. If we send a request to a replica that is already working
# on a long answer for somebody else, it waits.
#
# Let's build that workload.

# %%
shared_app, shared_app_df = chat_trace(n_sessions=160, n_turns=3, question=100, answer=200,
                                       preamble=4096, think=1.0, long_answer=2000,
                                       long_every=4, name="shared_app")

# %% [markdown]
# We send it to four replicas, with three routing policies.
#
# - `round_robin` sends each request to the next replica in turn. It does not look
#   at who sent the request, at what the replica holds, or at how busy it is. Every
#   replica gets the same number of requests, but a user's second turn almost always
#   lands somewhere other than the first, so its history has to be prefilled again.
# - `sticky_lor` picks a replica for each *user*, once. The user's first request goes
#   to the replica with the fewest outstanding requests, and every later turn from
#   that user follows it there. This gets the most hits of the three, but the choice
#   is made at the beginning and never revisited.
# - `dynamo_kv` decides separately for every request. It gives each replica a score —
#   the number of prompt blocks that replica would still have to compute, plus the
#   number of KV blocks its running requests are currently holding — and sends the
#   request to the lowest score. In other words it prefers the replica that already
#   has your tokens, but it will send you elsewhere if that replica is busy right
#   now.
#
# We run each policy twice, because routing only matters when the replicas are
# actually short of something.
#
# The first run leaves the scheduler at its default limit of 128 concurrent requests
# per replica. Four replicas can then work on 512 requests at once, which is more
# than this workload ever has in flight. Nothing waits, and an uneven split of users
# across replicas costs nothing.
#
# The second run limits each replica to 32 concurrent requests. A real replica runs
# into a limit like this when its KV cache fills up (lecture 2): beyond some number
# of simultaneous requests there is no memory for another one, and the rest wait in
# a queue. This is the regime where the router's choice can be wrong.
#
# :::{note}
# These prompts are longer than the ones earlier in the lecture, and a batch of them
# can exceed the context range the course's runtime predictor is fitted for, so
# these runs pass `max_tokens=65536`. The first one fits a wider predictor grid,
# which takes about a minute. Lecture 6 reuses the same grid.
# :::

# %%
def route(policy, cap):
    r = run(shared_app, shared_app_df, qps=16.0, num_replicas=4, prefix_caching=True,
            global_scheduler=policy, batch_size_cap=cap, max_tokens=65536)
    return {"KV cache hit rate": r.cache_hit_rate,
            **r.summary()[["TTFT p50 (ms)", "TTFT p99 (ms)", "E2E p50 (s)",
                           "total execution time (s)"]]}

POLICIES = ("round_robin", "sticky_lor", "dynamo_kv")
pd.DataFrame({p: route(p, 128) for p in POLICIES}).T.round(2)

# %% [markdown]
# With that much room, the simplest cache-aware answer is the best one.
# `sticky_lor` has the highest hit rate (0.97), the lowest TTFT at both the median
# and the tail, and it gets through the whole workload in the least time. Nothing is
# ever short of capacity, so there is no reason to think about load at all, and the
# only thing left to optimise is the cache — which pinning each user to one replica
# does perfectly. `dynamo_kv` gives away a little of the hit rate to balance a load
# that did not need balancing, and `round_robin` gives away more.
#
# Now the same three policies, with each replica limited to 32 concurrent requests.

# %%
pd.DataFrame({p: route(p, 32) for p in POLICIES}).T.round(2)

# %% [markdown]
# Every number has moved, and in different directions. Taking the policies one at a
# time:
#
# **`round_robin`** has the same hit rate as before, 0.87, because its routing does
# not depend on how loaded anything is. What changed is what that costs. The
# prefills it repeats are no longer free work done in spare capacity; they occupy
# replicas that other requests are now waiting for. The clearest sign is the last
# column: the cluster needs 366 seconds to finish the same 160 users, about a third
# longer than the other two policies, and its tail TTFT is a minute and a half.
# Wasted prefill turns into lost capacity as soon as capacity is scarce.
#
# **`sticky_lor`** keeps its 0.97 hit rate and still finishes quickly, but its median
# TTFT is now 1.1 seconds — thirteen times worse than the same policy had with
# headroom. The reason is not that it chose badly: it spread the users evenly, and
# the expensive users ended up spread evenly too. The reason is that it chose
# *once*. A user is attached to one replica for the rest of the conversation, so
# when that replica happens to be in the middle of a couple of long answers, the
# user waits, even though another replica is free at that moment and holds the same
# system prompt.
#
# **`dynamo_kv`** ends up at 0.92 — it loses a few hits whenever it moves a user
# away from their own history — and in exchange it never leaves a request queued
# behind a replica that is busy at that instant. It has the best median TTFT, 164 ms,
# and the best median end-to-end time, and it finishes as fast as `sticky_lor`. The
# shared system prompt is what makes this trade cheap: a moved user still finds most
# of its prompt on the new replica.
#
# No policy wins on everything. `sticky_lor` still has much the best tail, and
# `dynamo_kv`'s is the second worst. That is worth understanding, because it comes
# straight out of the score: `dynamo_kv` measures a replica's load as the KV blocks
# its *running* requests hold, which is not the same as the length of its queue. A
# replica that is already at its limit of 32 concurrent requests, all of them small,
# still looks cheap, so the router keeps sending to it and the queue behind it grows.
# The requests caught in that queue are the tail.
#
# ## Tuning the trade-off
#
# How much a KV-aware router should care about the cache, relative to load, is a
# knob rather than a law. In the score above, the cache term is multiplied by
# `overlap_score_weight`, which Dynamo also exposes. The default is 1. Larger values
# mean "prefer the replica that has my tokens, even if it is busy"; smaller values
# mean "just balance the load".
#
# It is worth knowing roughly how big the two terms are here before turning the
# knob. The cache term is at most about 4,800/16 = 300 blocks, which is one whole
# prompt, and usually far less because of the shared system prompt. The load term is
# the KV held by up to 32 running requests, which runs into the thousands of blocks.
# So at weight 1 this router is mostly balancing load and using the cache to break
# ties.

# %%
weights = {}
for w in (0.25, 1.0, 4.0):
    r = run(shared_app, shared_app_df, qps=16.0, num_replicas=4, prefix_caching=True,
            global_scheduler="dynamo_kv", batch_size_cap=32, max_tokens=65536,
            extra={"--dynamo_style_k_v_global_scheduler_config_overlap_score_weight": w})
    weights[w] = {"KV cache hit rate": r.cache_hit_rate,
                  **r.summary()[["TTFT p50 (ms)", "E2E p50 (s)", "total execution time (s)"]]}
pd.DataFrame(weights).T.rename_axis("overlap_score_weight").round(2)

# %% [markdown]
# At 0.25 the cache barely counts, the hit rate drops towards round-robin's, and
# latency gets worse. At 4 the cache term starts to outweigh the load term, so the
# router behaves like a sticky one that has stopped watching how busy anything is:
# the hit rate does not improve, and the median end-to-end time more than doubles,
# from 9.8 to 26.2 seconds. The default of 1 happens to be the balanced point for
# this workload. For a workload with shorter prompts, or more replicas, or less
# shared text, the balanced point would sit somewhere else — which is the honest
# summary of KV-aware routing: it is a cost function with a weight in it, and the
# weight depends on your traffic.
#
# Routing on its own is the subject of {doc}`08-cluster-routing`.

# %% [markdown]
# ## What prefix caching does not do
#
# Only prompt tokens can be reused. The answer still has to be generated one token
# at a time at the memory-bound speed of lecture 1, so a workload with short prompts
# and long answers gains very little from prefix caching, however repetitive its
# prompts are.
#
# Cached blocks also cost memory. A block kept for a prefix nobody comes back to is
# a block the running requests cannot use, so a replica that caches too eagerly
# ends up with smaller batches and less throughput. Offloading to CPU or SSD moves
# that problem rather than removing it: something still has to decide which
# prefixes are worth keeping and where to put them.
#
# Skipping most of the prefill work also changes what the replica spends its time
# on. Its steps now carry fewer prefill tokens, so it is more decode-dominated than
# the same deployment without caching, and the chunk size (lecture 3) and
# prefill:decode split (lecture 4) that were right before may not be right after.
#
# Finally, sharing blocks between users has a consequence beyond throughput. The
# preamble experiment worked because two strangers' requests touched the same
# blocks, and a hit is faster than a miss in a way anyone can measure: from its own
# TTFT, a request can tell whether the prefix it sent was already cached, and
# therefore whether someone else sent the same text recently. This is a timing side
# channel. It matters when the shared opening is not public — a tenant's
# confidential system prompt, or a document pasted above a question — and the usual
# answer is to give each tenant its own cache namespace and share only text that
# has been declared public, at some cost in hit rate.
#
# ## What is realistic here, and what is not
#
# The labs in this lecture are small on purpose, and a few of their simplifications
# would change the numbers in a real deployment. Worth having in mind before you
# quote any of them:
#
# | | In these labs | In a production deployment |
# |---|---|---|
# | Block size | 16 tokens | 16 in vLLM, 32 in TensorRT-LLM, token-granular in SGLang |
# | Matching | exact longest prefix, chained block hashes | the same in every major engine; research systems also reuse non-prefix segments |
# | Where blocks live | one GPU pool per replica, nothing below it | HBM, then CPU DRAM, then SSD, sometimes pooled across the whole cluster |
# | Losing a block | evicted and gone, so a miss costs a full re-prefill | usually demoted a tier, so a miss costs a transfer; hosted APIs also expire entries on a timer |
# | Sharing | anything with a matching prefix, including across users | the same by default, isolated per tenant where that matters |
# | Conversations | fixed-length turns, strictly append-only | variable, edited, regenerated, trimmed to a context limit |
# | Prompts | random token ids, so sharing is exactly what we arrange | real text, where a stray timestamp decides everything |
#
# What does carry over is the mechanism and the direction of every effect: which
# tokens can be reused, what destroys a hit, and which knobs trade one against
# another.
#
# ## Summary
#
# | | Effect |
# |---|---|
# | What is reused | KV blocks of a matching **prefix**, named by a chained hash of their tokens |
# | Biggest win | later turns of one conversation, and long openings (system prompt, tool definitions, pasted documents) shared across *different* users |
# | Who shares | any two requests starting with the same tokens, whether or not they belong to the same user |
# | Effect on TTFT | the computed part of the prompt stops growing with the conversation |
# | Effect on TPOT / decode | indirect only (fewer and shorter prefill chunks per step) |
# | Killed by | variable text placed before shared text; eviction under memory pressure; routing to another replica |
# | Interacts with | router policy (affinity vs. load), chunk size, KV-cache capacity |
#
# ## Exercises
#
# :::{admonition} Try it
# :class: exercise
# Each exercise asks you to change code above and rerun it. Write down your
# prediction *before* you run.
#
# 1. **How much more load can the replica take?** 83% of the prompt tokens
#    disappear, but prefill is only one part of the work. Sweep `qps` from 1 to 8 on
#    `chat40` with caching off and on, and for each find the highest arrival rate
#    that keeps p99 TTFT under 2 s (the goodput idea from lecture 2). Is the
#    improvement anywhere near 83%? Account for the gap.
# 2. **Block granularity.** `chat_trace` writes `block_size=16`, vLLM's default.
#    Rebuild the trace at 32 (TensorRT-LLM's) and at an exaggerated 256, and compare
#    hit rates. Which workload loses more from coarse blocks: the multi-turn one or
#    the shared-preamble one? Why would anyone still want a larger block, given
#    what the lecture said about moving blocks between memory tiers?
# 3. **A noisy neighbour.** Add to `chat40` a handful of extra sessions whose
#    prompts are long (3,000 tokens) and share nothing with anything else, as a
#    second tenant would. Run the mixture at `kv_blocks=6000`, where the 40
#    conversations on their own had almost no evictions. How much hit rate do the
#    ordinary conversations lose, and how few noisy sessions does it take to matter?
#    What does that say about serving several tenants from one replica?
# 4. **Does caching change the best chunk size?** Repeat lecture 3's `chunk_size`
#    sweep (128, 512, 2048) on `chat40` with caching off and on. Does the best
#    chunk size move?
# :::
#
# ## Check your understanding
#
# Click an answer to check it. Wrong answers can be retried.

# %% cellView="form" tags=["remove-input"]
#@title Questions (run this cell)
lsg.quiz([
    {"q": "Why is the KV cache split into fixed-size blocks instead of one contiguous array per request?",
     "options": ["To make the block hashes cheaper to compute",
                 "Because a request's length is not known in advance, so a contiguous reservation has to be sized for the worst case and most of it is then wasted",
                 "Because a GPU cannot address more than 16 tokens at a time"],
     "answer": 1,
     "explain": "This is PagedAttention (<a href=\"https://arxiv.org/abs/2309.06180\">Kwon et al., SOSP '23</a>): blocks need not be adjacent, so a request over-allocates by at most one block. Giving those blocks names is what makes prefix reuse possible on top of it."},
    {"q": "Why does reusing a cached KV block not change the model's output?",
     "options": ["The error is small enough to ignore",
                 "A token's keys and values depend only on the tokens before it, so the cached values are exactly what a fresh prefill would compute",
                 "The cache is only used for tokens the model has already answered about"],
     "answer": 1,
     "explain": "Attention is causal: the KV of token i is a function of tokens 0..i. If the prefix is identical, the KV is identical."},
    {"q": "Two prompts are identical except for a session id in the <b>first</b> 10 tokens. How much do they share in the cache?",
     "options": ["Everything after the session id", "Nothing", "Half"], "answer": 1,
     "explain": "A block's hash chains in the hash of the previous block, so a difference anywhere early gives every later block a different name."},
    {"q": "In the multi-turn experiment, which turn benefits least from prefix caching?",
     "options": ["The first turn", "The last turn", "All turns benefit equally"], "answer": 0,
     "explain": "Turn 0's prompt has never been seen, so it is a full miss. Later turns reuse everything up to the previous answer."},
    {"q": "An application puts a timestamp at the top of its system prompt. What is the likely effect?",
     "options": ["Nothing; the timestamp is tiny",
                 "The KV cache hit rate across users collapses, because the shared text that follows now hashes differently for every request",
                 "TPOT gets worse"],
     "answer": 1,
     "explain": "Size is irrelevant: it is position that matters. Variable text at the top invalidates the prefix for everything after it."},
    {"q": "Why did the KV cache hit rate fall when we shrank the KV cache?",
     "options": ["Smaller caches hash blocks differently",
                 "Cached blocks are evicted to make room for running requests, so a returning turn no longer finds its prefix",
                 "Prefix caching turns itself off below a memory threshold"],
     "answer": 1,
     "explain": "Cached prefixes and live requests share one pool. Under pressure the least recently used cached blocks are evicted, and the next turn has to recompute them."},
    {"q": "Why does round-robin routing lower the KV cache hit rate in a multi-replica cluster?",
     "options": ["It overloads one replica",
                 "Each replica has its own KV cache, so a turn routed elsewhere cannot see the prefix its session built",
                 "It disables prefix caching"],
     "answer": 1,
     "explain": "With a cache per replica, a session has to come back to the GPU that holds its history, which is what affinity-aware routing arranges. Clusters with a shared KV store (Mooncake) can instead fetch the prefix across the network."},
    {"q": "Why do production systems keep evicted KV blocks in CPU memory or on SSD instead of dropping them?",
     "options": ["To survive a GPU crash",
                 "Because fetching a prefix back is roughly an order of magnitude cheaper than prefilling it again",
                 "Because CPU memory is faster than HBM"],
     "answer": 1,
     "explain": "A 9,000-token prefix is ~2.3 GB here: a second or two to recompute, under a tenth of a second to transfer. For short prefixes the comparison flips, which is why the decision is made per request."},
    {"q": "A RAG application retrieves the same ten documents for two users, but in a different order. How much does strict prefix caching reuse?",
     "options": ["All ten documents", "Only up to the first document that differs", "Nothing, ever"],
     "answer": 1,
     "explain": "Prefix matching stops at the first difference, so a reordering destroys almost all of it. Systems like Prompt Cache and CacheBlend relax this, at the price of approximating the attention the chunks never paid to each other."},
    {"q": "A workload has 200-token prompts and 2,000-token answers. How much should prefix caching help?",
     "options": ["A lot: 2,200 tokens are cached",
                 "Very little: almost all the work is decode, which prefix caching cannot skip"],
     "answer": 1,
     "explain": "Only prompt tokens can be reused. Prefix caching pays off when prompts are long and repetitive, which is exactly the agentic workload of the next lecture."},
])
