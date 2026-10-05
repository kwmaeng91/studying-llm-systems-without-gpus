# [TEMP] 7. Agentic Workloads: Sessions, Prefix Caching, and Tool Calls

By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖

:::{admonition} Status: planned
:class: warning
This lecture is an outline. It builds on the main contribution of the simulator:
replaying real multi-turn agent traces
([GAIATrace](https://github.com/psu-paws/Vidur-Agent/tree/main/GAIATrace)).
:::

## Motivation

An AI agent does not send one request. It runs a **session** of dependent LLM
calls: plan, call a tool, read the result, plan again. Each turn's prompt is the
previous turn's prompt plus new text, so turns share long **prefixes**. And
turn *k+1* cannot start until turn *k* has finished *and* its tool call has
returned.

This changes what a good serving system looks like:

- **Prefix caching**: reuse the KV cache of the shared prefix instead of
  re-prefilling it. Hit rates depend on whether the next turn lands on the same
  replica and whether the KV survived eviction in the meantime.
- **Session-aware scheduling**: prioritize sessions that are close to finishing
  (shortest-job-first with starvation protection) to cut task-level latency.
- **Task-level metrics**: users care about how long the *whole agent task* takes,
  not about per-request TTFT.

## Planned content

1. Anatomy of an agent trace: requests per session, prefix overlap, tool-time
   distribution (OWL and MiroThinker on GAIA).
2. Prefix caching on/off (`prefix_caching=True`) and its effect on prefill tokens
   and TTFT; KV eviction under memory pressure.
3. Routing for cache affinity: `sticky_lor` and `dynamo_kv` vs. `round_robin`.
4. Session scheduling: `vllm_v1_scheduler_config_session_priority` and the SJF
   options, measured on task completion time.

Reference: *Characterizing How Complex Agentic AI Systems Handle General Tasks:
A Trace-Based Simulation Study*, IISWC 2026 ([arXiv:2606.01725](https://arxiv.org/abs/2606.01725)).
