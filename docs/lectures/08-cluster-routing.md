# [TEMP] 8. Scaling Out: Replicas and Request Routing

By [Kiwan Maeng](https://kiwanmaeng.com) ([LinkedIn](https://www.linkedin.com/in/kiwan-maeng-23b825165)) and [Claude Code](https://claude.com/claude-code) 🤖

:::{admonition} Status: planned
:class: warning
This lecture is an outline. The simulator features it needs are already
available through `lsg.simulate(num_replicas=..., global_scheduler=...)`.
:::

## Motivation

One replica saturates at a few req/s (lecture 2). Production services run many
replicas behind a **router**, and the routing policy decides how evenly load and
KV-cache reuse are spread across them.

## Planned content

1. **Linear scaling, in theory.** Throughput of *N* replicas vs. 1 under round-robin
   routing, and why p99 latency still improves faster than you might expect
   (statistical multiplexing).
2. **Routing policies** available in the simulator via `global_scheduler`:

   | Value | Policy |
   |---|---|
   | `random`, `round_robin` | load-oblivious baselines |
   | `lor` | least outstanding requests |
   | `lop` | least outstanding prefill tokens |
   | `sticky_round_robin`, `sticky_lor` | keep a session on one replica |
   | `load_aware`, `dynamo_kv` | load- and KV-affinity-aware (Dynamo-style) |

3. **Heterogeneous fleets.** Mixing H100 and A100 replicas with a
   [replica-groups JSON](../reference/simulator-knobs.md#replica-groups) and
   finding the routing policy that does not overload the slow pool.

## Lab sketch

```python
for policy in ["round_robin", "lor", "lop"]:
    print(policy, lsg.simulate(num_replicas=4, qps=40, global_scheduler=policy,
                              trace=mixed_trace).summary()["TTFT p99 (ms)"])
```

:::{note}
Lectures {doc}`05-prefix-caching` and {doc}`06-agentic-workloads` already measured
one consequence of routing — a request only hits the prefix cache on the replica
that holds its prefix, so `sticky_lor` and `dynamo_kv` beat `round_robin` on hit
rate while doing worse at the tail. This lecture takes the policies on their own
terms.
:::
