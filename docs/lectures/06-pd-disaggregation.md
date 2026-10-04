# [TEMP] 6. Prefill–Decode Disaggregation

:::{admonition} Status: planned
:class: warning
This lecture is an outline. The simulator supports PD disaggregation through
replica groups with `"role": "prefill"` / `"decode"`. The lab will pass such a
config with `lsg.simulate(replica_groups={...})`.
:::

## Motivation

Lecture 3 showed that prefills and decodes interfere when they share a GPU.
Systems such as DistServe, Splitwise, Mooncake, and NVIDIA Dynamo instead run
them on **separate pools of GPUs**. Each request is prefilled on a prefill
replica, its KV cache is **transferred** over the network, and it then decodes on
a decode replica.

## Planned content

1. **Interference-free decode.** TPOT p99 with and without disaggregation on the
   lecture 3 mixed workload.
2. **The price: KV transfer.** The simulator models transfer time from the KV size
   and the interconnect (`network_device`, `cross_node`). How large can a prompt
   get before transfer time shows up in TTFT?
3. **Pool sizing.** For a fixed budget of 8 GPUs, sweep the prefill:decode split
   (1:7 … 4:4) and the TP degree of each pool, and find the best goodput.

## Lab sketch

```python
pd_config = {
  "replica_groups": [
    {"role": "prefill", "num_replicas": 1,
     "replica_config": {"model_name": "meta-llama/Llama-2-7b-hf", "tensor_parallel_size": 2,
                        "device": "a100", "network_device": "a100_dgx", "pd_disaggregation": 1},
     "replica_scheduler_config": {"type": "vllm_v1", "batch_size_cap": 128}},
    {"role": "decode", "num_replicas": 1,
     "replica_config": {"model_name": "meta-llama/Llama-2-7b-hf", "tensor_parallel_size": 2,
                        "device": "a100", "network_device": "a100_dgx", "pd_disaggregation": 1},
     "replica_scheduler_config": {"type": "vllm_v1", "batch_size_cap": 128}},
  ],
  "replica_groups_pools": [{"prefill": [0], "decode": [1], "cross_node": False}],
}
r = lsg.simulate(replica_groups=pd_config, trace=mixed_trace, qps=8)
```
