# How This Course Works

## The simulator

Each lab drives an LLM inference simulator. Given a model, a GPU type, a cluster
layout, a scheduler, and a workload, it replays the workload request by request
and reports latency and throughput metrics, as a real serving system's logs would.

It does **not** run the model. It predicts how long each operation (matrix
multiplies, attention, all-reduce, ...) would take on the target GPU, using models
fitted to measurements taken on real hardware. See
{doc}`../reference/how-the-simulator-works`.

## The `llm_systems_wo_gpus` package

The simulator has hundreds of command-line flags. The labs use a small Python
interface,
[`labs/llm_systems_wo_gpus.py`](https://github.com/kwmaeng91/studying-llm-systems-without-gpus/blob/main/labs/llm_systems_wo_gpus.py),
so a simulation is one function call:

```python
import llm_systems_wo_gpus as lsg
lsg.setup()                                   # one-time clone + install

r = lsg.simulate(model="meta-llama/Llama-2-7b-hf", device="a100",
                qps=4, prefill_tokens=512, decode_tokens=128)
r.summary()        # TTFT / TPOT / E2E percentiles, throughput
r.requests         # one row per request (pandas DataFrame)

lsg.sweep("qps", [1, 2, 4, 8])                # one summary row per value
```

The full API is in {doc}`../reference/python-api`.

## Where to run

| Option | Setup | First run | Good for |
|---|---|---|---|
| {doc}`Google Colab <run-in-browser>` | Google account | ~1–2 min | most students |
| {doc}`Binder <run-in-browser>` | none | 2–10 min to start the server | no account at all |
| {doc}`Local install <run-locally>` | Python 3.10+, git | ~2 min | longer projects, saving work |

Every lab needs **< 1 GB of RAM** and 2 CPU cores, so all three options work.
