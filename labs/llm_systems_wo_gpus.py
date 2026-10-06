"""Python interface to the course's LLM-serving simulator.

Every lab notebook does::

    import llm_systems_wo_gpus as lsg
    lsg.setup()                      # one-time install of the simulator (Colab/Binder/local)
    r = lsg.simulate(qps=2, prefill_tokens=512, decode_tokens=128)
    r.summary()                      # TTFT / TPOT / E2E / throughput
    r.requests                       # per-request pandas DataFrame

`simulate` only builds the backend simulator's command line, so any backend flag
it does not expose can still be passed through ``extra={"--flag": value}``.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Backend simulator: where it comes from, the subdirectory holding its code, and
# the Python module to run. Everything backend-specific is confined to these lines
# and the flag names in simulate().
BACKEND_REPO = "https://github.com/psu-paws/Vidur-Agent"
BACKEND_SUBDIR = "Vidur-Agent"
BACKEND_MODULE = "vidur"

WORK_DIR = Path(os.environ.get("LSG_WORK_DIR", Path.home() / ".llm-systems-wo-gpus"))
BACKEND_DIR = Path(os.environ.get("LSG_BACKEND_DIR", WORK_DIR / "backend"))

# Packages the simulator imports, as {import name: pip name}. Deliberately
# unpinned: the backend pins old releases (numpy<2, scikit-learn 1.5) that have no
# wheels for recent Pythons and would compile from source on Colab for 10+ minutes.
# Current releases give identical results. Profiling-only deps (torch, ray, ...)
# are skipped.
_DEPS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "yaml": "pyyaml",
    "ddsketch": "ddsketch",
    "fasteners": "fasteners",
    "randomname": "randomname",
    "wandb": "wandb",
}

# The simulator pre-computes operator runtimes on a grid of (batch size, tokens, ...).
# Its defaults (600k tokens x batch 512) need >10 GB of RAM; this grid fits in
# ~1 GB, trains in ~15 s per (model, device, TP) and covers every lab here.
LITE_GRID = {
    "prediction_max_tokens_per_request": 16384,
    "prediction_max_batch_size": 128,
    "prediction_max_prefill_chunk_size": 4096,
    "num_estimators": 50,
    "max_depth": 16,
    "min_samples_split": 2,
    "k_fold_cv_splits": 2,
}


# Interconnect profiles used for tensor-parallel all-reduce. The DGX profiles
# (NVSwitch) cover TP 2-8; the pairwise-NVLink ones only TP 2-4.
_DEFAULT_NETWORK = {"a100": "a100_dgx", "h100": "h100_dgx", "a40": "a40_pairwise_nvlink"}


def _backend_dir() -> Path:
    return BACKEND_DIR / BACKEND_SUBDIR


def setup() -> Path:
    """Make sure the simulator is downloaded and its dependencies are importable.

    Only missing packages are installed, so on Colab this is usually a handful of
    small pure-Python packages.
    """
    if not (_backend_dir() / BACKEND_MODULE).exists():
        print(f"Downloading the simulator into {BACKEND_DIR} (one-time, ~1 min) ...", flush=True)
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", BACKEND_REPO, str(BACKEND_DIR)], check=True
        )
    missing = [pip for mod, pip in _DEPS.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing {', '.join(missing)} ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", *missing],
            check=True,
        )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Simulator ready at {BACKEND_DIR}")
    return _backend_dir()


def make_trace(prefill_tokens, decode_tokens, n: int, name: str | None = None, *,
               token_ids=None, session_id=None, turn_id=None, dep=None,
               think_time=None, request_id=None, block_size: int = 16) -> Path:
    """Write a request-length trace CSV.

    ``prefill_tokens`` / ``decode_tokens`` may be ints (every request identical) or
    sequences of length ``n``.

    The keyword arguments describe multi-turn sessions, and each is a sequence of
    length ``n``:

    * ``token_ids``: the request's prompt **and** output token ids
      (``len == prefill + decode``). Needed for token-exact prefix-cache matching:
      without it, requests never share a prefix.
    * ``session_id`` / ``turn_id``: which conversation a request belongs to and its
      position in it. Turn *k+1* is released when turn *k* completes.
    * ``dep``: turn ids within the session that must *all* finish before this
      request is released (defaults to the previous turn).
    * ``think_time``: seconds between the release condition and the arrival, e.g.
      how long a tool call took (written as ``inter_request_latency``).
    * ``request_id``: the id each request keeps in ``Result.requests``, so that rows
      can be joined back to the trace.
    """
    def col(v):
        return list(v) if hasattr(v, "__len__") and not isinstance(v, str) else [v] * n

    df = pd.DataFrame({"num_prefill_tokens": col(prefill_tokens),
                       "num_decode_tokens": col(decode_tokens)})
    if token_ids is not None:
        df["token_ids"] = [json.dumps(list(t)) for t in token_ids]
        df["block_size"] = block_size
    if session_id is not None:
        df["session_id"] = col(session_id)
    if turn_id is not None:
        df["turn_id"] = col(turn_id)
    if dep is not None:
        df["dep"] = [json.dumps(list(x)) for x in dep]
    if think_time is not None:
        df["inter_request_latency"] = col(think_time)
    if request_id is not None:
        df["request_id"] = col(request_id)
    if name is None:
        name = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()[:12]
    path = WORK_DIR / "traces" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@dataclass
class Result:
    config: dict
    out_dir: Path
    requests: pd.DataFrame = field(repr=False)
    log: str = field(default="", repr=False)

    @property
    def kv_cache_tokens(self) -> int:
        """Tokens of KV cache that fit on one replica after the weights are loaded."""
        m = re.search(r"GPU blocks=(\d+), block_size=(\d+)", self.log)
        return int(m.group(1)) * int(m.group(2)) if m else -1

    @property
    def cache(self) -> pd.DataFrame:
        """Per-replica prefix-cache statistics.

        ``prefill tokens`` counts every prompt token the replica was asked for,
        ``cached tokens`` the ones it found already in the KV cache, and
        ``evictions`` how many cached blocks it had to throw away to make room.
        ``total execution time (s)`` in :meth:`summary` is the wall-clock span of the
        whole run; the per-request end-to-end latency is ``E2E p50 (s)``.
        """
        rows = {}
        for f in sorted(self.out_dir.rglob("eviction_metrics_replica_*.json")):
            d = json.loads(f.read_text())
            rows[int(f.stem.rsplit("_", 1)[-1])] = {
                "prefill tokens": d["sum_prefill_tokens"],
                "cached tokens": d["sum_kvhit_tokens"],
                "KV cache hit rate": d["token_cache_hit_rate"],
                "evictions": d["num_evictions"],
            }
        return pd.DataFrame(rows).T.rename_axis("replica")

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of all prompt tokens that were served from the prefix cache."""
        c = self.cache
        return float(c["cached tokens"].sum() / max(c["prefill tokens"].sum(), 1))

    @property
    def ttft(self) -> pd.Series:
        """Time to first token (s): arrival -> end of prefill."""
        return self.requests["prefill_e2e_time"]

    @property
    def tpot(self) -> pd.Series:
        """Time per output token (s), averaged over each request's decode phase."""
        return self.requests["decode_time_execution_plus_preemption_normalized"]

    @property
    def e2e(self) -> pd.Series:
        return self.requests["request_e2e_time"]

    def summary(self) -> pd.Series:
        r = self.requests
        makespan = (r["request_arrived_at"] + r["request_e2e_time"]).max()
        return pd.Series({
            "requests": len(r),
            "TTFT p50 (ms)": 1e3 * self.ttft.median(),
            "TTFT p99 (ms)": 1e3 * self.ttft.quantile(0.99),
            "TPOT p50 (ms)": 1e3 * self.tpot.median(),
            "TPOT p99 (ms)": 1e3 * self.tpot.quantile(0.99),
            "E2E p50 (s)": self.e2e.median(),
            "queueing p50 (ms)": 1e3 * r["request_scheduling_delay"].median(),
            "throughput (tok/s)": r["request_num_tokens"].sum() / makespan,
            "total execution time (s)": makespan,
        }).round(3)

    @property
    def steps(self) -> pd.DataFrame:
        """One row per forward pass (needs ``keep_steps=True``): tokens, batch size,
        ``batch_execution_time`` (s), and the ``replica`` that ran it, in execution
        order within each replica."""
        f = glob.glob(str(self.out_dir / "**" / "batch_metrics.csv"), recursive=True)
        if not f:
            raise FileNotFoundError("batch_metrics.csv not found; rerun with keep_steps=True")
        return pd.read_csv(f[0]).drop(columns=["Batch Id"], errors="ignore")

    def cdf(self, metric: str = "batch_size") -> pd.DataFrame:
        """CDF of a batch-level metric (needs ``store_plots=True``).

        e.g. batch_size, batch_num_tokens, batch_num_prefill_tokens,
        batch_num_decode_tokens, batch_execution_time.
        """
        f = glob.glob(str(self.out_dir / "**" / "plots" / f"{metric}.csv"), recursive=True)
        if not f:
            raise FileNotFoundError(f"{metric}.csv not found; rerun with store_plots=True")
        return pd.read_csv(f[0])


def simulate(
    *,
    model: str = "meta-llama/Llama-2-7b-hf",
    device: str = "a100",
    network_device: str | None = None,
    tensor_parallel: int = 1,
    num_replicas: int = 1,
    qps: float | None = 2.0,
    num_requests: int = 100,
    prefill_tokens=512,
    decode_tokens=128,
    trace: str | Path | None = None,
    arrival_times=None,
    scheduler: str = "vllm_v1",
    batch_size_cap: int = 128,
    chunk_size: int = 512,
    global_scheduler: str = "round_robin",
    prefix_caching: bool = False,
    kv_blocks: int | None = None,
    max_tokens: int | None = None,
    replica_groups: dict | str | Path | None = None,
    seed: int = 42,
    store_plots: bool = False,
    keep_steps: bool = False,
    extra: dict | None = None,
    verbose: bool = False,
) -> Result:
    """Run one simulation and return its per-request metrics.

    qps=None sends every request at t=0 (an offline / batch workload).
    arrival_times (seconds, one per request) overrides qps with exact arrivals.
    keep_steps=True records every forward pass, readable as ``Result.steps``.
    Only the ``vllm_v1`` replica scheduler works in the simulator today; its
    ``chunk_size`` and ``batch_size_cap`` knobs emulate the older policies.
    """
    if network_device is None:
        network_device = _DEFAULT_NETWORK.get(device, f"{device}_dgx")
    backend = setup() if not (_backend_dir() / BACKEND_MODULE).exists() else _backend_dir()
    if trace is None:
        trace = make_trace(prefill_tokens, decode_tokens, num_requests)
    grid = dict(LITE_GRID)
    if max_tokens is not None:
        grid["prediction_max_tokens_per_request"] = max_tokens
    lengths = pd.read_csv(trace, usecols=["num_prefill_tokens", "num_decode_tokens"])
    longest = int((lengths["num_prefill_tokens"] + lengths["num_decode_tokens"]).max())
    if longest > grid["prediction_max_tokens_per_request"]:
        raise ValueError(
            f"A request has {longest:,} prompt+output tokens; this run's runtime predictor "
            f"covers at most {grid['prediction_max_tokens_per_request']:,} tokens. Raise "
            "simulate(max_tokens=...) (slower to fit, more memory) or use a shorter trace.")

    args: dict = {
        "--seed": seed,
        "--replica_config_model_name": model,
        "--replica_config_device": device,
        "--replica_config_network_device": network_device,
        "--replica_config_tensor_parallel_size": tensor_parallel,
        "--cluster_config_num_replicas": num_replicas,
        "--global_scheduler_config_type": global_scheduler,
        "--synthetic_request_generator_config_num_requests": num_requests,
        "--length_generator_config_type": "trace",
        "--trace_request_length_generator_config_trace_file": Path(trace).resolve(),
        "--trace_request_length_generator_config_max_tokens": grid["prediction_max_tokens_per_request"],
        "--replica_scheduler_config_type": scheduler,
        "--cache_config_enable_prefix_caching": prefix_caching,
    }
    if kv_blocks is not None:
        args["--cache_config_num_blocks"] = kv_blocks
    if arrival_times is not None:
        # The interval generator replays the gaps between consecutive rows and
        # skips the first one, so a leading 0 makes request i arrive at arrival_times[i].
        at = WORK_DIR / "traces" / ("arrivals_" + hashlib.md5(str(list(arrival_times)).encode()).hexdigest()[:12] + ".csv")
        at.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"arrived_at": [0.0] + [float(x) for x in arrival_times]}).to_csv(at, index=False)
        args["--interval_generator_config_type"] = "trace"
        args["--trace_request_interval_generator_config_trace_file"] = at
    elif qps is None:
        args["--interval_generator_config_type"] = "static"
    else:
        args["--interval_generator_config_type"] = "poisson"
        args["--poisson_request_interval_generator_config_qps"] = qps

    if scheduler in ("vllm_v1", "sarathi"):
        args[f"--{scheduler}_scheduler_config_chunk_size"] = chunk_size
    if scheduler in ("vllm_v1", "vllm", "sarathi", "orca"):
        args[f"--{scheduler}_scheduler_config_batch_size_cap"] = batch_size_cap

    p = "--random_forest_execution_time_predictor_config_"
    for k, v in grid.items():
        args[p + k] = v
    args[p + "num_training_job_threads"] = 2
    args[p + "cache_dir"] = WORK_DIR / "predictor_cache"

    if replica_groups is not None:
        if isinstance(replica_groups, dict):
            path = WORK_DIR / "replica_groups" / (
                hashlib.md5(json.dumps(replica_groups, sort_keys=True).encode()).hexdigest()[:12] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(replica_groups, indent=2))
            replica_groups = path
        args["--cluster_config_replica_groups_config"] = Path(replica_groups).resolve()

    args.update(extra or {})
    run_id = hashlib.md5(json.dumps({k: str(v) for k, v in args.items()}, sort_keys=True).encode()).hexdigest()[:12]
    out_dir = WORK_DIR / "runs" / run_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    args["--metrics_config_output_dir"] = out_dir
    args["--metrics_config_no_timestamp"] = True
    args["--metrics_config_store_plots"] = store_plots
    args["--metrics_config_store_utilization_metrics"] = False
    args["--metrics_config_keep_individual_batch_metrics"] = keep_steps

    cmd = [sys.executable, "-m", f"{BACKEND_MODULE}.main"]
    for k, v in args.items():
        if v is True:
            cmd.append(k)
        elif v is False:
            cmd.append("--no-" + k[2:])
        else:
            cmd += [k, str(v)]

    env = {**os.environ, "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "2"}
    proc = subprocess.run(cmd, cwd=backend, env=env, capture_output=True, text=True)
    if verbose or proc.returncode != 0:
        print(" ".join(cmd))
        print(proc.stdout[-4000:], proc.stderr[-4000:])
    if proc.returncode != 0:
        if "_max_micro_batch_size" in proc.stderr:
            raise RuntimeError(
                "A decode replica was handed more running requests than batch_size_cap. "
                "The simulator's PD-disaggregation mode cannot queue them, so the decode "
                "pool is overloaded: lower qps or add decode replicas.")
        raise RuntimeError("Simulation failed (see log above)")

    csv = glob.glob(str(out_dir / "**" / "request_metrics.csv"), recursive=True)[0]
    cfg = {k.lstrip("-"): v for k, v in args.items()}
    return Result(config=cfg, out_dir=out_dir, requests=_merge_pd_rows(pd.read_csv(csv)),
                  log=proc.stdout + proc.stderr)


def _merge_pd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """With PD disaggregation the simulator writes two rows per request: one from
    the prefill replica (TTFT columns) and one from the decode replica (decode and
    E2E columns). Merge them into one row, keeping both replica ids."""
    if not df["Request Id"].duplicated().any():
        return df
    pre = df[df["prefill_e2e_time"].notna()].set_index("Request Id")
    dec = df[df["prefill_e2e_time"].isna()].set_index("Request Id")
    merged = dec.combine_first(pre)
    merged["prefill_replica"] = pre["replica"]
    merged["decode_replica"] = dec["replica"]
    return merged.reset_index()[list(df.columns) + ["prefill_replica", "decode_replica"]]


# The agent traces that ship with the simulator repo: real LLM requests recorded
# while two agent systems solved GAIA tasks (Kim et al., IISWC '26,
# https://arxiv.org/abs/2606.01725).
GAIA_SUBDIR = "GAIATrace"

# Which model served a request, recognised from the first token ids of its prompt
# (the two systems in the recording used different models for different roles).
GAIA_MODELS = {"[200006, 17360": "gpt-oss-120b", "[200006, 77944": "gpt-4o"}

# Which agent inside the system issued a request (OWL's roles).
GAIA_ROLES = {0: "plan", 1: "coordinate", 2: "web search", 3: "code", 4: "web summarize",
              5: "web plan", 6: "web action", 7: "answer", 8: "document", 9: "other"}


def gaia_dir() -> Path:
    """Directory of the recorded agent traces, downloaded alongside the simulator."""
    d = BACKEND_DIR / GAIA_SUBDIR
    if not d.exists():
        raise FileNotFoundError(f"{d} not found; run lsg.setup() first")
    return d


def _gaia_tool_latency(root: Path) -> dict:
    """{(session file stem, turn id): seconds} from the measured tool benchmark.

    Each tool was re-run several times outside the agent; a turn's latency is the
    mean per tool call, summed over the calls that turn made (the agent awaits
    them one after another).
    """
    per_call: dict = {}
    for f in sorted((root / "raw" / "tool").glob("*.csv")):
        with open(f, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    seconds = float(row["effective_elapsed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if seconds > 0:
                    per_call.setdefault(
                        (row["source_file"], int(row["request_idx"]), row.get("args_json", "")),
                        []).append(seconds)
    out: dict = {}
    for (stem, turn, _args), values in per_call.items():
        out[(stem, turn)] = out.get((stem, turn), 0.0) + sum(values) / len(values)
    return out


def gaia_sessions(num_sessions: int = 20, max_tokens: int | None = None, seed: int = 0,
                  agent: str = "owl") -> pd.DataFrame:
    """Load recorded agent sessions as one DataFrame, one row per LLM request.

    Columns: ``session``, ``turn``, ``role`` (which agent in the system issued it),
    ``model`` (which model served it in the recording), ``dep`` (turns it waited
    for), ``num_prefill_tokens``, ``num_decode_tokens``, ``tool_time`` (s), and
    ``token_ids`` (prompt + output, decodable with :func:`decode`).

    Sessions whose longest request exceeds ``max_tokens`` are skipped, because the
    course's prediction grid only covers requests up to
    ``LITE_GRID["prediction_max_tokens_per_request"]`` tokens (the default here).
    """
    if agent != "owl":
        raise ValueError("only the OWL traces are packaged for the course (agent='owl')")
    if max_tokens is None:
        max_tokens = LITE_GRID["prediction_max_tokens_per_request"]
    root = gaia_dir() / agent
    files = [f for f in sorted((root / "traces" / "session_traces").glob("*.csv"))
             if not f.stem.endswith("_tools")]
    random.Random(seed).shuffle(files)
    tool_latency = _gaia_tool_latency(root)

    frames = []
    for f in files:
        if len(frames) >= num_sessions:
            break
        d = pd.read_csv(f).dropna(subset=["num_prefill_tokens", "num_decode_tokens"])
        if len(d) == 0 or (d["num_prefill_tokens"] + d["num_decode_tokens"]).max() > max_tokens:
            continue
        d = d.reset_index(drop=True)
        deps = [json.loads(x) if isinstance(x, str) else [] for x in d.get("dep", "[]")]
        frames.append(pd.DataFrame({
            "session": len(frames),
            "turn": range(len(d)),
            "role": [GAIA_ROLES.get(int(a), "other") for a in d["agent"]]
                    if "agent" in d else "other",
            "model": [next((m for p, m in GAIA_MODELS.items() if t.startswith(p)), "unknown")
                      for t in d["tokens"]],
            "dep": deps,
            "num_prefill_tokens": d["num_prefill_tokens"].astype(int),
            "num_decode_tokens": d["num_decode_tokens"].astype(int),
            "tool_time": [max([tool_latency.get((f.stem, int(i)), 0.0) for i in dep] or [0.0])
                          for dep in deps],
            "token_ids": d["tokens"],
        }))
    if len(frames) < num_sessions:
        raise ValueError(f"only {len(frames)} sessions fit in {max_tokens:,} tokens per request")
    return pd.concat(frames, ignore_index=True)


def gaia_trace(sessions: pd.DataFrame, name: str | None = None) -> Path:
    """Write the sessions from :func:`gaia_sessions` as a trace CSV for ``simulate``."""
    df = pd.DataFrame({
        "num_prefill_tokens": sessions["num_prefill_tokens"],
        "num_decode_tokens": sessions["num_decode_tokens"],
        "token_ids": sessions["token_ids"],
        "block_size": 16,
        "session_id": sessions["session"],
        "turn_id": sessions["turn"],
        "dep": [json.dumps(list(d)) for d in sessions["dep"]],
        "inter_request_latency": sessions["tool_time"],
        "request_id": 1000 * sessions["session"] + sessions["turn"],
    })
    if name is None:
        name = "gaia_" + hashlib.md5(
            df.drop(columns=["token_ids"]).to_csv(index=False).encode()).hexdigest()[:12]
    path = WORK_DIR / "traces" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def decode(token_ids) -> str:
    """Decode recorded token ids back to text (``o200k_harmony``, as the traces were made)."""
    try:
        import tiktoken
    except ImportError:
        print("Installing tiktoken ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off",
                        "tiktoken"], check=True)
        import tiktoken
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    return tiktoken.get_encoding("o200k_harmony").decode(list(token_ids))


def sweep(param: str, values, **kwargs) -> pd.DataFrame:
    """Run ``simulate(param=v, **kwargs)`` for each value; one summary row per run."""
    rows = []
    for v in values:
        s = simulate(**{param: v}, **kwargs).summary()
        s[param] = v
        rows.append(s)
    return pd.DataFrame(rows).set_index(param)


def catalog() -> pd.DataFrame:
    """Which (device, model, TP) combinations have profiling data, and their limits.

    The simulator's runtime predictors are random forests: they interpolate well inside the
    profiled range but cannot extrapolate past ``max_context`` or ``max_batch``.
    """
    rows = []
    root = _backend_dir() / "data" / "profiling" / "compute"
    for f in sorted(root.glob("*/*/*/attention.csv")):
        d = pd.read_csv(f, usecols=["kv_cache_size", "batch_size", "num_tensor_parallel_workers"])
        rows.append({
            "device": f.parts[-4],
            "model": f"{f.parts[-3]}/{f.parts[-2]}",
            "tensor_parallel": sorted(int(t) for t in d["num_tensor_parallel_workers"].unique()),
            "max_context": int(d["kv_cache_size"].max()) + 64,
            "max_batch": int(d["batch_size"].max()),
        })
    return pd.DataFrame(rows)


_QUIZ_CSS = """
.lsg-quiz{font-family:inherit;max-width:46em}
.lsg-q{border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:.7em .9em;margin:.7em 0}
.lsg-q p{margin:.1em 0 .5em;font-weight:600}
.lsg-opt{display:block;width:100%;text-align:left;margin:.25em 0;padding:.4em .7em;cursor:pointer;
  border:1px solid rgba(128,128,128,.45);border-radius:4px;background:transparent;color:inherit;font:inherit}
.lsg-opt:hover{border-color:#2980b9}
.lsg-opt.right{border-color:#27ae60;background:rgba(39,174,96,.15)}
.lsg-opt.wrong{border-color:#c0392b;background:rgba(192,57,43,.12)}
.lsg-fb{display:none;margin-top:.5em;padding:.45em .7em;border-left:3px solid #2980b9;font-size:95%}
.lsg-score{font-size:90%;opacity:.8}
"""


def quiz(questions: list[dict]):
    """Render clickable multiple-choice questions with instant feedback.

    Each question is ``{"q": str, "options": [str, ...], "answer": int, "explain": str}``
    (``answer`` is the 0-based index of the correct option). Strings may contain
    HTML such as ``<code>``. The output is self-contained HTML+JS, so it works in
    Jupyter, Colab, and on the rendered website (where the saved output is shown).
    """
    from IPython.display import HTML
    return HTML(quiz_html(questions))


def quiz_html(questions: list[dict]) -> str:
    """The HTML+JS behind :func:`quiz`, also used by the docs' ``{quiz}`` directive."""
    import html
    import uuid

    qid = "lsg" + uuid.uuid4().hex[:8]
    parts = [f'<style>{_QUIZ_CSS}</style><div class="lsg-quiz" id="{qid}">']
    for i, q in enumerate(questions):
        parts.append(f'<div class="lsg-q" data-answer="{int(q["answer"])}"><p>{i + 1}. {q["q"]}</p>')
        for j, opt in enumerate(q["options"]):
            parts.append(f'<button class="lsg-opt" data-i="{j}">{opt}</button>')
        parts.append(f'<div class="lsg-fb" data-explain="{html.escape(q.get("explain", ""), quote=True)}"></div></div>')
    parts.append(f'<div class="lsg-score"></div></div>')
    parts.append(f"""<script>(function(){{
  var root=document.getElementById("{qid}"); if(!root) return;
  var total=root.querySelectorAll(".lsg-q").length, done=0, right=0;
  root.querySelectorAll(".lsg-q").forEach(function(q){{
    var ans=+q.dataset.answer, fb=q.querySelector(".lsg-fb"), answered=false;
    q.querySelectorAll(".lsg-opt").forEach(function(b){{
      b.addEventListener("click",function(){{
        var ok=(+b.dataset.i===ans);
        q.querySelectorAll(".lsg-opt").forEach(function(o){{o.classList.remove("right","wrong");}});
        b.classList.add(ok?"right":"wrong");
        if(ok) q.querySelectorAll(".lsg-opt")[ans].classList.add("right");
        fb.style.display="block";
        fb.innerHTML=(ok?"<b>Correct.</b> ":"<b>Not quite &mdash; try again.</b> ")+(ok?fb.dataset.explain:"");
        if(!answered){{answered=true; done++; if(ok) right++;
          root.querySelector(".lsg-score").textContent="First-try score: "+right+" / "+done+(done<total?" (answered "+done+" of "+total+")":"");}}
      }});
    }});
  }});
}})();</script>""")
    return "".join(parts)
