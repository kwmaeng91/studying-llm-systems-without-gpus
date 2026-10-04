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

import glob
import hashlib
import importlib.util
import json
import os
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


def make_trace(prefill_tokens, decode_tokens, n: int, name: str | None = None) -> Path:
    """Write a request-length trace CSV.

    ``prefill_tokens`` / ``decode_tokens`` may be ints (every request identical) or
    sequences of length ``n``.
    """
    p = prefill_tokens if hasattr(prefill_tokens, "__len__") else [prefill_tokens] * n
    d = decode_tokens if hasattr(decode_tokens, "__len__") else [decode_tokens] * n
    df = pd.DataFrame({"num_prefill_tokens": list(p), "num_decode_tokens": list(d)})
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
            "makespan (s)": makespan,
        }).round(3)

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
    scheduler: str = "vllm_v1",
    batch_size_cap: int = 128,
    chunk_size: int = 512,
    global_scheduler: str = "round_robin",
    prefix_caching: bool = False,
    replica_groups: dict | str | Path | None = None,
    seed: int = 42,
    store_plots: bool = False,
    extra: dict | None = None,
    verbose: bool = False,
) -> Result:
    """Run one simulation and return its per-request metrics.

    qps=None sends every request at t=0 (an offline / batch workload).
    Only the ``vllm_v1`` replica scheduler works in the simulator today; its
    ``chunk_size`` and ``batch_size_cap`` knobs emulate the older policies.
    """
    if network_device is None:
        network_device = _DEFAULT_NETWORK.get(device, f"{device}_dgx")
    backend = setup() if not (_backend_dir() / BACKEND_MODULE).exists() else _backend_dir()
    if trace is None:
        trace = make_trace(prefill_tokens, decode_tokens, num_requests)

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
        "--trace_request_length_generator_config_max_tokens": LITE_GRID["prediction_max_tokens_per_request"],
        "--replica_scheduler_config_type": scheduler,
        "--cache_config_enable_prefix_caching": prefix_caching,
    }
    if qps is None:
        args["--interval_generator_config_type"] = "static"
    else:
        args["--interval_generator_config_type"] = "poisson"
        args["--poisson_request_interval_generator_config_qps"] = qps

    if scheduler in ("vllm_v1", "sarathi"):
        args[f"--{scheduler}_scheduler_config_chunk_size"] = chunk_size
    if scheduler in ("vllm_v1", "vllm", "sarathi", "orca"):
        args[f"--{scheduler}_scheduler_config_batch_size_cap"] = batch_size_cap

    p = "--random_forest_execution_time_predictor_config_"
    for k, v in LITE_GRID.items():
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
        raise RuntimeError("Simulation failed (see log above)")

    csv = glob.glob(str(out_dir / "**" / "request_metrics.csv"), recursive=True)[0]
    cfg = {k.lstrip("-"): v for k, v in args.items()}
    return Result(config=cfg, out_dir=out_dir, requests=pd.read_csv(csv),
                  log=proc.stdout + proc.stderr)


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
