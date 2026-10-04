# Run in Your Browser

Every lab page has two buttons at the top right:

<span class="launch-btn colab">Open in Colab</span> <span class="launch-btn binder">Launch Binder</span>

## Google Colab (recommended)

1. Click **Open in Colab** on a lab page. The notebook opens directly from GitHub.
2. Choose **Runtime → Run all** (or run cells one by one with <kbd>Shift</kbd>+<kbd>Enter</kbd>).
3. The first cell downloads the `llm_systems_wo_gpus` package and the simulator,
   and installs their dependencies. This takes about a minute.
4. To keep your edits, use **File → Save a copy in Drive**.

:::{tip}
You do **not** need a GPU runtime. The default CPU runtime is what you want. A
GPU runtime would just sit idle and use up your quota.
:::

Colab sessions are temporary. When a session restarts, the setup cell runs again
and the predictor cache (~15 s per model/GPU combination) is rebuilt.

## Binder (no account)

1. Click **Launch Binder**. [mybinder.org](https://mybinder.org) builds or
   reuses a container image for the course and opens JupyterLab.
2. The image already contains the simulator and a warm predictor cache for the
   default model, so the first simulation starts immediately.

Binder is free and needs no login, but launches can take several minutes when the
image has to be rebuilt, and idle sessions shut down after ~10 minutes. **Download
your notebook** (*File → Download*) before you leave.

## Why not run entirely in the browser (WebAssembly)?

We looked at [JupyterLite](https://jupyterlite.readthedocs.io/)/Pyodide. The simulator
runs as a subprocess, reads ~600 MB of profiling data, and fits scikit-learn
models on startup, none of which fits well into a browser tab today. Colab
and Binder give the same "click and run" experience with a real Python kernel.
