# Sphinx configuration for "LLM Systems Without GPUs".
# Build locally:  sphinx-build -b html docs docs/_build/html

project = "LLM Systems Without GPUs"
author = "PAWS Lab, Penn State"
copyright = "2026, PAWS Lab"
release = "0.1"

# Where the course lives on GitHub. Colab/Binder buttons and the notebooks'
# setup cell are built from this, so update it once the repo is pushed.
github_user = "kwmaeng91"
github_repo = "studying-llm-systems-without-gpus"
github_version = "main"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

# Notebooks are committed with outputs (executed on CPU with labs/build_notebooks.py);
# the docs build itself never runs the simulator.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "attrs_inline"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "sphinx_rtd_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
}
html_context = {
    "display_github": True,
    "github_user": github_user,
    "github_repo": github_repo,
    "github_version": github_version,
    "conf_py_path": "/docs/",
}
