# TreeTracer

A desktop app for exploring posterior tree distributions from
[BEAST X](https://beast.community/) MCMC runs. Load `.trees` files
directly, compute pairwise Robinson–Foulds distances with a fast Rust
core, and visualise convergence interactively — between-run mixing in
tree space, within-run trajectories, RF traces, pseudo-ESS, and
consensus tree summaries.

---

## Install with `uv`

TreeTracer is distributed as a Python package and launched through
[`uv`](https://docs.astral.sh/uv/) — a standalone tool that handles
Python installation, dependency resolution, and virtual environments
in one step. You don't need to set up Python yourself.

### 1. Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Homebrew alternative)
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new shell so `uv` picks up its install path, then `uv --version`
should print a version number.

### 2. Clone and launch

#### macOS / Windows

```bash
git clone https://github.com/beast-dev/treetracer.git
cd treetracer
uv run treetracer
```

#### Linux

`pywebview` needs the system GTK / WebKit libraries plus Python bindings
that compile against them. Install the system headers first:

```bash
# Ubuntu 24.04+
sudo apt install libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1 \
    gir1.2-gtk-3.0 libgirepository-2.0-dev libcairo2-dev \
    pkg-config zenity

# Ubuntu 22.04 (older WebKit ABI)
sudo apt install libwebkit2gtk-4.0-dev gir1.2-webkit2-4.0 \
    gir1.2-gtk-3.0 libgirepository-1.0-dev libcairo2-dev \
    pkg-config zenity
```

Then clone and add the Python bindings into the project venv:

```bash
git clone https://github.com/beast-dev/treetracer.git
cd treetracer
uv add PyGObject pycairo
uv run treetracer
```

**No root / no apt?** Skip the system-headers step and use the Qt
backend instead:

```bash
uv add qtpy pyqt6 pyqt6-webengine
uv run treetracer
```

---

## Updating

```bash
cd treetracer
git pull
uv run treetracer
```

`uv` detects when `uv.lock` has changed and re-syncs automatically.

---

## Removing

```bash
rm -rf treetracer/        # removes the project venv too
uv cache clean            # optional — frees uv's package cache
```

---

## Documentation

For a tutorial on how to use TreeTracer, see the [BEAST X community website](https://beast.community/analysing_beast_output.html).

## TODO

- [ ] Fréchet correlation ESS
- [ ] Standalone packaging

