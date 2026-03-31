# TreeTracer

Posterior tree space exploration in Bayesian phylogenetics.

TreeTracer is a desktop application for visualizing and analyzing phylogenetic tree distributions from MCMC analyses (BEAST, MrBayes, etc.). It computes pairwise Robinson-Foulds distances, performs MDS/PCoA embedding, and provides interactive scatter plots for exploring between-run convergence and within-run chain trajectories.

## Features

- **Load NEXUS `.trees` files** directly — no preprocessing required
- **RF distance computation** via compiled Rust extension (`rapidtrees`)
- **MDS/PCoA embedding** 
- **Between-run analysis**: 3D scatter + three 2D projections
- **Within-run analysis**: linked 2D scatter with gradient coloring, sliding window animation, and point selection (click/box/lasso)
- **Diagnostics**: log-likelihood traces, RF distance traces with burn-in, KDE density panels
- **Export**: selected trees as NEXUS `.trees` files, plots as PDF
- **Async computation**: RF and MDS run in background processes — UI stays responsive
- **Multiple RF matrices**: compute and store multiple distance matrices, select which to use for MDS
- **Native desktop window**: runs in a native window via pywebview (WebKit on macOS/Linux, Edge on Windows)

---

## Quick Start

### Option 1: Run from source (recommended for development)

#### 1. Install [`uv`](https://docs.astral.sh/uv/)

```bash
# macOS (Homebrew)
brew install uv

# Linux/macOS (curl)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Clone and run

```bash
git clone https://github.com/beast-dev/treetracer.git
cd treetracer
uv run treetracer
```

This opens TreeTracer in a native desktop window. `uv` handles Python installation and all dependencies automatically.

### Option 2: Download pre-built app

Download `TreeTracer.app` (macOS) from the releases page. Double-click to run — no Python or terminal required. First launch takes ~30-60s to set up the environment.

---

## Development Setup

### Reference environment

- **Machine**: MacBook Pro 14" (M4 Pro, 2024)
- **OS**: macOS 15.x (Sequoia)
- **Python**: 3.13 (managed by `uv`)
- **Rust**: 1.91+ (for `rapidtrees` and packaging via PyApp)

### Prerequisites

| Tool | Install | Purpose |
|------|---------|---------|
| `uv` | `brew install uv` | Python package manager, venv, and runner |
| Rust | [rustup.rs](https://rustup.rs/) | Required for `rapidtrees` (RF distances) and packaging |
| Xcode CLI tools | `xcode-select --install` | C compiler for some Python packages (macOS) |

### Getting started

```bash
git clone https://github.com/beast-dev/treetracer.git
cd treetracer

# Install dependencies and run (uv handles everything)
uv run treetracer

# Run Python commands (always use uv run)
uv run python -c "import treetracer; print('OK')"
```

### Project structure

```
src/treetracer/
├── app.py              # Dash app creation, server startup, desktop mode
├── ui.py               # Full UI layout (all components, stores, tabs)
├── state.py            # Server-side storage (RF matrices on disk as uint16 .npy)
├── logger.py           # In-memory logging system
├── plot_utils.py       # Between-run Plotly figure construction
├── __init__.py          # Package entry point (exports main)
├── __main__.py          # Enables `python -m treetracer`
├── assets/             # Static assets (icons, CSS)
├── callbacks/          # All Dash callback logic
│   ├── compute.py      # RF computation, MDS, within-run MDS (async subprocess)
│   ├── treespace.py    # Between-run 3D/2D scatter visualization
│   ├── within_run.py   # Within-run visualization, selection, animation, export
│   ├── diagnostics.py  # LnL traces, RF traces, PDF export
│   ├── sidebar.py      # File loading, downsampling, clear data
│   ├── shell.py        # Navbar, about modal, log panel
│   └── _helpers.py     # Native file dialogs, validation
├── db/                 # Tree data layer
│   ├── tree_manager.py # Pandas DataFrame with byte-offset newick I/O
│   ├── tree_service.py # High-level API for tree operations
│   └── process_trees.py # NEXUS streaming parser
└── rf/                 # Computation layer
    ├── rf.py           # RF distance wrappers (rapidtrees)
    ├── _worker.py      # Subprocess workers (RF + MDS)
    └── mds.py          # Classical PCoA via eigendecomposition
```

### Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `dash` | ≥4.1.0 | Web framework |
| `dash-mantine-components` | 2.x | UI component library (React 18) |
| `rapidtrees` | 0.3.0 | Rust-compiled RF distance computation |
| `plotly` | latest | Interactive charting |
| `pandas` | latest | DataFrame operations |
| `numpy` | latest | Matrix operations |
| `scipy` | latest | KDE for diagnostic traces |
| `kaleido` | latest | PDF export |
| `pywebview` | ≥6.1 | Native desktop window |

---

## Packaging & Distribution

TreeTracer can be packaged as a standalone desktop application (~3 MB) that end users double-click to run. See [docs/packaging.md](docs/packaging.md) for full details.

### Quick build (macOS)

```bash
# Prerequisites: Rust toolchain (cargo), uv

# Step 1: Build the PyApp binary
./packaging/build.sh

# Step 2: Wrap in a macOS .app bundle
./packaging/make_macos_app.sh

# Step 3 (optional): Create a .dmg for distribution
hdiutil create -volname TreeTracer \
  -srcfolder packaging/dist/TreeTracer.app \
  -ov -format UDZO \
  packaging/dist/TreeTracer-0.1.0.dmg
```

**Output:** `packaging/dist/TreeTracer.app` — a native macOS application with the TreeTracer icon, launchable from Finder.

### How it works

The build uses [PyApp](https://github.com/ofek/pyapp)

1. `build.sh` builds a Python wheel and compiles a Rust binary that embeds it
2. `make_macos_app.sh` wraps the binary in a `.app` bundle with icon and `Info.plist`
3. On first launch, the binary downloads Python 3.13 via `uv`, creates an isolated venv, and installs all dependencies
4. Subsequent launches are instant (cached environment)
