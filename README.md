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

#### 2. Install system dependencies (Linux only)

TreeTracer uses pywebview for its native desktop window. On Linux, this requires GTK WebKit system packages:

```bash
# Ubuntu 24.04+
sudo apt install libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1 \
    gir1.2-gtk-3.0 libgirepository-2.0-dev libcairo2-dev \
    pkg-config zenity

# Ubuntu 22.04
sudo apt install libwebkit2gtk-4.0-dev gir1.2-webkit2-4.0 \
    gir1.2-gtk-3.0 libgirepository-1.0-dev libcairo2-dev \
    pkg-config zenity
```

#### 3. Clone and run

```bash
git clone https://github.com/beast-dev/treetracer.git
cd treetracer

# macOS — just run directly:
uv run treetracer

# Linux — install PyGObject bindings, then run:
uv add PyGObject pycairo
uv run treetracer
```

This opens TreeTracer in a native desktop window. `uv` handles Python installation and all pip dependencies automatically. On Linux, `PyGObject` and `pycairo` are compiled from source against the system GTK/GObject headers installed via `apt`.

**Alternative (Linux, no root):** If you can't install system packages, use the Qt backend instead:
```bash
uv add qtpy pyqt6 pyqt6-webengine
uv run treetracer
```

### Option 2: Download pre-built app

Download `TreeTracer.app` (macOS) from the releases page. Double-click to run — no Python or terminal required. First launch takes ~30-60s to set up the environment.

---

## Packaging & Distribution

TreeTracer can be packaged as a standalone desktop application (~3 MB) that end users double-click to run.

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

---

## TODO

### Convergence Diagnostics

- [ ] **ESS computation**
- [ ] **Pseudo ESS** 
- [ ] **ASDSF (Average Standard Deviation of Split Frequencies)** 
- [ ] **Frechet correlation ESS** 
