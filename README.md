# treetracer

Posterior tree space exploration in Bayesian phylogenetics

`treetracer` is tool to visualize phylogenetic tree space. The tool currently loads TSV files generated in R with MDS coordinates obtained from a distance matrix using some tree metric like Robinson-Foulds or SPR.

## Recommended installation and usage (Linux/macOS)

### 1. Install the [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) using one of these suggested options

Using `curl`
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using `wget`
```
wget -qO- https://astral.sh/uv/install.sh | sh
```

Or using Homebrew:

```
brew install uv
```

### 2. Clone the repo

```
git clone https://github.com/beast-dev/treetracer.git
cd treetracer
```

### 3. Run treetracer

```
uv run treetracer
```

## TODO

Desired workflow:

Load in trees file ----(Stage 1)---> Compute pairwise distances ----(Stage 2)---> Compute MDS ----(Done)---> Plot MDS


(Stage 1): Pairwise distances:
- [x] Load .trees files instead of MDS files and have TreeTracer compute the distance matrix in the app
	- [x] Set up a local temporary database to store the trees file? (duckdb)
	- [x] Set up the nexus parsing and storage
	- [x] Find fast RF distances compute packages?
		- [x] Using Joon's rust package

(Stage 2): Multidimensional scaling:
- [x] Load distance matrix instead of MDS files and have TreeTracer compute the MDS file in the app.
	- [x] Have MDS upload functionality
	- [x] Have MDS computation functionality
	- [x] Feed MDS results to plotting functionality
	- [x] Correctly format MDS results to reflect groups defined by pairwise distance matrix produced in Stage 1. (Assumes group names stored before underscore in tree names)

