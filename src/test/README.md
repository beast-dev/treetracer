# TreeTracer CI test suite

Cross-platform validation. Runs against the live `treetracer` package
in this repo plus three external references — `arviz`, `dendropy`, and
`scipy` — to triangulate the numerical bits.

## Run locally

```
# Install test deps once
uv sync --group test

# Full suite (~5 s on a workstation)
bash run_tests.sh

# Pytest passthrough
bash run_tests.sh -k consensus tree -v
```

## Layout

| File | What it covers |
| --- | --- |
| `conftest.py` | Session fixtures: NEXUS parse, rapidtrees presence, DendroPy parse. |
| `test.trees` | 100-tree BEAST fixture (5 MB). The CI integration anchor. |
| `test_app_smoke.py` | App imports, callback registration, figure-layout invariants. |
| `test_ess.py` | `effective_sample_size` vs AR(1) closed form + arviz cross-check (iid, AR(2), MA(5), heavy-tail, multimodal). |
| `test_pseudo_ess.py` | `compute_pseudo_ess` shape, n-cap, rank-norm bound, row-order sensitivity. |
| `test_pcoa.py` | `compute_mds` Procrustes-equivalent to scipy on synthetic Euclidean + real RF. |
| `test_state_registry.py` | distmat index, consensus tree cache, consensus tree registry, LRU eviction, cascade clears. |
| `test_translate_remap.py` | `_build_canonical_remaps` for cross-source consensus tree export. |
| `test_file_io.py` | NEXUS → DB → byte-offset re-read round-trip, translate map round-trip, metadata extraction. |
| `test_rf_matrix.py` | rapidtrees pairwise RF matrix element-wise vs DendroPy rooted-clade sym-diff (50 trees / 1225 pairs). |
| `test_rf_trace.py` | `compute_rf_trace_data` first/last reference, error paths. |
| `test_consensus_tree.py` | `compute_consensus_tree_index` algorithm match + full pipeline (rapidtrees → presence → consensus tree vs DendroPy consensus tree) + runner-up discrimination check. |

## Markers

- `@pytest.mark.integration` — needs the 100-tree fixture; depends on rapidtrees / DendroPy.

## Adding a test

* Reuse the session fixtures in `conftest.py` (`rapidtrees_full`, `rapidtrees_50`, `dendropy_trees_50`, `dendropy_trees_100`, etc.) so the suite stays fast.
* Anchor at least one assertion to an **external reference** (arviz / DendroPy / scipy / closed form) — otherwise the test is a tautology over the implementation.
* If the test parses or compares topologies, remember to set `is_rooted=False` on DendroPy trees before `encode_bipartitions` — see the `dendropy_trees_50` fixture in `conftest.py`. BEAST writes `[&R]` on every tree which DendroPy honours as rooted by default, producing a ghost-split that desyncs from rapidtrees' unrooted RF.
