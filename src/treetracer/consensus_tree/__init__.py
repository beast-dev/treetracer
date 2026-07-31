"""Maximum Clade Credibility (consensus tree) tree, computed from rapidtrees' presence
matrix without round-tripping through dendropy.

The interned-snapshot representation that ``rapidtrees`` already writes
alongside every RF computation gives us, per tree, a uint8 bitvector of
which bipartitions it contains. Standard consensus tree reduces to "from the input
set, pick the tree whose splits have the highest log-product of clade
frequencies (= fraction of input trees containing that split)". With a
presence matrix in hand this is one numpy reduction.

Because the answer is one of the input trees, downstream code can just
emit that tree's existing newick — no new newick has to be synthesized,
which dodges a lot of taxa-alignment fiddling.
"""

from __future__ import annotations

import numpy as np

from .. import state
from ._canonical_remap import (
    _build_canonical_remaps,
    _substitute_newick_labels,
)


def compute_consensus_tree_index(presence_subset: np.ndarray) -> tuple[int, float]:
    """Pick the consensus tree from a presence matrix.

    Args:
        presence_subset: uint8 ``(n_trees, n_splits)`` array. ``[i, j] = 1``
            means tree ``i`` contains split ``j``.

    Returns:
        ``(idx, log_clade_credibility)`` — the row index of the tree with
        the highest log-product of clade frequencies in the subset, and
        that tree's score (sum over its splits of ``log(P(split))``).
    """
    n = presence_subset.shape[0]
    if n == 0:
        raise ValueError("empty selection")
    counts = presence_subset.sum(axis=0)              # (n_splits,) int64
    freq = counts / n                                 # (n_splits,) float64
    # log of zero would NaN — for splits absent from every selected tree we
    # pin freq to 1 so log = 0 contributes nothing (those columns are also
    # zero in the presence matrix, so they wouldn't add anything anyway).
    log_freq = np.log(np.where(counts > 0, freq, 1.0)).astype(np.float32)

    # Per-tree score = ⟨presence_row, log_freq⟩. The previous
    # ``(presence_subset.astype(int64) * log_freq).sum(axis=1)`` was
    # numerically identical but materialised a full (n_trees, n_splits)
    # buffer — at 4 k selected trees × 550 k splits the int64 cast alone
    # blows the working set up to ~18 GB. Stream the computation in
    # row-chunks instead so peak memory is bounded by
    # ``CHUNK * n_splits * 4 B`` (~140 MB at CHUNK=64, n_splits=550 k)
    # regardless of n_trees, and each chunk dispatches to BLAS sgemv for
    # a further ~2-3× speed-up over the explicit multiply + reduce path.
    scores = np.empty(n, dtype=np.float64)
    CHUNK = 64
    for i in range(0, n, CHUNK):
        block = presence_subset[i:i + CHUNK].astype(np.float32)
        scores[i:i + CHUNK] = block @ log_freq

    idx = int(scores.argmax())
    return idx, float(scores[idx])


def _scan_brackets_until_paren(s: str) -> tuple[list[tuple[int, int]], int]:
    """Walk ``s`` from the start, collecting ``(start, end)`` for every
    balanced ``[...]`` block until we hit the first ``(`` outside any
    bracket — that ``(`` is taken as the start of the newick body. Returns
    ``(blocks, paren_idx)``; ``paren_idx == -1`` if no ``(`` is found.
    """
    blocks: list[tuple[int, int]] = []
    paren_idx = -1
    i = 0
    while i < len(s):
        c = s[i]
        if c == "[":
            depth = 1
            start = i
            i += 1
            while i < len(s) and depth > 0:
                if s[i] == "[":
                    depth += 1
                elif s[i] == "]":
                    depth -= 1
                i += 1
            blocks.append((start, i))
        elif c == "(":
            paren_idx = i
            break
        else:
            i += 1
    return blocks, paren_idx


def _inject_tree_annotation(line: str, key: str, value: str) -> str:
    """Append ``key=value`` to a BEAST-style key-value bracket block found
    anywhere between the tree name and the start of the newick body.

    BEAST-1 puts tree-level metadata BEFORE the ``=`` sign
    (``tree STATE_X [&lnP=…,joint=…] = [&R] (…)``), while other dialects
    put it AFTER (``tree STATE_X = [&lnP=…] (…)``). We support both by
    scanning everything left of the first newick ``(``.

    Behaviour:
      * If a ``[&k1=v1,…]`` block exists in that prefix, the new key/value
        is appended before its closing ``]`` (joining the existing block).
      * If only flag-style blocks like ``[&R]`` exist, a fresh ``[&key=value]``
        block is inserted right before the newick body.
      * If there is no ``(`` (malformed input), the line is returned
        unchanged.
    """
    blocks, paren_idx = _scan_brackets_until_paren(line)
    if paren_idx < 0:
        return line

    new_kv = f"{key}={value}"

    # First key-value block (one whose body has '=') wins. ``[&R]`` etc.
    # don't have ``=`` so they're skipped.
    for start, end in blocks:
        block = line[start:end]
        # Strip leading "[&" / "[" and trailing "]" to inspect the body.
        if block.startswith("[&"):
            body = block[2:-1]
        elif block.startswith("["):
            body = block[1:-1]
        else:
            continue
        if "=" in body:
            new_block = block[:-1] + f",{new_kv}]"
            return line[:start] + new_block + line[end:]

    # No existing key-value block — drop in a fresh one right before "(".
    prefix = line[:paren_idx].rstrip()
    suffix = line[paren_idx:]
    sep = "" if not prefix or prefix.endswith(" ") else " "
    return f"{prefix}{sep}[&{new_kv}] {suffix}"


def compute_consensus_tree_for_selection(matched_rows, db_manager, source_distmat):
    """Pick the consensus tree from the user's selection.

    Args:
        matched_rows: DataFrame of selected rows from
            ``db_manager._trees`` (must include columns ``name``,
            ``file_source``, ``line_offset``, ``line_length``).
        db_manager: TreeManager-like object exposing ``get_translate_map``
            and ``_read_newick``.
        source_distmat: distmat name registered via
            ``state.register_distmat``; used to look up the snapshot file
            and the canonical row ordering.

    Returns:
        ``(consensus_tree_row, consensus_tree_newick_line, log_clade_credibility, counts, cols_in_consensus_tree, missing_taxa)``:
            * ``consensus_tree_row`` — the matched DB row for the consensus tree (a pandas
              Series). ``None`` when ``missing_taxa`` is non-empty.
            * ``consensus_tree_newick_line`` — the full ``tree NAME = …;`` line read
              from disk, with integer labels remapped to the canonical
              translate when the consensus tree came from a non-canonical
              source. ``None`` when ``missing_taxa`` is non-empty.
            * ``log_clade_credibility`` — sum of ``log(P(split))`` for
              the chosen tree's splits. ``None`` when ``missing_taxa``
              is non-empty.
            * ``counts`` — ``np.int32`` array of length ``n_bipartitions``,
              the column-sum of the snapshot's presence matrix over
              the selected rows. Cached on the registry entry by
              callers so ``compute_clade_frequencies`` skips the
              row-sum work at compare time. ``None`` when ``missing_taxa``
              is non-empty.
            * ``cols_in_consensus_tree`` — frozenset of presence-matrix column
              indices that appear in the chosen consensus tree itself. These
              are the interned bipartition IDs of the consensus tree's clades and
              feed the Clade Frequency Comparison membership filter.
              ``None`` when ``missing_taxa`` is non-empty.
            * ``missing_taxa`` — set of taxa present in some non-canonical
              source but absent from the canonical translate. Non-empty
              means consensus tree could not be computed; caller should bail and
              surface the names to the user.
    """
    if len(matched_rows) == 0:
        raise ValueError("matched_rows is empty")

    canonical_source = matched_rows["file_source"].iloc[0]
    unique_sources = list(matched_rows["file_source"].unique())
    remaps, missing_taxa = _build_canonical_remaps(
        unique_sources, db_manager.get_translate_map, canonical_source,
    )
    if missing_taxa:
        return None, None, None, None, None, missing_taxa

    # Snapshot covers every tree that went into the distmat, in the order
    # state stored them. We index into it by name.
    snap = np.load(state.get_snapshots_path(source_distmat), allow_pickle=False)
    presence = snap["presence"]                       # (N_total, n_splits) uint8
    full_names = state.get_distmat_names(source_distmat)
    name_to_idx = {n: i for i, n in enumerate(full_names)}

    selected_idx = [name_to_idx[n] for n in matched_rows["name"]]
    presence_sub = presence[selected_idx]             # (n_sel, n_splits)
    consensus_tree_local, log_clade_cred = compute_consensus_tree_index(presence_sub)
    consensus_tree_row = matched_rows.iloc[consensus_tree_local]

    # Column-sum is the per-consensus-tree sufficient statistic for the Clade
    # Frequency Comparison feature — pre-compute here while we already
    # have ``presence_sub`` in scope, and stash on the registry entry.
    counts = presence_sub.sum(axis=0).astype(np.int32)
    cols_in_consensus_tree = frozenset(np.flatnonzero(presence_sub[consensus_tree_local]).tolist())

    line = db_manager._read_newick(
        consensus_tree_row["file_source"],
        int(consensus_tree_row["line_offset"]),
        int(consensus_tree_row["line_length"]),
    )
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    line = _substitute_newick_labels(line, remaps.get(consensus_tree_row["file_source"], {}))
    line = _inject_tree_annotation(line, "lnCladeCred",
                                   format(log_clade_cred, ".4f"))

    return consensus_tree_row, line, log_clade_cred, counts, cols_in_consensus_tree, set()


def extract_log_posterior(consensus_tree_row) -> float | None:
    """Pull a log-posterior scalar for a consensus tree row, or ``None``.

    BEAST commonly writes ``lnP`` (log of joint up to the prior) on
    every tree; MrBayes writes ``posterior`` / ``joint``. This helper
    looks for any of those (priority ``lnP > posterior > joint``) and
    returns the first that parses to a float. Pure log-likelihood
    fields (``lnL`` / ``loglikelihood``) are not consulted — they're
    a different quantity.
    """
    if consensus_tree_row is None:
        return None
    meta = consensus_tree_row.get("metadata") if hasattr(consensus_tree_row, "get") else None
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(meta, dict):
        return None
    for key in ("lnP", "posterior", "joint"):
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                continue
    return None


def assemble_consensus_tree_nexus(matched_rows, db_manager, source_distmat):
    """Compute the consensus tree from *matched_rows* and assemble the full NEXUS
    bytes ready to write to disk or hand to a peartree window.

    The NEXUS layout matches what the original ``Export consensus tree`` callback
    used to write directly:

        <canonical source's preamble — #NEXUS, taxa block, begin trees;,
         Translate { … };>
        <consensus_tree_line, with original tree name + lnCladeCred annotation>
        End;

    Returns ``(nexus_bytes, consensus_tree_row, log_clade_credibility, counts, cols_in_consensus_tree, missing_taxa)``:
        * ``nexus_bytes`` — bytes of the assembled NEXUS file. ``None``
          when ``missing_taxa`` is non-empty.
        * ``consensus_tree_row`` — the matched DB row of the chosen consensus tree
          (Series, contains ``name`` and source columns). ``None``
          when ``missing_taxa`` is non-empty.
        * ``log_clade_credibility`` — score of the chosen tree.
          ``None`` when ``missing_taxa`` is non-empty.
        * ``counts`` — per-bipartition presence column-sum over the
          selection (np.int32). Forwarded from ``compute_consensus_tree_for_selection``
          so callers can stash it on the registry entry. ``None`` when
          ``missing_taxa`` is non-empty.
        * ``cols_in_consensus_tree`` — frozenset of presence-matrix column indices
          that appear in the chosen consensus tree itself. Forwarded from
          ``compute_consensus_tree_for_selection``. ``None`` when ``missing_taxa``
          is non-empty.
        * ``missing_taxa`` — set of taxa missing from the canonical
          translate (caller surfaces this as an export error).
    """
    (consensus_tree_row, consensus_tree_line, log_clade_cred, counts,
     cols_in_consensus_tree, missing_taxa) = compute_consensus_tree_for_selection(
        matched_rows, db_manager, source_distmat,
    )
    if missing_taxa:
        return None, None, None, None, None, missing_taxa

    canonical_source = matched_rows["file_source"].iloc[0]
    canonical_preamble = (
        db_manager._source_preambles.get(canonical_source)
        or b"#NEXUS\n\nbegin trees;\n"
    )
    body = consensus_tree_line if consensus_tree_line.endswith("\n") else consensus_tree_line + "\n"
    nexus_bytes = canonical_preamble + body.encode("utf-8") + b"End;\n"
    return nexus_bytes, consensus_tree_row, log_clade_cred, counts, cols_in_consensus_tree, set()
