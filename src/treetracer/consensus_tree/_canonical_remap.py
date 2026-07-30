"""Cross-source label-remapping helpers, shared between ``consensus tree`` and
``callbacks/treespace``.

These used to live at the top of ``callbacks/treespace.py`` and were
imported by ``consensus_tree.py`` — a real circular-import shape (consensus tree imports from
callbacks, callbacks imports from consensus tree) that worked at runtime only
because ``consensus_tree.py``'s imports of these helpers were inside function
bodies. Moving them here breaks the cycle properly, and lets the consensus tree
subprocess worker (``consensus_tree._subprocess_worker``) call them without
dragging in any of the Dash callback machinery.
"""

from __future__ import annotations

import re


# Match an integer immediately following ``(`` or ``,`` in a newick —
# this is what NEXUS translate maps refer to.
_NEWICK_LABEL_RE = re.compile(r'(?<=[(,])(\d+)')

# A BEAST/MrBayes inline annotation block like ``[&lnP=…]`` — we have
# to mask these out of the substitution pass so any digits or commas
# inside them aren't treated as labels.
_METADATA_BLOCK_RE = re.compile(r'\[[^\]]*\]')


def _substitute_newick_labels(newick, mapping):
    """Substitute integer taxon labels in a newick using ``mapping`` (a dict
    of int-label-string → replacement-string). Labels not in the mapping
    pass through unchanged.

    NEXUS metadata blocks ``[...]`` are stashed first so any digits or
    commas inside them aren't treated as labels.
    """
    if not mapping:
        return newick
    blocks = []

    def _stash(match):
        blocks.append(match.group(0))
        return f'\x00{len(blocks) - 1}\x00'

    stripped = _METADATA_BLOCK_RE.sub(_stash, newick)
    transformed = _NEWICK_LABEL_RE.sub(
        lambda m: mapping.get(m.group(1), m.group(1)),
        stripped,
    )
    return re.sub(r'\x00(\d+)\x00',
                  lambda m: blocks[int(m.group(1))],
                  transformed)


def _build_canonical_remaps(file_sources, get_translate_map, canonical_source):
    """Build per-source ``int_label → canonical_int_label`` remaps.

    Returns ``(remaps, missing_taxa)``:
        remaps[source]      = dict (empty for sources whose translate already
                              matches the canonical mapping).
        missing_taxa        = set of taxa names present in some non-canonical
                              source but absent from the canonical translate
                              (caller should surface this as an export error).
    """
    canonical_translate = get_translate_map(canonical_source) or {}
    canonical_taxon_to_int = {taxon: int_label
                              for int_label, taxon in canonical_translate.items()}
    remaps = {}
    missing_taxa = set()
    for source in file_sources:
        if source == canonical_source:
            remaps[source] = {}
            continue
        src_translate = get_translate_map(source) or {}
        remap = {}
        for src_int, taxon in src_translate.items():
            canonical_int = canonical_taxon_to_int.get(taxon)
            if canonical_int is None:
                missing_taxa.add(taxon)
                continue
            if canonical_int != src_int:
                remap[src_int] = canonical_int
        remaps[source] = remap
    return remaps, missing_taxa
