"""Nexus file parser that records byte offsets instead of copying newick strings.

Opens the file in binary mode and tracks the exact byte position of each newick
string. Only the offset and length are passed to the tree manager -- the
newick text is never held in Python memory beyond the current line.

Also captures the NEXUS preamble (taxa block, Translate section) and parses
the Translate mapping for later export.
"""

import re
import time
import os
from typing import Tuple, Dict, Any

from ..logger import add_log


def parse_tree_line_metadata(left_part: str) -> Tuple[str, Dict[str, Any]]:
    """Parse tree line to extract name and metadata from square brackets.

    Extracts tree name and parses Beast/MrBayes style annotations from
    square brackets into structured JSON key-value pairs.

    Args:
        left_part: The part before ' = ' in tree line

    Returns:
        Tuple of (tree_name, metadata_dict)
    """
    bracket_pattern = r'\[([^\]]+)\]'
    brackets = re.findall(bracket_pattern, left_part)
    clean_part = re.sub(bracket_pattern, '', left_part).strip()
    parts = clean_part.split()
    tree_name = parts[0] if parts else ""

    metadata_dict = {}
    for bracket_content in brackets:
        if bracket_content.startswith('&'):
            content = bracket_content[1:]
            pairs = content.split(',')
            for pair in pairs:
                pair = pair.strip()
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if '.' in value or 'e' in value.lower():
                            metadata_dict[key] = float(value)
                        else:
                            metadata_dict[key] = int(value)
                    except ValueError:
                        metadata_dict[key] = value
                else:
                    if pair == 'R':
                        metadata_dict['rooted'] = True
                    elif pair == 'U':
                        metadata_dict['rooted'] = False
                    else:
                        metadata_dict[pair] = True
        else:
            metadata_dict['bracket_content'] = bracket_content

    return tree_name, metadata_dict


def _parse_translate_block(preamble_text: str) -> Dict[str, str]:
    """Parse the Translate section from preamble text.

    Extracts the number-to-taxon-name mapping from lines like:
        1 TaxonName,
        2 AnotherTaxon,
        ...
        89 LastTaxon

    Args:
        preamble_text: Decoded preamble string

    Returns:
        Dict mapping number strings to taxon names (e.g. {'1': 'TaxonA', '2': 'TaxonB'})
    """
    translate_map = {}

    # Find the Translate block
    translate_match = re.search(r'Translate\s*\n(.*?)\n\s*;', preamble_text,
                                re.DOTALL | re.IGNORECASE)
    if not translate_match:
        return translate_map

    translate_body = translate_match.group(1)
    for line in translate_body.split('\n'):
        line = line.strip().rstrip(',')
        if not line:
            continue
        # Split on first whitespace: "1 TaxonName"
        parts = line.split(None, 1)
        if len(parts) == 2:
            translate_map[parts[0]] = parts[1]

    return translate_map


def _extract_tip_labels(newick: str) -> set:
    """Extract tip (leaf) labels from a newick string.

    Tips appear after '(' or ',' and before ':', ',', ')', or '['.
    Internal node labels (after ')') are excluded by this pattern.
    """
    return set(re.findall(r'(?<=[(,])\s*([^\s():,\[\]]+)', newick))


def process_nexus_trees_streaming(nexus_file: str, db_manager, file_source: str,
                                  batch_size: int = 200, transaction_size: int = 1000) -> int:
    """Stream a nexus file, storing newick byte offsets instead of strings.

    Reads the file in binary mode to track precise byte positions. For each
    tree line, records where the newick string starts and how long it is in
    the original file. Only metadata and offsets are sent to the tree manager.

    Also captures the preamble (everything before the first tree line) and
    parses the Translate mapping, storing both in the tree manager.

    Args:
        nexus_file: Path to the nexus file
        db_manager: A TreeManagerPandas instance
        file_source: Identifier for this file
        batch_size: Trees per insert batch (default: 200)
        transaction_size: Trees per transaction commit (default: 1000)

    Returns:
        Total number of trees inserted
    """
    add_log("Streaming trees (offset mode) directly to database...")

    # Register the source file so db_manager can read newicks back later
    db_manager.register_source_file(file_source, nexus_file)

    tree_count = 0
    batch_data = []
    total_inserted = 0
    trees_in_current_transaction = 0
    # Per-file rooting accumulator. We inspect each tree's leading
    # ``[&R]`` / ``[&U]`` flag (BEAST/MrBayes/RevBayes convention)
    # and finalise one value per file at the end of the loop:
    #   - Any tree has ``[&R]`` (and none have ``[&U]``)  → rooted
    #   - Any tree has ``[&U]`` (and none have ``[&R]``)  → unrooted
    #   - Mixed ``[&R]`` and ``[&U]`` flags within a file → rooted + warn
    #   - No tree had any flag at all                     → UNROOTED + warn
    #     (NEXUS standard's default; BEAST/MrBayes both write ``[&R]``
    #      explicitly when they mean rooted, so a totally flag-less
    #      file is much more likely RevBayes/empirical-tree-sample
    #      output where the topology is treated as unrooted.)
    file_rooted_counts = {True: 0, False: 0, None: 0}
    preamble_captured = False
    preamble_bytes = b''

    start_time = time.time()
    base_filename = os.path.splitext(file_source)[0]

    db_manager.get_connection().execute("BEGIN TRANSACTION")

    # Binary mode for precise byte offset tracking
    byte_pos = 0
    with open(nexus_file, 'rb') as f:
        while True:
            line_start = byte_pos
            raw_line = f.readline()
            if not raw_line:
                break
            byte_pos += len(raw_line)

            stripped = raw_line.strip()
            if not stripped.lower().startswith(b'tree '):
                # Accumulate preamble lines until first tree line
                if not preamble_captured:
                    preamble_bytes += raw_line
                continue

            # First tree line: capture preamble
            if not preamble_captured:
                preamble_captured = True
                preamble_text = preamble_bytes.decode('utf-8', errors='replace')
                translate_map = _parse_translate_block(preamble_text)
                if hasattr(db_manager, 'set_source_preamble'):
                    db_manager.set_source_preamble(
                        file_source, preamble_bytes, translate_map
                    )
                add_log(f"Captured preamble ({len(preamble_bytes)} bytes, "
                      f"{len(translate_map)} taxa in Translate)")

            eq_pos = stripped.find(b' = ')
            if eq_pos <= 0:
                continue

            # Decode only the small left part for metadata parsing
            left_part = stripped[5:eq_pos].decode('utf-8', errors='replace')
            tree_name, metadata_dict = parse_tree_line_metadata(left_part)
            if not tree_name:
                tree_name = f"tree_{tree_count + 1}"

            # Calculate newick byte offset and length in the original file
            newick_start_in_stripped = eq_pos + 3
            newick_bytes = stripped[newick_start_in_stripped:]
            newick_length = len(newick_bytes)

            # Detect rooting from the newick's leading bytes. BEAST and
            # MrBayes/RevBayes both place ``[&R]`` / ``[&U]`` immediately
            # after the ``=``. The metadata in ``parse_tree_line_metadata``
            # above only catches flags on the LEFT side of ``=`` (rare).
            _peek = newick_bytes.lstrip()
            if _peek.startswith(b'[&R]'):
                tree_rooted = True
            elif _peek.startswith(b'[&U]'):
                tree_rooted = False
            else:
                # Fall back to whatever (if anything) the left-side
                # metadata parser inferred.
                tree_rooted = metadata_dict.get('rooted')
            if tree_rooted is True:
                file_rooted_counts[True] += 1
            elif tree_rooted is False:
                file_rooted_counts[False] += 1
            else:
                file_rooted_counts[None] += 1

            # Fallback: if no Translate block, extract taxa from first tree
            if tree_count == 0 and not translate_map:
                newick_str = newick_bytes.decode('utf-8', errors='replace').rstrip(';')
                tips = _extract_tip_labels(newick_str)
                translate_map = {name: name for name in sorted(tips)}
                if hasattr(db_manager, 'set_source_preamble'):
                    db_manager.set_source_preamble(
                        file_source, preamble_bytes, translate_map
                    )
                add_log(f"No Translate block; extracted {len(translate_map)} taxa from first tree")

            leading_ws = len(raw_line) - len(raw_line.lstrip())
            newick_offset = line_start + leading_ws + newick_start_in_stripped

            # Full tree line offset and length (for verbatim export)
            line_offset = line_start
            line_length = len(raw_line)

            group_name = base_filename
            metadata_for_db = metadata_dict if metadata_dict else {}

            batch_data.append((
                tree_name, newick_offset, newick_length,
                line_offset, line_length,
                file_source, group_name, metadata_for_db
            ))
            tree_count += 1

            if len(batch_data) >= batch_size:
                batch_start = time.time()
                inserted = db_manager.insert_trees_batch_raw(batch_data)
                batch_time = time.time() - batch_start

                total_inserted += inserted
                trees_in_current_transaction += inserted
                batch_data = []

                add_log(f"Inserted batch of {inserted} trees in {batch_time:.2f}s (total: {total_inserted})")

                if trees_in_current_transaction >= transaction_size:
                    commit_start = time.time()
                    db_manager.get_connection().execute("COMMIT")
                    commit_time = time.time() - commit_start
                    add_log(f"Committed transaction ({trees_in_current_transaction} trees in {commit_time:.2f}s)")
                    trees_in_current_transaction = 0
                    db_manager.get_connection().execute("BEGIN TRANSACTION")

    # Insert remaining
    if batch_data:
        batch_start = time.time()
        inserted = db_manager.insert_trees_batch_raw(batch_data)
        batch_time = time.time() - batch_start
        total_inserted += inserted
        trees_in_current_transaction += inserted
        add_log(f"Final batch of {inserted} trees in {batch_time:.2f}s")

    if trees_in_current_transaction > 0:
        db_manager.get_connection().execute("COMMIT")
        add_log(f"Final commit ({trees_in_current_transaction} trees)")

    # Flush pending rows
    if hasattr(db_manager, 'flush'):
        db_manager.flush()

    total_time = time.time() - start_time
    add_log(f"Streaming complete: {total_inserted} trees in {total_time:.2f}s")

    # Finalise the file's rooting convention. See ``file_rooted_counts``
    # initialisation above for the decision table.
    n_rooted = file_rooted_counts[True]
    n_unrooted = file_rooted_counts[False]
    n_unknown = file_rooted_counts[None]
    if n_rooted and n_unrooted:
        # Within-file inconsistency. Pick rooted (BEAST convention)
        # but flag loudly so the user can investigate.
        add_log(
            f"WARNING: {file_source!r} mixes rooted ({n_rooted}) and unrooted "
            f"({n_unrooted}) trees. Treating the file as rooted.", "WARNING",
        )
        file_rooted = True
    elif n_rooted and not n_unrooted:
        # Any explicit [&R] → rooted, even if some trees lack the flag.
        file_rooted = True
    elif n_unrooted and not n_rooted:
        # Any explicit [&U] → unrooted.
        file_rooted = False
    else:
        # No tree had any flag at all. NEXUS standard says "default to
        # unrooted"; BEAST writes [&R] when it means rooted, so a
        # flag-less file is more likely an unrooted RevBayes /
        # empirical-tree-sampling output.
        add_log(
            f"WARNING: {file_source!r} has no [&R] or [&U] flag on any tree. "
            "Treating as UNROOTED per NEXUS default. Add `[&R]` to the tree "
            "lines if these trees are meant to be rooted.", "WARNING",
        )
        file_rooted = False
    if hasattr(db_manager, 'set_source_rooted'):
        db_manager.set_source_rooted(file_source, file_rooted)
    add_log(
        f"Detected rooting for {file_source!r}: "
        f"{'rooted' if file_rooted else 'unrooted'} "
        f"(flags: {n_rooted}R / {n_unrooted}U / {n_unknown} none)"
    )

    return total_inserted
