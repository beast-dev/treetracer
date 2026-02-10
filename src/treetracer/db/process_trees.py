"""Nexus file parser that records byte offsets instead of copying newick strings.

Opens the file in binary mode and tracks the exact byte position of each newick
string. Only the offset and length are passed to the tree manager -- the
newick text is never held in Python memory beyond the current line.
"""

import re
import time
import os
from typing import Tuple, Dict, Any


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


def process_nexus_trees_streaming(nexus_file: str, db_manager, file_source: str,
                                  batch_size: int = 200, transaction_size: int = 1000) -> int:
    """Stream a nexus file, storing newick byte offsets instead of strings.

    Reads the file in binary mode to track precise byte positions. For each
    tree line, records where the newick string starts and how long it is in
    the original file. Only metadata and offsets are sent to the tree manager.

    Args:
        nexus_file: Path to the nexus file
        db_manager: A TreeManagerPandas instance
        file_source: Identifier for this file
        batch_size: Trees per insert batch (default: 200)
        transaction_size: Trees per transaction commit (default: 1000)

    Returns:
        Total number of trees inserted
    """
    print(f"Streaming trees (offset mode) directly to database...")

    # Register the source file so db_manager can read newicks back later
    db_manager.register_source_file(file_source, nexus_file)

    tree_count = 0
    batch_data = []
    total_inserted = 0
    trees_in_current_transaction = 0

    start_time = time.time()
    base_filename = os.path.splitext(os.path.basename(nexus_file))[0]

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
            if not stripped.startswith(b'tree '):
                continue

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
            newick_bytes = stripped[newick_start_in_stripped:].rstrip(b';')
            newick_length = len(newick_bytes)

            leading_ws = len(raw_line) - len(raw_line.lstrip())
            newick_offset = line_start + leading_ws + newick_start_in_stripped

            group_name = base_filename
            metadata_for_db = metadata_dict if metadata_dict else {}

            batch_data.append((
                tree_name, newick_offset, newick_length,
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

                print(f"Inserted batch of {inserted} trees in {batch_time:.2f}s (total: {total_inserted})")

                if trees_in_current_transaction >= transaction_size:
                    commit_start = time.time()
                    db_manager.get_connection().execute("COMMIT")
                    commit_time = time.time() - commit_start
                    print(f"  ✓ Committed transaction ({trees_in_current_transaction} trees in {commit_time:.2f}s)")
                    trees_in_current_transaction = 0
                    db_manager.get_connection().execute("BEGIN TRANSACTION")

    # Insert remaining
    if batch_data:
        batch_start = time.time()
        inserted = db_manager.insert_trees_batch_raw(batch_data)
        batch_time = time.time() - batch_start
        total_inserted += inserted
        trees_in_current_transaction += inserted
        print(f"Final batch of {inserted} trees in {batch_time:.2f}s")

    if trees_in_current_transaction > 0:
        db_manager.get_connection().execute("COMMIT")
        print(f"  ✓ Final commit ({trees_in_current_transaction} trees)")

    total_time = time.time() - start_time
    print(f"Streaming complete: {total_inserted} trees in {total_time:.2f}s")

    return total_inserted
