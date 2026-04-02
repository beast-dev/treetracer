"""RF distance trace computation.

Computes the RF distance from every tree to a user-selected reference tree,
using a pre-computed distance matrix. Pure computation — no Dash imports.
"""

import pandas as pd

from ..logger import add_log
from ..state import load_distmat
from ..db.tree_service import get_tree_service


def compute_rf_trace_data(distmat_key, ref_group, ref_position):
    """Compute RF distances from every tree to a reference tree.

    Args:
        distmat_key: Key of the distance matrix in the state index.
        ref_group: Group name of the reference tree.
        ref_position: "first" or "last" tree in the group.

    Returns:
        (trace_df, ref_name) on success, where trace_df has columns:
        rf_distance, group, name, file_source, treenum.

        (error_message, None) on failure.
    """
    # Load matrix from disk
    try:
        distmat_names, distmat_matrix = load_distmat(distmat_key)
    except KeyError:
        return "Distance matrix not available. Please recompute RF distances.", None

    # Build name→index lookup for O(1) distance access
    name_to_idx = {n: i for i, n in enumerate(distmat_names)}

    add_log(f"Computing RF trace to {ref_position} tree of group '{ref_group}' (using pre-computed matrix)...")

    # Build ordered tree list: prefer DB if trees are loaded, otherwise derive from distmat names
    tree_service = get_tree_service()
    tree_service.db_manager.flush()
    all_df = tree_service.db_manager._trees

    if len(all_df) > 0:
        all_df = all_df.sort_values('id')
        tree_names = all_df['name'].tolist()
        tree_groups = all_df['group_name'].tolist()
        tree_file_sources = all_df['file_source'].tolist()
    else:
        tree_names = distmat_names
        tree_groups = [
            name.rsplit("/", 1)[0] if "/" in name else name
            for name in tree_names
        ]
        tree_file_sources = ["(from distance matrix)"] * len(tree_names)

    # Filter to reference group and pick first/last
    ref_trees_in_group = [
        name for name, grp in zip(tree_names, tree_groups) if grp == ref_group
    ]

    if not ref_trees_in_group:
        msg = f"No trees found in group '{ref_group}'."
        add_log(msg, "ERROR")
        return msg, None

    ref_name = ref_trees_in_group[0] if ref_position == "first" else ref_trees_in_group[-1]
    add_log(f"Reference tree: name='{ref_name}' ({ref_position} of group '{ref_group}')")

    if ref_name not in name_to_idx:
        msg = f"Reference tree '{ref_name}' not found in distance matrix."
        add_log(msg, "ERROR")
        return msg, None

    ref_idx = name_to_idx[ref_name]

    # Look up RF distance for every tree from the pre-computed matrix
    all_records = []
    for tree_name, group, file_source in zip(tree_names, tree_groups, tree_file_sources):
        if tree_name == ref_name:
            continue
        if tree_name not in name_to_idx:
            add_log(f"Tree '{tree_name}' not found in distance matrix, skipping.", "WARNING")
            continue
        tree_idx = name_to_idx[tree_name]
        all_records.append({
            'rf_distance': int(distmat_matrix[ref_idx, tree_idx]),
            'group': group,
            'name': tree_name,
            'file_source': file_source,
        })

    if not all_records:
        return "No trees available for RF trace.", None

    trace_df = pd.DataFrame(all_records)
    trace_df['treenum'] = trace_df.groupby('group').cumcount() + 1

    return trace_df, ref_name
