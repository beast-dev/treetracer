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

    The reference tree and the per-tree list are both drawn from
    ``distmat_names`` — the matrix's own self-description — rather
    than from the DB. The DB drifts when the user mutates trees
    between matrices (reset, burnin, downsample, or computing a
    second matrix on a different subset); picking "first/last of
    group X" from a drifted DB used to silently return a tree the
    selected matrix doesn't contain, hence the
    ``Reference tree '...' not found in distance matrix`` failure
    after switching back to an older matrix. Using ``distmat_names``
    makes this a pure function of ``(distmat_key, ref_group,
    ref_position)`` — same matrix → same answer, regardless of what
    happened to the DB since.

    The DB is consulted only for the optional ``file_source``
    column shown in hover; trees that are in the matrix but no
    longer in the DB get a fallback label.

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

    name_to_idx = {n: i for i, n in enumerate(distmat_names)}

    add_log(f"Computing RF trace to {ref_position} tree of group "
            f"'{ref_group}' in '{distmat_key}' (using pre-computed matrix)...")

    # ``process_trees`` stores names as ``"<group>/<tree>"`` (see
    # ``insert_trees_batch_raw``), so we can parse the group back out
    # of each distmat name without touching the DB.
    distmat_groups = [
        n.rsplit("/", 1)[0] if "/" in n else n for n in distmat_names
    ]

    ref_trees_in_group = [
        name for name, grp in zip(distmat_names, distmat_groups)
        if grp == ref_group
    ]
    if not ref_trees_in_group:
        msg = (f"No trees of group '{ref_group}' in distance matrix "
               f"'{distmat_key}'.")
        add_log(msg, "ERROR")
        return msg, None

    ref_name = (ref_trees_in_group[0] if ref_position == "first"
                else ref_trees_in_group[-1])
    add_log(f"Reference tree: '{ref_name}' "
            f"({ref_position} of '{ref_group}' in '{distmat_key}')")
    ref_idx = name_to_idx[ref_name]

    # Optional per-row ``file_source`` for hover. We pull it from the
    # DB when available, but a tree missing from the DB (e.g. dropped
    # by a subsequent reset/downsample) gets a fallback label rather
    # than disappearing from the trace.
    tree_service = get_tree_service()
    tree_service.db_manager.flush()
    all_df = tree_service.db_manager._trees
    if len(all_df) > 0:
        name_to_fs = dict(zip(all_df['name'].tolist(),
                              all_df['file_source'].astype(str).tolist()))
    else:
        name_to_fs = {}

    all_records = []
    for tree_name, group in zip(distmat_names, distmat_groups):
        if tree_name == ref_name:
            continue
        tree_idx = name_to_idx[tree_name]
        all_records.append({
            'rf_distance': int(distmat_matrix[ref_idx, tree_idx]),
            'group':       group,
            'name':        tree_name,
            'file_source': name_to_fs.get(tree_name, "(from distance matrix)"),
        })

    if not all_records:
        return "No trees available for RF trace.", None

    trace_df = pd.DataFrame(all_records)
    trace_df['treenum'] = trace_df.groupby('group').cumcount() + 1
    return trace_df, ref_name
