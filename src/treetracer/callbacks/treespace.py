import re

from dash import html, callback, clientside_callback, Input, Output, Patch, State, no_update
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from ..logger import add_log
from ..state import get_mds_result
from ..plot_utils import (
    make_plot_grid, add_trace_multiplot_interleaved, placeholder_fig,
)


# Matches an integer taxon label that sits at a label position in newick —
# right after `(` or `,`. Branch lengths come after `:` and aren't matched.
_NEWICK_LABEL_RE = re.compile(r'(?<=[(,])(\d+)')

# Matches the `tree NAME` token at the start of a NEXUS tree line.
_TREE_NAME_RE = re.compile(r'^(\s*tree\s+)([^\s=]+)', re.IGNORECASE)

# Used to stash NEXUS metadata blocks (e.g. ``[&rate=0.05]``) before
# integer-label substitution so commas / digits inside them don't trip the
# label regex. The placeholders use a NUL marker that won't appear in real
# NEXUS content.
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


def _sanitize_tree_name_token(text):
    """Make a string safe to use as an unquoted NEXUS tree name."""
    return re.sub(r'[^A-Za-z0-9_.]', '_', text)


def _rewrite_tree_line(line, new_name, label_remap):
    """Take a raw NEXUS ``tree <name> = <newick>;`` line and emit it with
    the name replaced and any integer labels remapped via ``label_remap``
    (source int → canonical int). Pass an empty dict to leave labels alone.
    """
    renamed = _TREE_NAME_RE.sub(
        lambda m: f'{m.group(1)}{new_name}',
        line,
        count=1,
    )
    return _substitute_newick_labels(renamed, label_remap)


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


# Trace-layout invariants set by ``add_trace_multiplot_interleaved``:
# the last 4 traces are always selection overlays in this exact order.
SELECTION_OVERLAY_OFFSETS = (-4, -3, -2, -1)
N_SELECTION_OVERLAYS = 4


def _panels_2d(x, y, z):
    """Return [(xcol, ycol), ...] for the 3 right-column 2D panels in order:
    (x,y), (x,z), (y,z). Same order add_trace_multiplot_interleaved uses,
    which is what makes the trailing overlay offsets meaningful."""
    return [(x, y), (x, z), (y, z)]


def _overlay_panels_data(df, selected_pairs, x, y, z):
    """Build (xs, ys, customdata) for each of the 4 selection overlays.

    Returns a 4-tuple in trace order:
        index 0 → 3D overlay  (x, y, z arrays + customdata)
        index 1 → 2D x-y overlay  (x, y arrays + customdata)
        index 2 → 2D x-z overlay
        index 3 → 2D y-z overlay

    When nothing is selected the arrays are empty so the trace stays present
    but draws nothing.
    """
    empty_3d = {"x": [], "y": [], "z": [], "customdata": []}
    empty_2d = {"x": [], "y": [], "customdata": []}
    if not selected_pairs:
        return [dict(empty_3d), dict(empty_2d), dict(empty_2d), dict(empty_2d)]

    selected_set = {(g, int(t)) for g, t in selected_pairs}
    keys = list(zip(df["group"], df["treenum"].astype(int)))
    mask = pd.Series([k in selected_set for k in keys], index=df.index)
    sel = df[mask]
    if len(sel) == 0:
        return [dict(empty_3d), dict(empty_2d), dict(empty_2d), dict(empty_2d)]

    labels = sel["tree"].str.split("/").str[-1].str.strip().tolist()
    treenums = sel["treenum"].astype(int).tolist()
    groups = sel["group"].tolist()
    cd = list(zip(treenums, labels, groups))

    return [
        {"x": sel[x].tolist(), "y": sel[y].tolist(),
         "z": sel[z].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[y].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[z].tolist(), "customdata": cd},
        {"x": sel[y].tolist(), "y": sel[z].tolist(), "customdata": cd},
    ]


def register_treespace_callbacks():
    # Populate the MDS-result selector dropdown
    @callback(
        Output("treespace-result-select", "data"),
        Output("treespace-result-select", "value"),
        Input("mds-result-store", "data"),
        State("treespace-result-select", "value"),
    )
    def populate_result_selector(results, current_value):
        if not results:
            return [], None
        options = [
            {"value": k,
             "label": f"{v.get('filename', k)} ({v['rows']} trees, {len(v.get('groups', []))} runs)"}
            for k, v in results.items()
        ]
        if current_value and current_value in results:
            return options, current_value
        return options, list(results.keys())[-1]

    # When a result is picked, configure controls and reset the plot canvas.
    @callback(
        Output("plot-config-store", "data", allow_duplicate=True),
        Output("dim-x-select", "data"),
        Output("dim-x-select", "value"),
        Output("dim-y-select", "data"),
        Output("dim-y-select", "value"),
        Output("dim-z-select", "data"),
        Output("dim-z-select", "value"),
        Output("treenum-slider", "min"),
        Output("treenum-slider", "max"),
        Output("treenum-slider", "value"),
        Output("treenum-slider", "marks"),
        Output("treespace-info", "children"),
        Output("treespace-controls-paper", "style"),
        Output("graph", "figure", allow_duplicate=True),
        Output("plot-button", "children", allow_duplicate=True),
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Output("treespace-export-trees", "disabled", allow_duplicate=True),
        Output("treespace-view-mcc", "disabled", allow_duplicate=True),
        Output("treespace-selection-info", "children", allow_duplicate=True),
        Input("treespace-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def configure_for_selected_result(selected_key, results):
        empty_state = (
            {},
            [], None,
            [], None,
            [], None,
            1, 100, [1, 100], [],
            html.Div(),
            {"display": "none"},
            placeholder_fig("No MDS result selected. Compute an MDS in the Compute tab."),
            "Plot",
            [],
            True,
            True,
            html.Div(),
        )
        if not selected_key or not results or selected_key not in results:
            return empty_state

        mds_result = get_mds_result(selected_key)
        if not mds_result or not mds_result.get("data"):
            return empty_state

        metadata = mds_result["metadata"]
        combined_df = pd.DataFrame(mds_result["data"])
        mdscols = metadata["dimensions"]
        groups = combined_df["group"].unique().tolist()
        group_colors = px.colors.qualitative.Dark24[:len(groups)]
        color_dict = {g: c for g, c in zip(groups, group_colors)}
        MIN_TREENUM = metadata["MIN_TREENUM"]
        MAX_TREENUM = metadata["MAX_TREENUM"]

        plot_config = {
            "combined_data": mds_result["data"],
            "mdscols": mdscols,
            "min_treenum": MIN_TREENUM,
            "max_treenum": MAX_TREENUM,
            "groups": groups,
            "color_dict": color_dict,
        }

        dim_options = [{"value": col, "label": col} for col in mdscols]
        z_default = mdscols[2] if len(mdscols) > 2 else mdscols[0]

        marks = [
            {"value": MIN_TREENUM, "label": str(MIN_TREENUM)},
            {"value": MAX_TREENUM, "label": str(MAX_TREENUM)},
        ]

        info = dmc.Group([
            dmc.Badge(f"Trees: {len(combined_df)}",
                      variant="light", color="grape", size="sm"),
            dmc.Badge(f"Runs: {len(groups)}",
                      variant="light", color="teal", size="sm"),
        ], gap="xs")

        return (
            plot_config,
            dim_options, mdscols[0],
            dim_options, mdscols[1],
            dim_options, z_default,
            MIN_TREENUM, MAX_TREENUM, [MIN_TREENUM, MAX_TREENUM], marks,
            info,
            {"display": "flex"},
            placeholder_fig("Click 'Plot' to visualize data"),
            "Plot",
            [],
            True,
            True,
            html.Div(),
        )

    # Plot button — explicit user trigger so dropdown / slider changes don't
    # auto-rebuild the (heavy) multiplot.
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Output("plot-button", "children", allow_duplicate=True),
        Input("plot-button", "n_clicks"),
        State("dim-x-select", "value"),
        State("dim-y-select", "value"),
        State("dim-z-select", "value"),
        State("treenum-slider", "value"),
        State("show-lines-checkbox", "checked"),
        State("graph", "figure"),
        State("plot-config-store", "data"),
        State("treespace-dragmode", "value"),
        State("treespace-selected-trees-store", "data"),
        prevent_initial_call=True,
    )
    def update_graph_on_button_click(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                                     show_lines, current_fig, plot_config, dragmode,
                                     selected):
        if not n_clicks or not plot_config or not all([dim_x, dim_y, dim_z]):
            return no_update, no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])

        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0])
            & (combined_df["treenum"] <= treenum_range[1])
        ]
        mds_selected = [dim_x, dim_y, dim_z]
        add_log(f"Plotting {len(filtered_dff)} trees (range {treenum_range[0]}-{treenum_range[1]}), dims: {mds_selected}")

        # Create new plot with filtered data. The interleaved variant splits
        # each group's points into chunks and stacks them by chunk-index so
        # no single run sits entirely on top of the others in the 2D panels.
        # It also appends 4 trailing selection-overlay traces (1 3D + 3 2D).
        fig = make_plot_grid()
        add_trace_multiplot_interleaved(
            fig, filtered_dff, dim_x, dim_y, dim_z,
            plot_config["groups"], plot_config["color_dict"],
            show_lines=show_lines,
        )
        fig.update_layout(dragmode=dragmode or "zoom")

        # Try to preserve visibility settings if updating an existing plot.
        # Skip when the count differs (after dim/range/show-lines toggles the
        # number of chunk traces can change; a mismatch means we just emit a
        # fresh figure).
        if current_fig and "data" in current_fig:
            old_data = current_fig["data"]
            if len(old_data) == len(fig.data):
                for i in range(len(fig.data)):
                    old_vis = old_data[i].get("visible") if isinstance(old_data[i], dict) else None
                    if old_vis is not None:
                        fig.data[i].visible = old_vis

        # Re-apply current selection so the red overlays survive a rebuild.
        # Filter against the FILTERED dataframe so selected trees outside the
        # treenum-range stay invisible (matching how their normal markers
        # would have been hidden too).
        if selected:
            sel_data = _overlay_panels_data(filtered_dff, selected, dim_x, dim_y, dim_z)
            d3, dxy, dxz, dyz = sel_data
            n = len(fig.data)
            fig.data[n - 4].x = d3["x"]
            fig.data[n - 4].y = d3["y"]
            fig.data[n - 4].z = d3["z"]
            fig.data[n - 4].customdata = d3["customdata"]
            for offset, d in zip((-3, -2, -1), (dxy, dxz, dyz)):
                idx = n + offset
                fig.data[idx].x = d["x"]
                fig.data[idx].y = d["y"]
                fig.data[idx].customdata = d["customdata"]

        return fig, "Update Plot"

    # ------ click / box / lasso → selection store ------
    @callback(
        Output("treespace-selected-trees-store", "data"),
        Input("graph", "clickData"),
        State("treespace-selected-trees-store", "data"),
        prevent_initial_call=True,
    )
    def handle_click_select(click_data, current_selection):
        if not click_data:
            return no_update
        point = click_data["points"][0]
        cd = point.get("customdata")
        if not cd or len(cd) < 3:
            return no_update
        try:
            treenum = int(cd[0])
            group = cd[2]
        except (ValueError, TypeError, IndexError):
            return no_update

        selected = {(g, int(t)) for g, t in (current_selection or [])}
        key = (group, treenum)
        if key in selected:
            selected.discard(key)
        else:
            selected.add(key)
        return sorted(([g, t] for g, t in selected), key=lambda p: (str(p[0]), p[1]))

    @callback(
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Input("graph", "selectedData"),
        State("treespace-selected-trees-store", "data"),
        prevent_initial_call=True,
    )
    def handle_region_select(selected_data, current_selection):
        if not selected_data or not selected_data.get("points"):
            return no_update
        new_pairs = set()
        for point in selected_data["points"]:
            cd = point.get("customdata")
            if not cd or len(cd) < 3:
                continue
            try:
                new_pairs.add((cd[2], int(cd[0])))
            except (ValueError, TypeError, IndexError):
                continue
        if not new_pairs:
            return no_update
        selected = {(g, int(t)) for g, t in (current_selection or [])}
        selected |= new_pairs
        return sorted(([g, t] for g, t in selected), key=lambda p: (str(p[0]), p[1]))

    @callback(
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Input("treespace-clear-selection", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_selection(n_clicks):
        if not n_clicks:
            return no_update
        return []

    # ------ dragmode → patch layout.dragmode (preserves zoom) ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("treespace-dragmode", "value"),
        prevent_initial_call=True,
    )
    def update_dragmode(dragmode):
        if not dragmode:
            return no_update
        patch = Patch()
        patch["layout"]["dragmode"] = dragmode
        return patch

    # ------ selection-info badge + Export-trees / Export-MCC enable ------
    @callback(
        Output("treespace-selection-info", "children"),
        Output("treespace-export-trees", "disabled"),
        Output("treespace-view-mcc", "disabled"),
        Input("treespace-selected-trees-store", "data"),
    )
    def update_selection_info(selected):
        if not selected:
            return html.Div(), True, True
        return (
            dmc.Badge(f"Selected: {len(selected)} trees",
                      color="red", variant="light", size="lg"),
            False,
            False,
        )

    # ------ selection store change → patch only the last 4 traces ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("treespace-selected-trees-store", "data"),
        State("graph", "figure"),
        State("dim-x-select", "value"),
        State("dim-y-select", "value"),
        State("dim-z-select", "value"),
        State("plot-config-store", "data"),
        prevent_initial_call=True,
    )
    def update_selection_overlay(selected, current_fig,
                                 dim_x, dim_y, dim_z, plot_config):
        if not current_fig or not plot_config:
            return no_update
        n_traces = len(current_fig.get("data", []))
        # We need at least the 4 overlays plus some real traces. Bail if the
        # graph hasn't been plotted yet or the trace count doesn't have room
        # for the 4 trailing overlays.
        if n_traces < N_SELECTION_OVERLAYS:
            return no_update
        if not all([dim_x, dim_y, dim_z]):
            return no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])
        sel_data = _overlay_panels_data(combined_df, selected, dim_x, dim_y, dim_z)

        patch = Patch()
        # Drop the dashed box/lasso rectangle Plotly persists in
        # layout.selections after a box-select.
        patch["layout"]["selections"] = []
        # Reset selectedpoints on every 2D trace so Plotly's auto-fade on
        # out-of-box markers doesn't mute points after box-select. Scatter3d
        # doesn't support selectedpoints — skipping the 3D traces avoids a
        # ValueError when go.Figure later reconstructs the dict (e.g. PDF
        # export).
        traces = current_fig.get("data", [])
        for i in range(n_traces):
            if traces[i].get("type") == "scatter3d":
                continue
            patch["data"][i]["selectedpoints"] = None

        # Last 4 traces in order: 3D, 2D x-y, 2D x-z, 2D y-z.
        d3, dxy, dxz, dyz = sel_data
        patch["data"][n_traces - 4]["x"] = d3["x"]
        patch["data"][n_traces - 4]["y"] = d3["y"]
        patch["data"][n_traces - 4]["z"] = d3["z"]
        patch["data"][n_traces - 4]["customdata"] = d3["customdata"]
        for offset, d in zip((-3, -2, -1), (dxy, dxz, dyz)):
            idx = n_traces + offset
            patch["data"][idx]["x"] = d["x"]
            patch["data"][idx]["y"] = d["y"]
            patch["data"][idx]["customdata"] = d["customdata"]
        return patch

    # ------ reset zoom button ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("treespace-reset-button", "n_clicks"),
        State("dim-x-select", "value"),
        State("dim-y-select", "value"),
        State("dim-z-select", "value"),
        State("treenum-slider", "value"),
        State("show-lines-checkbox", "checked"),
        State("plot-config-store", "data"),
        State("treespace-selected-trees-store", "data"),
        State("treespace-dragmode", "value"),
        prevent_initial_call=True,
    )
    def reset_axes(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                   show_lines, plot_config, selected, dragmode):
        if not n_clicks or not plot_config or not all([dim_x, dim_y, dim_z]):
            return no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])
        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0])
            & (combined_df["treenum"] <= treenum_range[1])
        ]

        fig = make_plot_grid()
        add_trace_multiplot_interleaved(
            fig, filtered_dff, dim_x, dim_y, dim_z,
            plot_config["groups"], plot_config["color_dict"],
            show_lines=show_lines,
        )
        fig.update_layout(dragmode=dragmode or "zoom")
        # Force a fresh uirevision so reset *does* throw away the user's zoom.
        fig.update_layout(uirevision=f"reset-{n_clicks}")

        # Re-apply current selection so the overlay survives the reset.
        if selected:
            sel_data = _overlay_panels_data(filtered_dff, selected, dim_x, dim_y, dim_z)
            d3, dxy, dxz, dyz = sel_data
            n = len(fig.data)
            fig.data[n - 4].x = d3["x"]
            fig.data[n - 4].y = d3["y"]
            fig.data[n - 4].z = d3["z"]
            fig.data[n - 4].customdata = d3["customdata"]
            for offset, d in zip((-3, -2, -1), (dxy, dxz, dyz)):
                idx = n + offset
                fig.data[idx].x = d["x"]
                fig.data[idx].y = d["y"]
                fig.data[idx].customdata = d["customdata"]
        return fig

    # ------ export selected trees as a .trees NEXUS file ------
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("treespace-export-trees", "n_clicks"),
        State("treespace-selected-trees-store", "data"),
        State("plot-config-store", "data"),
        prevent_initial_call=True,
    )
    def export_selected_trees(n_clicks, selected_pairs, plot_config):
        from ..logger import notif_id
        from ..db.tree_service import get_tree_service
        from ._helpers import _save_file_dialog
        if not n_clicks or not selected_pairs or not plot_config:
            return no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])
        selected_set = {(g, int(t)) for g, t in selected_pairs}
        keys = list(zip(combined_df["group"], combined_df["treenum"].astype(int)))
        mask = pd.Series([k in selected_set for k in keys], index=combined_df.index)
        sel_df = combined_df[mask]
        tree_names = sel_df["tree"].tolist()
        if not tree_names:
            return dmc.Notification(title="Export Error",
                                    message="No matching trees found.",
                                    color="red", action="show",
                                    autoClose=4000, id=notif_id())

        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_trees = tree_service.db_manager._trees
        matched = all_trees[all_trees["name"].isin(tree_names)].sort_values("id")
        if len(matched) == 0:
            return dmc.Notification(
                title="Export Error",
                message="Selected trees not found in database. They may have been cleared.",
                color="red", action="show", autoClose=4000, id=notif_id())

        # Selected trees can come from multiple source files with different
        # translate tables. We pick the FIRST matched row's source as
        # canonical, write its preamble verbatim (which already includes the
        # `Translate` block), and remap the integer taxon labels in every
        # non-canonical tree's newick so they line up with the canonical
        # numbering. Tree names get prefixed with the group identifier so
        # cross-run STATE_X collisions are visible in the output.
        canonical_source = matched["file_source"].iloc[0]
        canonical_preamble = tree_service.db_manager._source_preambles.get(canonical_source)
        unique_sources = list(matched["file_source"].unique())
        remaps, missing_taxa = _build_canonical_remaps(
            unique_sources,
            tree_service.db_manager.get_translate_map,
            canonical_source,
        )
        if missing_taxa:
            sample = ", ".join(sorted(missing_taxa)[:5])
            more = "…" if len(missing_taxa) > 5 else ""
            return dmc.Notification(
                title="Export Error",
                message=(
                    f"Cannot align translate tables: taxa [{sample}{more}] "
                    f"are present in some selected runs but not in "
                    f"{canonical_source}'s Translate block. "
                    "Either deselect those runs or pick selections from runs "
                    "with matching taxa."
                ),
                color="red", action="show", autoClose=8000, id=notif_id())

        path = _save_file_dialog(default_filename=f"selected_{len(matched)}_trees.trees")
        if not path:
            return no_update

        try:
            with open(path, "wb") as out:
                if canonical_preamble:
                    out.write(canonical_preamble)
                else:
                    out.write(b"#NEXUS\n\nbegin trees;\n")
                for _, row in matched.iterrows():
                    file_source = row["file_source"]
                    line = tree_service.db_manager._read_newick(
                        file_source,
                        int(row["line_offset"]),
                        int(row["line_length"]),
                    )
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    db_name = row["name"]                      # "<group>/<orig>"
                    group_name = row["group_name"]
                    original_name = (
                        db_name.split("/", 1)[1]
                        if "/" in db_name else db_name
                    )
                    new_name = _sanitize_tree_name_token(
                        f"{group_name}_{original_name}"
                    )
                    rewritten = _rewrite_tree_line(
                        line, new_name, remaps.get(file_source, {}),
                    )
                    out.write(rewritten.encode("utf-8"))
                    if not rewritten.endswith("\n"):
                        out.write(b"\n")
                out.write(b"End;\n")
        except Exception as e:
            return dmc.Notification(title="Export Error", message=str(e),
                                    color="red", action="show",
                                    autoClose=6000, id=notif_id())

        add_log(f"Exported {len(matched)} selected trees to {path}")
        return dmc.Notification(title="Trees Exported",
                                message=f"Exported {len(matched)} trees to {path}",
                                color="green", action="show",
                                autoClose=4000, id=notif_id())

    # ------ View MCC tree in a peartree window ------
    # Same selection-filter plumbing as ``export_selected_trees``, but
    # instead of writing a NEXUS file we hand the assembled bytes to
    # ``state.cache_mcc_tree`` and emit ``{"uuid", "name"}`` into the
    # view-mcc store. A clientside callback below picks up that store and
    # opens ``/peartree/<uuid>`` in a new browser window.
    @callback(
        Output("treespace-view-mcc-store", "data"),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("treespace-view-mcc", "n_clicks"),
        State("treespace-selected-trees-store", "data"),
        State("plot-config-store", "data"),
        State("treespace-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def view_mcc_tree(n_clicks, selected_pairs, plot_config,
                      selected_key, results):
        from ..logger import notif_id
        from ..db.tree_service import get_tree_service
        from ..mcc import assemble_mcc_nexus
        from .. import state as _state
        if not n_clicks or not selected_pairs or not plot_config:
            return no_update, no_update

        results = results or {}
        if not selected_key or selected_key not in results:
            return no_update, dmc.Notification(
                title="MCC Error",
                message="No MDS result is currently selected.",
                color="red", action="show", autoClose=5000, id=notif_id())
        source_distmat = (results[selected_key] or {}).get("source_distmat")
        if not source_distmat:
            return no_update, dmc.Notification(
                title="MCC Error",
                message="No RF/snapshot data is associated with this MDS result.",
                color="red", action="show", autoClose=5000, id=notif_id())

        combined_df = pd.DataFrame(plot_config["combined_data"])
        selected_set = {(g, int(t)) for g, t in selected_pairs}
        keys = list(zip(combined_df["group"], combined_df["treenum"].astype(int)))
        mask = pd.Series([k in selected_set for k in keys], index=combined_df.index)
        sel_df = combined_df[mask]
        tree_names = sel_df["tree"].tolist()
        if not tree_names:
            return no_update, dmc.Notification(
                title="MCC Error",
                message="No matching trees found.",
                color="red", action="show", autoClose=4000, id=notif_id())

        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_trees = tree_service.db_manager._trees
        matched = all_trees[all_trees["name"].isin(tree_names)].sort_values("id")
        if len(matched) == 0:
            return no_update, dmc.Notification(
                title="MCC Error",
                message="Selected trees not found in database. They may have been cleared.",
                color="red", action="show", autoClose=4000, id=notif_id())

        try:
            nexus_bytes, mcc_name, missing_taxa = assemble_mcc_nexus(
                matched, tree_service.db_manager, source_distmat,
            )
        except Exception as e:
            return no_update, dmc.Notification(
                title="MCC Error", message=str(e),
                color="red", action="show", autoClose=6000, id=notif_id())

        if missing_taxa:
            sample = ", ".join(sorted(missing_taxa)[:5])
            more = "…" if len(missing_taxa) > 5 else ""
            canonical_source = matched["file_source"].iloc[0]
            return no_update, dmc.Notification(
                title="MCC Error",
                message=(
                    f"Cannot align translate tables: taxa [{sample}{more}] "
                    f"are present in some selected runs but not in "
                    f"{canonical_source}'s Translate block."
                ),
                color="red", action="show", autoClose=8000, id=notif_id())

        uid = _state.cache_mcc_tree(nexus_bytes)
        add_log(f"Cached MCC tree '{mcc_name}' (from {len(matched)} selected) as {uid}")
        notification = dmc.Notification(
            title="MCC Tree Ready",
            message=f"MCC tree is '{mcc_name}' (from {len(matched)} selected) — opening in PearTree…",
            color="green", action="show", autoClose=4000, id=notif_id())
        return {"uuid": uid, "name": mcc_name}, notification

    # Clientside: when the view-mcc store changes, open the peartree
    # viewer. In desktop pywebview mode we call the Python-side JS API
    # (``window.pywebview.api.open_peartree``) which spawns a sibling
    # native window — keeps the experience inside the desktop app and
    # leaves both windows same-origin so future postMessage between them
    # is unblocked. In ``--browser`` mode (or any context without
    # pywebview), fall back to a normal ``window.open`` that opens a new
    # browser tab.
    clientside_callback(
        """
        function(payload) {
            if (payload && payload.uuid) {
                const name = payload.name || '';
                if (window.pywebview && window.pywebview.api
                    && window.pywebview.api.open_peartree) {
                    window.pywebview.api.open_peartree(payload.uuid, name);
                } else {
                    const url = '/peartree/' + payload.uuid
                              + '?name=' + encodeURIComponent(name);
                    const features = 'width=1200,height=800,resizable=yes,scrollbars=yes';
                    window.open(url, 'peartree-' + payload.uuid, features);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("treespace-view-mcc-store", "data", allow_duplicate=True),
        Input("treespace-view-mcc-store", "data"),
        prevent_initial_call=True,
    )

    # ------ export PDF ------
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("treespace-export-pdf", "n_clicks"),
        State("graph", "figure"),
        prevent_initial_call=True,
    )
    def export_pdf(n_clicks, fig_dict):
        from ..logger import notif_id
        from ._helpers import _save_file_dialog
        if not n_clicks or not fig_dict:
            return no_update
        path = _save_file_dialog(default_filename="treespace.pdf")
        if not path:
            return no_update
        # skip_invalid=True drops any browser-only properties that Plotly's
        # Python validator rejects (e.g. selectedpoints accidentally left on
        # a 3D trace by an old Patch).
        fig = go.Figure(fig_dict, skip_invalid=True)
        fig.update_layout(template="simple_white")
        fig.write_image(path, width=1800, height=1200, scale=2)
        add_log(f"Exported between-run plot to {path}")
        return dmc.Notification(title="PDF Exported",
                                message=f"Saved to {path}",
                                color="green", action="show",
                                autoClose=3000, id=notif_id())
