from dash import html, callback, Input, Output, State, no_update, ALL
import dash_mantine_components as dmc
import os
import pandas as pd

from ..logger import add_log
from ..db.tree_service import get_tree_service
from ._helpers import _save_file_dialog, _open_tsv_dialog, _validate_group_names


def register_compute_callbacks():
    # Callback to render the Compute tab RF table
    @callback(
        Output("compute-trees-table", "children"),
        Output("compute-rf-button", "disabled"),
        Input("tree-offset-store", "data"),
    )
    def render_compute_table(stored_summaries):
        if not stored_summaries:
            return html.Div(
                dmc.Text("No .trees files loaded yet.", c="dimmed"),
                style={"padding": "20px"},
            ), True

        rows = []
        for filename, summary in stored_summaries.items():
            rows.append(
                dmc.TableTr([
                    dmc.TableTd(filename),
                    dmc.TableTd(str(summary.get("n_taxa", "—"))),
                    dmc.TableTd(str(summary.get("total_trees", 0))),
                    dmc.TableTd(
                        dmc.Checkbox(
                            id={"type": "compute-tree-checkbox", "index": filename},
                            checked=True,
                        )
                    ),
                ])
            )

        table = dmc.Table(
            [
                dmc.TableThead(
                    dmc.TableTr([
                        dmc.TableTh("File"),
                        dmc.TableTh("Taxa"),
                        dmc.TableTh("Trees"),
                        dmc.TableTh("Select"),
                    ])
                ),
                dmc.TableTbody(rows),
            ],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
        )

        return table, False

    # Callback to validate taxa and trigger RF computation
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-rf-output", "children"),
        Output("distmat-store", "data", allow_duplicate=True),
        Output("export-rf-button", "disabled"),
        Output("compute-rf-trace-button", "disabled", allow_duplicate=True),
        Input("compute-rf-button", "n_clicks"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "checked"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "id"),
        State("tree-offset-store", "data"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_rf(n_clicks, checked_list, id_list, stored_summaries,
                          stored_distmats):
        if not n_clicks or not stored_summaries:
            return no_update, no_update, no_update, no_update, no_update

        # Determine which files are selected
        selected_files = [
            id_item["index"]
            for id_item, checked in zip(id_list, checked_list)
            if checked
        ]

        add_log(f"Compute RF requested for {len(selected_files)} file(s): {', '.join(selected_files)}")

        if len(selected_files) < 1:
            msg = "At least 1 file must be selected to compute RF distances."
            add_log(msg, "WARNING")
            return dmc.Notification(
                title="Selection Error",
                message=msg,
                color="yellow",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), no_update, no_update, no_update, no_update

        # Collect taxa counts for selected files
        taxa_counts = {}
        for filename in selected_files:
            summary = stored_summaries.get(filename, {})
            taxa_counts[filename] = summary.get("n_taxa", 0)

        add_log(f"Taxa counts: {taxa_counts}")

        unique_counts = set(taxa_counts.values())
        if len(unique_counts) > 1:
            details = "; ".join(
                f"{fname}: {count} taxa" for fname, count in taxa_counts.items()
            )
            msg = f"Selected files have different numbers of taxa ({details}). All files must share the same taxa set to compute RF distances."
            add_log(f"Taxa mismatch — aborting RF computation: {details}", "ERROR")
            return dmc.Notification(
                title="Taxa Mismatch",
                message=msg,
                color="red",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), no_update, no_update, no_update, no_update

        # --- All taxa counts match — run RF computation ---
        n_taxa = unique_counts.pop()
        add_log(f"Taxa validation passed: all {len(selected_files)} files have {n_taxa} taxa")

        tree_service = get_tree_service()

        # Compute total trees across selected files
        total_trees = sum(
            stored_summaries[f].get("total_trees", 0) for f in selected_files
        )
        add_log(f"Fetching all {total_trees} trees from {len(selected_files)} files...")

        # Fetch all trees from selected files
        sample = tree_service.get_sample_for_analysis(
            file_sources=selected_files,
            sample_size=total_trees,
            strategy="random",
        )
        sampled_trees = sample["trees"]
        add_log(f"Retrieved {len(sampled_trees)} trees for RF computation")

        if len(sampled_trees) < 2:
            msg = "Not enough trees retrieved for RF computation."
            add_log(msg, "ERROR")
            return dmc.Notification(
                title="RF Error",
                message=msg,
                color="red",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), no_update, no_update, no_update, no_update

        # Build names, newicks, translate maps, and map indices
        names = [t["name"] for t in sampled_trees]
        newicks = tree_service.prepare_trees_for_rf_analysis(sampled_trees)

        # Collect unique translate maps and build per-tree map indices
        translate_maps = []
        file_to_map_idx = {}
        for fname in selected_files:
            tmap = tree_service.db_manager.get_translate_map(fname)
            if tmap is not None:
                file_to_map_idx[fname] = len(translate_maps)
                translate_maps.append(tmap)

        map_indices = [
            file_to_map_idx.get(t["file_source"], 0) for t in sampled_trees
        ]

        add_log(
            f"Starting RF computation: {len(names)} trees, "
            f"{len(translate_maps)} translate map(s), {n_taxa} taxa"
        )

        try:
            import time as _time
            t0 = _time.time()

            from ..rf.rf import rf_distance_from_newicks, matrix_to_dict

            result_names, matrix = rf_distance_from_newicks(
                names, newicks, translate_maps, map_indices, rooted=False,
            )
            elapsed = _time.time() - t0
            add_log(
                f"RF computation complete: {len(result_names)} trees, "
                f"{elapsed:.2f}s"
            )
        except Exception as e:
            msg = f"RF computation failed: {e}"
            add_log(msg, "ERROR")
            return dmc.Notification(
                title="RF Computation Error",
                message=msg,
                color="red",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), dmc.Text(msg, c="red"), no_update, no_update, no_update

        # Convert to dict-of-dicts and store in distmat-store (single matrix only)
        distmat_dict = matrix_to_dict(result_names, matrix)
        rf_filename = "RF_distances.tsv"

        stored_distmats = {rf_filename: distmat_dict}
        add_log(f"Stored RF distance matrix as '{rf_filename}' ({len(result_names)}x{len(result_names)})")

        notification = dmc.Notification(
            title="RF Distances Computed",
            message=f"Computed {len(result_names)}x{len(result_names)} RF distance matrix in {elapsed:.2f}s.",
            color="green",
            action="show",
            autoClose=3000,
            id="compute-rf-notification",
        )

        output_indicator = dmc.Alert(
            title="RF Distance Matrix",
            children=dmc.Text(
                f"{rf_filename}: {len(result_names)} x {len(result_names)} trees",
                size="sm",
            ),
            color="green",
            variant="light",
        )

        return notification, output_indicator, stored_distmats, False, False

    # ------ MDS SECTION ON COMPUTE TAB ------

    # Enable MDS button when a distance matrix is available
    @callback(
        Output("compute-mds-button", "disabled"),
        Output("mds-status-text", "children"),
        Input("distmat-store", "data"),
    )
    def toggle_mds_button(distmat_data):
        if not distmat_data:
            return True, dmc.Text("No distance matrix computed yet.", c="dimmed",style={"padding": "20px"})
        name = next(iter(distmat_data))
        df = pd.DataFrame(distmat_data[name])
        return False, dmc.Text(f"Distance matrix: {name} ({df.shape[0]}x{df.shape[1]})", c="green")

    # Compute MDS from distance matrix
    @callback(
        Output("mds-result-store", "data"),
        Output("compute-mds-output", "children"),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("export-mds-button", "disabled"),
        Output("plot-config-store", "data", allow_duplicate=True),
        Input("compute-mds-button", "n_clicks"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_mds(n_clicks, distmat_data):
        if not n_clicks or not distmat_data:
            return no_update, no_update, no_update, no_update, no_update

        selected_distmat = next(iter(distmat_data))
        add_log(f"Computing MDS from {selected_distmat}...")

        distmat_df_data = distmat_data[selected_distmat]
        distmat_df = pd.DataFrame(distmat_df_data)
        add_log(f"Distance matrix {selected_distmat}: {distmat_df.shape[0]}x{distmat_df.shape[1]}")

        distance_matrix = distmat_df.values

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            msg = f"Distance matrix is not square: {distance_matrix.shape}"
            add_log(msg, "ERROR")
            return no_update, dmc.Text(msg, c="red"), no_update, no_update, no_update

        try:
            import time as _time
            t0 = _time.time()
            from treetracer.rf.mds import compute_mds
            n_components = min(6, distance_matrix.shape[0] - 1)
            add_log(f"Computing classical PCoA with {n_components} components...")
            embedding = compute_mds(distance_matrix, n_components=n_components, algorithm="pcoa")
            elapsed = _time.time() - t0
            add_log(f"PCoA completed in {elapsed:.2f}s: {embedding.shape[0]} points in {embedding.shape[1]}D space")
        except Exception as e:
            msg = f"MDS computation failed: {str(e)}"
            add_log(msg, "ERROR")
            return no_update, dmc.Text(msg, c="red"), no_update, no_update, no_update

        # Build MDS result dataframe
        tree_names = distmat_df.index.astype(str).tolist()
        mdscols = [f"MDS{i+1}" for i in range(n_components)]

        mds_df = pd.DataFrame(embedding, columns=mdscols)
        mds_df["tree"] = tree_names
        mds_df["group"] = mds_df["tree"].apply(lambda x: str(x).split("/")[0].strip())
        mds_df["group"] = mds_df["group"].astype(str)

        group_mapping = {val: idx for idx, val in enumerate(sorted(mds_df["group"].unique()))}
        mds_df["group_col"] = mds_df["group"].map(group_mapping)
        mds_df["treenum"] = mds_df.groupby("group").cumcount() + 1
        mds_df["size"] = 6

        mds_filename = selected_distmat.replace('.tsv', '_MDS.tsv')
        mds_df["file"] = mds_filename

        # Build metadata
        metadata = {
            "filename": mds_filename,
            "source_distmat": selected_distmat,
            "rows": len(mds_df),
            "dimensions": mdscols,
            "groups": mds_df["group"].unique().tolist(),
            "MIN_TREENUM": int(mds_df["treenum"].min()),
            "MAX_TREENUM": int(mds_df["treenum"].max()),
        }

        mds_result = {
            "metadata": metadata,
            "data": mds_df.to_dict("records"),
        }

        add_log(f"Created MDS result '{mds_filename}': {len(mds_df)} trees, {len(mds_df['group'].unique())} groups")

        n_groups = len(mds_df["group"].unique())
        output_indicator = dmc.Alert(
            title="MDS Embedding",
            children=dmc.Text(
                f"{mds_filename}: {len(mds_df)} points, {n_components}D, {n_groups} groups",
                size="sm",
            ),
            color="blue",
            variant="light",
        )

        notification = dmc.Notification(
            title="MDS Computed",
            message=f"MDS embedding from {selected_distmat}: {len(mds_df)} points, {n_components} dimensions in {elapsed:.2f}s.",
            color="green",
            action="show",
            autoClose=3000,
            id="compute-mds-notification",
        )

        return mds_result, output_indicator, notification, False, {}

    # ------ EXPORT CALLBACKS ------

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-rf-button", "n_clicks"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def export_rf_matrix(n_clicks, distmat_data):
        if not n_clicks or not distmat_data:
            return no_update
        rf_filename = next(iter(distmat_data))
        path = _save_file_dialog(default_filename=rf_filename)
        if not path:
            return no_update
        df = pd.DataFrame(distmat_data[rf_filename])
        df.to_csv(path, sep="\t")
        add_log(f"Exported RF distance matrix to {path}")
        return dmc.Notification(
            title="RF Matrix Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id="export-rf-notification",
        )

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-mds-button", "n_clicks"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def export_mds(n_clicks, mds_result):
        if not n_clicks or not mds_result or not mds_result.get("data"):
            return no_update
        metadata = mds_result["metadata"]
        path = _save_file_dialog(default_filename=metadata["filename"])
        if not path:
            return no_update
        mds_df = pd.DataFrame(mds_result["data"])
        cols = metadata["dimensions"] + ["tree", "group", "treenum"]
        export_df = mds_df[[c for c in cols if c in mds_df.columns]]
        export_df.to_csv(path, sep="\t", index=False)
        add_log(f"Exported MDS result to {path}")
        return dmc.Notification(
            title="MDS Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id="export-mds-notification",
        )

    # ------ LOAD RF / MDS FROM FILE ------

    @callback(
        Output("distmat-store", "data", allow_duplicate=True),
        Output("compute-rf-output", "children", allow_duplicate=True),
        Output("export-rf-button", "disabled", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-rf-trace-button", "disabled", allow_duplicate=True),
        Input("load-rf-button", "n_clicks"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def load_rf_matrix(n_clicks, stored_distmats):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update

        file_path = _open_tsv_dialog()
        if not file_path:
            return no_update, no_update, no_update, no_update, no_update

        try:
            df = pd.read_csv(file_path, sep="\t", index_col=0)
        except Exception as e:
            msg = f"Failed to read file: {e}"
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Load Error", message=msg, color="red",
                action="show", autoClose=8000, id="load-rf-notification",
            ), no_update

        # Validate tree names have group prefix
        all_names = list(df.index.astype(str)) + list(df.columns.astype(str))
        valid, err_msg = _validate_group_names(all_names)
        if not valid:
            add_log(f"RF load validation failed: {err_msg}", "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid Tree Names", message=err_msg, color="red",
                action="show", autoClose=8000, id="load-rf-notification",
            ), no_update

        filename = os.path.basename(file_path)
        stored_distmats = stored_distmats or {}
        stored_distmats[filename] = df.to_dict()
        add_log(f"Loaded RF distance matrix from {filename}: {df.shape[0]}x{df.shape[1]}")

        output_indicator = dmc.Alert(
            title="RF Distance Matrix",
            children=dmc.Text(
                f"{filename}: {df.shape[0]} x {df.shape[1]} trees",
                size="sm",
            ),
            color="green",
            variant="light",
        )

        notification = dmc.Notification(
            title="RF Matrix Loaded",
            message=f"Loaded {df.shape[0]}x{df.shape[1]} distance matrix from {filename}.",
            color="green",
            action="show",
            autoClose=3000,
            id="load-rf-notification",
        )

        return stored_distmats, output_indicator, False, notification, False

    @callback(
        Output("mds-result-store", "data", allow_duplicate=True),
        Output("compute-mds-output", "children", allow_duplicate=True),
        Output("export-mds-button", "disabled", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("load-mds-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def load_mds(n_clicks):
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        file_path = _open_tsv_dialog()
        if not file_path:
            return no_update, no_update, no_update, no_update

        try:
            mds_df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            msg = f"Failed to read file: {e}"
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Load Error", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        if "group" not in mds_df.columns:
            msg = "MDS file must contain a 'group' column."
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid MDS File", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        if mds_df["group"].isna().any() or (mds_df["group"].astype(str).str.strip() == "").any():
            msg = "The 'group' column must not contain empty values."
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid MDS File", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        mds_df["group"] = mds_df["group"].astype(str)

        group_mapping = {val: idx for idx, val in enumerate(sorted(mds_df["group"].unique()))}
        mds_df["group_col"] = mds_df["group"].map(group_mapping)

        if "treenum" not in mds_df.columns:
            mds_df["treenum"] = mds_df.groupby("group").cumcount() + 1
        mds_df["size"] = 6

        mds_filename = os.path.basename(file_path)
        mds_df["file"] = mds_filename

        # Detect MDS dimension columns
        mdscols = [c for c in mds_df.columns if c.startswith("MDS")]
        if not mdscols:
            msg = "No MDS dimension columns found (expected columns starting with 'MDS')."
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid MDS File", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        metadata = {
            "filename": mds_filename,
            "source_distmat": "loaded_from_file",
            "rows": len(mds_df),
            "dimensions": mdscols,
            "groups": mds_df["group"].unique().tolist(),
            "MIN_TREENUM": int(mds_df["treenum"].min()),
            "MAX_TREENUM": int(mds_df["treenum"].max()),
        }

        mds_result = {
            "metadata": metadata,
            "data": mds_df.to_dict("records"),
        }

        n_groups = len(mds_df["group"].unique())
        add_log(f"Loaded MDS from {mds_filename}: {len(mds_df)} points, {len(mdscols)}D, {n_groups} groups")

        output_indicator = dmc.Alert(
            title="MDS Embedding",
            children=dmc.Text(
                f"{mds_filename}: {len(mds_df)} points, {len(mdscols)}D, {n_groups} groups",
                size="sm",
            ),
            color="blue",
            variant="light",
        )

        notification = dmc.Notification(
            title="MDS Loaded",
            message=f"Loaded MDS from {mds_filename}: {len(mds_df)} points, {len(mdscols)} dimensions, {n_groups} groups.",
            color="green",
            action="show",
            autoClose=6000,
            id="load-mds-notification",
        )

        return mds_result, output_indicator, False, notification
