"""Pandas-based tree metadata manager with flat-file newick storage.

Tree metadata (name, group, file source, newick offset/length) is stored in a
pandas DataFrame. Newick strings are read on demand from the original .trees
file via seek + read. No temporary database files, no SQL, no extra
dependencies beyond pandas.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


class TreeManagerPandas:
    """Tree metadata manager backed by a pandas DataFrame.

    Stores only lightweight metadata in memory. Newick strings stay in the
    original .trees files and are read on demand via byte offsets.
    """

    def __init__(self):
        self._trees = pd.DataFrame(columns=[
            'id', 'name', 'newick_offset', 'newick_length',
            'file_source', 'group_name', 'metadata',
        ])
        self._trees = self._trees.astype({
            'id': 'int64',
            'newick_offset': 'int64',
            'newick_length': 'int32',
        })
        self._current_max_id = 0
        self._source_files = {}       # file_source -> absolute file path
        self._source_handles = {}     # file_source -> open file handle
        self._pending_rows = []       # buffer for batch append

    # ------------------------------------------------------------------
    # Source file registration & newick I/O
    # ------------------------------------------------------------------

    def register_source_file(self, file_source: str, file_path: str):
        """Register the original file path for a given file_source."""
        self._source_files[file_source] = os.path.abspath(file_path)

    def _get_source_handle(self, file_source: str):
        if file_source not in self._source_handles:
            path = self._source_files[file_source]
            self._source_handles[file_source] = open(path, 'rb')
        return self._source_handles[file_source]

    def _read_newick(self, file_source: str, offset: int, length: int) -> str:
        fh = self._get_source_handle(file_source)
        fh.seek(offset)
        return fh.read(length).decode('utf-8')

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_trees_batch_raw(self, raw_trees_data: List[tuple]) -> int:
        """Append tree metadata rows.

        Args:
            raw_trees_data: List of tuples
                (name, newick_offset, newick_length, file_source, group_name, metadata)

        Returns:
            Number of trees inserted
        """
        if not raw_trees_data:
            return 0

        for name, newick_offset, newick_length, file_source, group_name, metadata in raw_trees_data:
            self._current_max_id += 1
            metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata
            self._pending_rows.append({
                'id': self._current_max_id,
                'name': name,
                'newick_offset': newick_offset,
                'newick_length': newick_length,
                'file_source': file_source,
                'group_name': group_name,
                'metadata': metadata_json,
            })

        return len(raw_trees_data)

    def flush(self):
        """Flush pending rows into the DataFrame.

        Called automatically by get_trees_sample / get_database_stats,
        but can be called explicitly after loading is complete.
        """
        if not self._pending_rows:
            return
        batch_df = pd.DataFrame(self._pending_rows)
        batch_df['id'] = batch_df['id'].astype('int64')
        batch_df['newick_offset'] = batch_df['newick_offset'].astype('int64')
        batch_df['newick_length'] = batch_df['newick_length'].astype('int32')
        batch_df['file_source'] = batch_df['file_source'].astype('category')
        batch_df['group_name'] = batch_df['group_name'].astype('category')
        if len(self._trees) == 0:
            self._trees = batch_df
        else:
            self._trees = pd.concat([self._trees, batch_df], ignore_index=True)
        self._pending_rows = []

    # ------------------------------------------------------------------
    # Commit / transaction stubs (no-ops — keeps interface compatible
    # with process_trees_offset which calls BEGIN/COMMIT)
    # ------------------------------------------------------------------

    def get_connection(self):
        """Return self so that conn.execute("BEGIN/COMMIT") works as a no-op."""
        return self

    def execute(self, sql, params=None):
        """No-op for transaction SQL statements."""
        pass

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _resolve_newick(self, rows: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame rows to list of dicts with newick strings read from file."""
        trees = []
        for _, row in rows.iterrows():
            newick = self._read_newick(row['file_source'], int(row['newick_offset']), int(row['newick_length']))
            meta = row['metadata']
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    pass
            trees.append({
                'id': int(row['id']),
                'name': row['name'],
                'newick': newick,
                'file_source': row['file_source'],
                'group_name': row['group_name'],
                'metadata': meta,
            })
        return trees

    def get_trees_sample(self,
                        filters: Optional[Dict[str, Any]] = None,
                        limit: int = 500,
                        strategy: str = 'random') -> List[Dict[str, Any]]:
        """Sample trees. Returns dicts with 'newick' key (read from file).

        Strategies:
            random     - uniform random sample
            uniform    - evenly spaced by insertion order
            stratified - proportional per group_name
        """
        self.flush()

        df = self._trees

        # Apply filters
        if filters:
            if 'file_source' in filters:
                df = df[df['file_source'] == filters['file_source']]
            if 'group_name' in filters:
                df = df[df['group_name'] == filters['group_name']]

        if len(df) == 0:
            return []

        if strategy == 'random':
            n = min(limit, len(df))
            sampled = df.sample(n=n)

        elif strategy == 'uniform':
            if len(df) <= limit:
                sampled = df
            else:
                step = max(1, len(df) // limit)
                sampled = df.iloc[::step].head(limit)

        elif strategy == 'stratified':
            groups = df['group_name'].unique()
            total = len(df)
            parts = []
            for g in groups:
                g_df = df[df['group_name'] == g]
                n_g = max(1, int(len(g_df) / total * limit))
                n_g = min(n_g, len(g_df))
                parts.append(g_df.sample(n=n_g))
            sampled = pd.concat(parts)
        else:
            sampled = df.head(limit)

        return self._resolve_newick(sampled)

    def get_last_trees(self,
                       file_source: str,
                       limit: int = 500) -> List[Dict[str, Any]]:
        """Get the last N trees (by insertion order) for a given file_source.

        Useful for retrieving the end of an MCMC chain.

        Returns trees in chronological order (earliest first).
        """
        self.flush()
        df = self._trees[self._trees['file_source'] == file_source]
        if len(df) == 0:
            return []
        # Sort descending by id, take last N, then reverse to chronological order
        last_n = df.sort_values('id', ascending=False).head(limit)
        last_n = last_n.iloc[::-1]
        return self._resolve_newick(last_n)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_database_stats(self) -> Dict[str, Any]:
        self.flush()
        df = self._trees
        return {
            'total_trees': len(df),
            'trees_per_file': df.groupby('file_source', observed=True).size().to_dict() if len(df) > 0 else {},
            'trees_per_group': df[df['group_name'].notna()].groupby('group_name', observed=True).size().to_dict() if len(df) > 0 else {},
        }

    # ------------------------------------------------------------------
    # Clear / cleanup
    # ------------------------------------------------------------------

    def clear_trees(self, file_source: Optional[str] = None):
        if file_source:
            self._trees = self._trees[self._trees['file_source'] != file_source]
            if file_source in self._source_handles:
                self._source_handles[file_source].close()
                del self._source_handles[file_source]
        else:
            self._trees = self._trees.iloc[0:0]
            self._current_max_id = 0
            self._pending_rows = []
            for fh in self._source_handles.values():
                fh.close()
            self._source_handles.clear()

    def cleanup(self):
        """Close source file handles. No temp files to delete."""
        for fh in self._source_handles.values():
            fh.close()
        self._source_handles.clear()
        self._trees = self._trees.iloc[0:0]
        self._pending_rows = []


# Global singleton
_tree_manager = None

def get_tree_manager() -> TreeManagerPandas:
    """Get the global tree manager instance."""
    global _tree_manager
    if _tree_manager is None:
        _tree_manager = TreeManagerPandas()
    return _tree_manager
