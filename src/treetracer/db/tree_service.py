"""High-level service for phylogenetic tree loading, storage, and sampling.

This module provides the main API for TreeTracer's tree processing functionality:
- Loading nexus files with high-performance streaming
- Flexible sampling strategies (random, uniform, stratified)
- Integration between database storage and analysis preparation
- Metadata management for phylogenetic annotations
"""

from typing import List, Dict, Any, Optional
import json
import time
import pandas as pd
from .process_trees import process_nexus_trees_streaming
from .tree_manager import get_tree_manager


class TreeService:
    """High-level service for phylogenetic tree operations.
    
    Provides a clean API for loading nexus files, managing tree storage,
    and sampling trees for analysis. Optimized for large-scale phylogenetic
    data processing with configurable performance parameters.
    """
    
    def __init__(self):
        self.db_manager = get_tree_manager()
    
    def load_nexus_file(self, nexus_file_path: str, file_source: Optional[str] = None) -> Dict[str, Any]:
        """Load a nexus file into the database with optimized streaming.
        
        Uses high-performance streaming parser to handle massive files (GB+)
        with large newick strings. Automatically extracts metadata from
        Beast/MrBayes annotations and uses filename as group identifier.
        
        Args:
            nexus_file_path: Path to the nexus file to load
            file_source: Identifier for this file (defaults to filename)
            
        Returns:
            Dict containing:
            - success: Boolean indicating if load succeeded
            - trees_loaded: Number of trees successfully loaded
            - file_source: The file identifier used
            - database_stats: Current database statistics
            - error: Error message if success=False
        """
        if file_source is None:
            file_source = nexus_file_path.split('/')[-1]
        
        try:
            start_time = time.time()
            
            # Stream directly to database with optimized batch processing
            inserted_count = process_nexus_trees_streaming(
                nexus_file_path, 
                self.db_manager, 
                file_source, 
                batch_size=200,    # Optimized for ~700 trees/second
                transaction_size=1000  # Balance memory vs commit frequency
            )
            
            total_time = time.time() - start_time
            print(f"Total loading time: {total_time:.2f}s")
            
            stats = self.db_manager.get_database_stats()
            
            return {
                'success': True,
                'trees_loaded': inserted_count,
                'file_source': file_source,
                'database_stats': stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'trees_loaded': 0
            }
    
    
    def get_sample_for_analysis(self, 
                               file_sources: Optional[List[str]] = None,
                               group_names: Optional[List[str]] = None,
                               sample_size: int = 500,
                               strategy: str = 'stratified') -> Dict[str, Any]:
        """Get a sample of trees for phylogenetic analysis.
        
        Supports multiple sampling strategies and filtering options.
        Designed to prepare data for RF distance computation and MDS analysis.
        
        Args:
            file_sources: Filter by specific file sources (None for all files)
            group_names: Filter by specific groups (None for all groups) 
            sample_size: Target number of trees to sample
            strategy: Sampling method - 'random', 'uniform', or 'stratified'
            
        Returns:
            Dict containing:
            - trees: List of sampled tree dictionaries
            - sample_size: Actual number of trees sampled
            - strategy: Strategy used for sampling
            - filters_applied: Filters that were applied
            - ready_for_analysis: Boolean indicating readiness
            - strata_info: Breakdown by strata (for stratified sampling)
        """
        
        # Build filters
        filters = {}
        if file_sources and len(file_sources) == 1:
            filters['file_source'] = file_sources[0]
        if group_names and len(group_names) == 1:
            filters['group_name'] = group_names[0]
        
        # Handle stratified sampling across multiple files/groups
        if strategy == 'stratified' and (
            (file_sources and len(file_sources) > 1) or 
            (group_names and len(group_names) > 1)
        ):
            return self._stratified_sample_multiple(
                file_sources, group_names, sample_size
            )
        
        # Simple sampling
        sampled_trees = self.db_manager.get_trees_sample(
            filters=filters,
            limit=sample_size,
            strategy=strategy
        )
        
        return {
            'trees': sampled_trees,
            'sample_size': len(sampled_trees),
            'strategy': strategy,
            'filters_applied': filters,
            'ready_for_analysis': True
        }
    
    def _stratified_sample_multiple(self, 
                                   file_sources: Optional[List[str]], 
                                   group_names: Optional[List[str]], 
                                   total_sample_size: int) -> Dict[str, Any]:
        """Perform stratified sampling across multiple files/groups."""
        
        # Get database stats to determine strata
        stats = self.db_manager.get_database_stats()
        
        sampled_trees = []
        strata_info = []
        
        # Determine strata (files or groups)
        if file_sources and len(file_sources) > 1:
            strata = [(f, 'file_source') for f in file_sources]
            strata_counts = stats.get('trees_per_file', {})
        elif group_names and len(group_names) > 1:
            strata = [(g, 'group_name') for g in group_names]
            strata_counts = stats.get('trees_per_group', {})
        else:
            # Fallback to simple sampling
            return self.get_sample_for_analysis(
                file_sources, group_names, total_sample_size, 'random'
            )
        
        # Calculate sample size per stratum (proportional allocation)
        total_trees_in_strata = sum(strata_counts.get(s[0], 0) for s in strata)
        
        if total_trees_in_strata == 0:
            return {
                'trees': [],
                'sample_size': 0,
                'strategy': 'stratified',
                'error': 'No trees found for specified strata'
            }
        
        for stratum_name, stratum_type in strata:
            stratum_count = strata_counts.get(stratum_name, 0)
            if stratum_count == 0:
                continue
                
            # Proportional allocation
            stratum_sample_size = max(1, int(
                (stratum_count / total_trees_in_strata) * total_sample_size
            ))
            
            # Sample from this stratum
            filters = {stratum_type: stratum_name}
            stratum_trees = self.db_manager.get_trees_sample(
                filters=filters,
                limit=stratum_sample_size,
                strategy='random'
            )
            
            sampled_trees.extend(stratum_trees)
            strata_info.append({
                'stratum': stratum_name,
                'total_trees': stratum_count,
                'sampled_trees': len(stratum_trees)
            })
        
        return {
            'trees': sampled_trees,
            'sample_size': len(sampled_trees),
            'strategy': 'stratified',
            'strata_info': strata_info,
            'ready_for_analysis': True
        }
    
    def get_metadata_traces(self, file_sources: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Extract numeric time-series fields from tree metadata.

        Scans tree metadata for known log-likelihood / posterior fields and
        returns one DataFrame per field found.

        Args:
            file_sources: Restrict to these files (None = all loaded files).

        Returns:
            Dict mapping field name to DataFrame with columns:
            treenum, value, group, file_source.
        """
        self.db_manager.flush()
        df = self.db_manager._trees

        if file_sources:
            df = df[df['file_source'].isin(file_sources)]

        if len(df) == 0:
            return {}

        # Known numeric metadata keys (BEAST / MrBayes conventions)
        target_fields = {'lnP', 'loglikelihood', 'lnL', 'posterior', 'joint'}

        # Parse metadata JSON and collect values
        records = []
        for _, row in df.iterrows():
            meta = row['metadata']
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(meta, dict):
                continue
            for field in target_fields:
                if field in meta:
                    try:
                        val = float(meta[field])
                    except (ValueError, TypeError):
                        continue
                    records.append({
                        'id': int(row['id']),
                        'field': field,
                        'value': val,
                        'group': row['group_name'],
                        'file_source': row['file_source'],
                    })

        if not records:
            return {}

        all_df = pd.DataFrame(records)

        # Build per-field DataFrames with treenum = cumulative count per group
        result = {}
        for field_name, field_df in all_df.groupby('field'):
            field_df = field_df.sort_values('id')
            field_df['treenum'] = field_df.groupby('group').cumcount() + 1
            result[field_name] = field_df[['treenum', 'value', 'group', 'file_source']].reset_index(drop=True)

        return result

    def prepare_trees_for_rf_analysis(self, sampled_trees: List[Dict[str, Any]]) -> List[str]:
        """Extract newick strings for Robinson-Foulds distance computation.
        
        Args:
            sampled_trees: List of tree dictionaries from get_sample_for_analysis
            
        Returns:
            List of newick strings ready for RF distance libraries
        """
        return [tree['newick'] for tree in sampled_trees]
    


# Global instance
_tree_service = None

def get_tree_service() -> TreeService:
    """Get the global tree service instance."""
    global _tree_service
    if _tree_service is None:
        _tree_service = TreeService()
    return _tree_service