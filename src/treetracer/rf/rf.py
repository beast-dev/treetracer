"""Robinson-Foulds distance computation module for phylogenetic trees.

This module provides high-performance Robinson-Foulds distance computation for
rooted, bifurcating phylogenetic trees stored in the TreeTracer database.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import dendropy

# Handle imports for both direct execution and package import
try:
    from ..db.tree_service import TreeService
    from ..db.db_manager import get_db_manager
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from treetracer.db.tree_service import TreeService
    from treetracer.db.db_manager import get_db_manager


class RobinsonFouldsCalculator:
    """High-performance Robinson-Foulds distance calculator for rooted, bifurcating trees."""
    
    def __init__(self):
        self.tree_service = TreeService()
        self.db_manager = get_db_manager()
        self._taxon_namespace = None
        
    def sample_trees_from_database(self, 
                                 file_source: str, 
                                 sample_size: int, 
                                 strategy: str = 'random',
                                 use_last_trees: bool = False) -> List[Dict[str, Any]]:
        """Sample trees from database for RF distance computation."""
        if use_last_trees:
            # Get the last N trees from the database (end of MCMC chain)
            conn = self.db_manager.get_connection()
            query = """
                SELECT id, name, newick, file_source, group_name, metadata
                FROM trees 
                WHERE file_source = ?
                ORDER BY id DESC
                LIMIT ?
            """
            results = conn.execute(query, (file_source, sample_size)).fetchall()
            
            trees = []
            for row in results:
                trees.append({
                    'id': row[0],
                    'name': row[1],
                    'newick': row[2],
                    'file_source': row[3],
                    'group_name': row[4],
                    'metadata': row[5]
                })
            
            # Reverse to get chronological order
            trees = trees[::-1]
            print(f"Retrieved last {len(trees)} trees from database (IDs: {[t['id'] for t in trees]})")
            
        else:
            # Use original sampling method
            result = self.tree_service.get_sample_for_analysis(
                file_sources=[file_source],
                sample_size=sample_size,
                strategy=strategy
            )
            trees = result.get('trees', [])
            print(f"Retrieved {len(trees)} trees from database using {strategy} sampling")
        
        # Debug: Check tree data integrity
        if trees:
            first_tree = trees[0]
            newick_len = len(first_tree.get('newick', ''))
            print(f"First tree newick length: {newick_len} characters")
            print(f"Tree IDs in sample: {[t['id'] for t in trees]}")
        
        return trees
    
    def convert_to_dendropy_trees(self, tree_records: List[Dict[str, Any]]) -> List[dendropy.Tree]:
        """Convert sampled tree records to DendroPy Tree objects.
        
        All trees are treated as rooted and bifurcating.
        """
        if not tree_records:
            return []
        
        # Create shared taxon namespace for consistent tree comparison
        if self._taxon_namespace is None:
            self._taxon_namespace = dendropy.TaxonNamespace()
        
        dendropy_trees = []
        
        for i, record in enumerate(tree_records):
            try:
                newick_str = record['newick']
                
                # Debug: Check newick string integrity
                if not newick_str or not newick_str.strip():
                    print(f"Warning: Empty newick string for tree {i} (ID: {record.get('id')})")
                    continue
                
                # Clean and validate newick string
                original_newick = newick_str
                newick_str = newick_str.strip()
                if len(newick_str) != len(original_newick):
                    print(f"Info: Trimmed whitespace from newick string for tree {i} (ID: {record.get('id')})")
                
                # Remove Beast/MrBayes annotations from the beginning (e.g., [&R])
                import re
                if newick_str.startswith('['):
                    # Find the closing bracket and remove everything up to it
                    match = re.match(r'\[[^\]]*\]\s*', newick_str)
                    if match:
                        removed_annotation = match.group(0)
                        newick_str = newick_str[match.end():]
                        print(f"Info: Removed annotation '{removed_annotation.strip()}' from newick string for tree {i} (ID: {record.get('id')})")
                
                # Basic newick format validation
                if not newick_str.startswith('('):
                    print(f"Warning: Newick string doesn't start with '(' for tree {i} (ID: {record.get('id')})")
                    print(f"  Starts with: {newick_str[:50]}...")
                    continue
                
                # Ensure newick string ends with semicolon
                if not newick_str.endswith(';'):
                    print(f"Info: Adding semicolon to newick string for tree {i} (ID: {record.get('id')})")
                    newick_str += ';'
                
                # Check for balanced parentheses
                open_count = newick_str.count('(')
                close_count = newick_str.count(')')
                if open_count != close_count:
                    print(f"Warning: Unbalanced parentheses for tree {i} (ID: {record.get('id')}): {open_count} open, {close_count} close")
                    continue
                
                # Debug: Check for obvious truncation
                if len(newick_str) < 50:  # Very short trees might be truncated
                    print(f"Warning: Suspiciously short newick string for tree {i} (ID: {record.get('id')}, length: {len(newick_str)})")
                    print(f"  Newick preview: {newick_str[:100]}...")
                
                # Create rooted DendroPy tree
                tree = dendropy.Tree.get(
                    data=newick_str,
                    schema='newick',
                    taxon_namespace=self._taxon_namespace,
                    rooting='force-rooted'
                )
                
                # Store original database ID for reference
                tree.database_id = record.get('id')
                tree.tree_index = record.get('tree_index')
                
                dendropy_trees.append(tree)
                
            except Exception as e:
                print(f"Warning: Failed to parse tree {i} (ID: {record.get('id')}): {e}")
                # Debug: Show more details about the problematic tree
                newick_str = record.get('newick', '')
                print(f"  Tree length: {len(newick_str)} characters")
                print(f"  Tree starts: {newick_str[:100]}...")
                print(f"  Tree ends: ...{newick_str[-100:]}")
                continue
        
        print(f"Successfully parsed {len(dendropy_trees)} out of {len(tree_records)} trees")
        return dendropy_trees
    
    def compute_pairwise_rf_distances(self, trees: List[dendropy.Tree]) -> np.ndarray:
        """Compute pairwise Robinson-Foulds distances between rooted trees."""
        n_trees = len(trees)
        if n_trees == 0:
            return np.array([])
        
        # Initialize distance matrix
        rf_matrix = np.zeros((n_trees, n_trees), dtype=float)
        
        # Compute pairwise distances
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                try:
                    # Unweighted Robinson-Foulds distance for rooted trees
                    rf_distance = dendropy.calculate.treecompare.symmetric_difference(
                        trees[i], trees[j]
                    )
                    
                    # Store in symmetric matrix
                    rf_matrix[i, j] = rf_distance
                    rf_matrix[j, i] = rf_distance
                    
                except Exception as e:
                    print(f"Warning: Failed to compute RF distance between trees {i} and {j}: {e}")
                    rf_matrix[i, j] = np.nan
                    rf_matrix[j, i] = np.nan
        
        return rf_matrix
    
    def compute_rf_distances_from_database(self, 
                                         file_source: str,
                                         sample_size: int,
                                         strategy: str = 'random',
                                         use_last_trees: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Complete pipeline: sample trees from database and compute RF distances."""
        # Sample trees from database
        tree_records = self.sample_trees_from_database(file_source, sample_size, strategy, use_last_trees)
        
        if not tree_records:
            return np.array([]), []
        
        # Convert to DendroPy objects
        dendropy_trees = self.convert_to_dendropy_trees(tree_records)
        
        if not dendropy_trees:
            return np.array([]), tree_records
        
        # Compute RF distances
        rf_matrix = self.compute_pairwise_rf_distances(dendropy_trees)
        
        return rf_matrix, tree_records
    
    def get_database_stats(self, file_source: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics for tree files."""
        conn = self.db_manager.get_connection()
        
        if file_source:
            count_result = conn.execute(
                "SELECT COUNT(*) FROM trees WHERE file_source = ?", 
                (file_source,)
            ).fetchone()
            
            return {
                'file_source': file_source,
                'tree_count': count_result[0] if count_result else 0
            }
        else:
            file_stats = conn.execute("""
                SELECT file_source, COUNT(*) as tree_count
                FROM trees 
                GROUP BY file_source
                ORDER BY tree_count DESC
            """).fetchall()
            
            total_trees = conn.execute("SELECT COUNT(*) FROM trees").fetchone()[0]
            
            return {
                'total_trees': total_trees,
                'files': [{'file_source': row[0], 'tree_count': row[1]} for row in file_stats]
            }
    
    def clear_taxon_namespace(self):
        """Clear the shared taxon namespace for memory cleanup."""
        self._taxon_namespace = None


def benchmark_rf_computation(file_source: str, 
                           sample_sizes: List[int],
                           strategy: str = 'random') -> Dict[str, Any]:
    """Benchmark RF computation performance across different sample sizes."""
    calculator = RobinsonFouldsCalculator()
    results = []
    
    for sample_size in sample_sizes:
        print(f"Benchmarking RF computation for {sample_size} trees...")
        
        start_time = time.time()
        
        rf_matrix, tree_records = calculator.compute_rf_distances_from_database(
            file_source=file_source,
            sample_size=sample_size,
            strategy=strategy
        )
        
        end_time = time.time()
        
        computation_time = end_time - start_time
        trees_processed = len(tree_records)
        comparisons_made = (trees_processed * (trees_processed - 1)) // 2
        
        result = {
            'sample_size': sample_size,
            'trees_processed': trees_processed,
            'computation_time': computation_time,
            'comparisons_made': comparisons_made,
            'comparisons_per_second': comparisons_made / computation_time if computation_time > 0 else 0,
            'matrix_shape': rf_matrix.shape,
            'mean_rf_distance': np.nanmean(rf_matrix[rf_matrix > 0]) if rf_matrix.size > 0 else 0,
            'max_rf_distance': np.nanmax(rf_matrix) if rf_matrix.size > 0 else 0,
            'successful_comparisons': np.sum(~np.isnan(rf_matrix[rf_matrix > 0])) if rf_matrix.size > 0 else 0
        }
        
        results.append(result)
        print(f"  Processed {trees_processed} trees in {computation_time:.2f}s")
        print(f"  {comparisons_made} comparisons at {result['comparisons_per_second']:.1f} comparisons/sec")
    
    return {
        'file_source': file_source,
        'strategy': strategy,
        'results': results
    }