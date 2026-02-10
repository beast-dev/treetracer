"""Robinson-Foulds distance computation module for phylogenetic trees.

This module provides high-performance Robinson-Foulds distance computation for
rooted, bifurcating phylogenetic trees stored in the TreeTracer database.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import dendropy
import multiprocessing as mp
from functools import partial

# Handle imports for both direct execution and package import
try:
    from ..db.tree_service import TreeService
    from ..db.tree_manager import get_tree_manager
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from treetracer.db.tree_service import TreeService
    from treetracer.db.tree_manager import get_tree_manager


def _compute_rf_distance_worker(tree_pair_data: Tuple[str, str, int, int]) -> Tuple[int, int, float, float]:
    """Worker function for parallel RF distance computation.
    
    Args:
        tree_pair_data: Tuple of (tree1_newick, tree2_newick, i, j)
        
    Returns:
        Tuple of (i, j, rf_distance, computation_time)
    """
    tree1_newick, tree2_newick, i, j = tree_pair_data
    
    start_time = time.time()
    
    try:
        # Create a temporary taxon namespace for this worker
        temp_namespace = dendropy.TaxonNamespace()
        
        # Parse both trees with the same namespace
        tree1 = dendropy.Tree.get(
            data=tree1_newick,
            schema='newick',
            taxon_namespace=temp_namespace,
            rooting='force-rooted'
        )
        
        tree2 = dendropy.Tree.get(
            data=tree2_newick,
            schema='newick',
            taxon_namespace=temp_namespace,
            rooting='force-rooted'
        )
        
        # Compute RF distance
        rf_distance = dendropy.calculate.treecompare.symmetric_difference(tree1, tree2)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        return i, j, rf_distance, computation_time
        
    except Exception as e:
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Warning: Failed to compute RF distance between trees {i} and {j}: {e}")
        return i, j, np.nan, computation_time


class RobinsonFouldsCalculator:
    """Robinson-Foulds distance calculator for rooted, bifurcating trees."""
    
    def __init__(self):
        self.tree_service = TreeService()
        self.db_manager = get_tree_manager()
        self._taxon_namespace = None
        
    def sample_trees_from_database(self, 
                                 file_source: str, 
                                 sample_size: int, 
                                 strategy: str = 'random',
                                 use_last_trees: bool = False) -> List[Dict[str, Any]]:
        """Sample trees from database for RF distance computation."""
        if use_last_trees:
            # Get the last N trees from the database (end of MCMC chain)
            trees = self.db_manager.get_last_trees(file_source, limit=sample_size)
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
    
    def convert_to_dendropy_trees(self, tree_records: List[Dict[str, Any]]) -> Tuple[List[dendropy.Tree], Dict[str, Any]]:
        """Convert sampled tree records to DendroPy Tree objects.
        
        All trees are treated as rooted and bifurcating.
        Returns trees and timing statistics.
        """
        if not tree_records:
            return [], {}
        
        # Create shared taxon namespace for consistent tree comparison
        if self._taxon_namespace is None:
            self._taxon_namespace = dendropy.TaxonNamespace()
        
        dendropy_trees = []
        tree_sizes = []
        parse_times = []
        
        total_start_time = time.time()
        
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
                    # print(f"Info: Trimmed whitespace from newick string for tree {i} (ID: {record.get('id')})")
                    continue
                    
                # Remove Beast/MrBayes annotations from the beginning (e.g., [&R])
                import re
                if newick_str.startswith('['):
                    # Find the closing bracket and remove everything up to it
                    match = re.match(r'\[[^\]]*\]\s*', newick_str)
                    if match:
                        removed_annotation = match.group(0)
                        newick_str = newick_str[match.end():]
                        # print(f"Info: Removed annotation '{removed_annotation.strip()}' from newick string for tree {i} (ID: {record.get('id')})")
                
                # Basic newick format validation
                if not newick_str.startswith('('):
                    # print(f"Warning: Newick string doesn't start with '(' for tree {i} (ID: {record.get('id')})")
                    # print(f"  Starts with: {newick_str[:50]}...")
                    continue
                
                # Ensure newick string ends with semicolon
                if not newick_str.endswith(';'):
                    # print(f"Info: Adding semicolon to newick string for tree {i} (ID: {record.get('id')})")
                    newick_str += ';'
                
                # Check for balanced parentheses
                open_count = newick_str.count('(')
                close_count = newick_str.count(')')
                if open_count != close_count:
                    # print(f"Warning: Unbalanced parentheses for tree {i} (ID: {record.get('id')}): {open_count} open, {close_count} close")
                    continue
                
                # Debug: Check for obvious truncation
                if len(newick_str) < 50:  # Very short trees might be truncated
                    # print(f"Warning: Suspiciously short newick string for tree {i} (ID: {record.get('id')}, length: {len(newick_str)})")
                    # print(f"  Newick preview: {newick_str[:100]}...")
                    continue
                
                # Time the tree parsing
                parse_start_time = time.time()
                
                # Create rooted DendroPy tree
                tree = dendropy.Tree.get(
                    data=newick_str,
                    schema='newick',
                    taxon_namespace=self._taxon_namespace,
                    rooting='force-rooted'
                )
                
                parse_end_time = time.time()
                parse_time = parse_end_time - parse_start_time
                parse_times.append(parse_time)
                
                # Store original database ID for reference
                tree.database_id = record.get('id')
                tree.tree_index = record.get('tree_index')
                
                # Calculate tree size metrics
                tree_size = {
                    'newick_length': len(newick_str),
                    'num_taxa': len(tree.taxon_namespace),
                    'num_nodes': len(tree.nodes()),
                    'num_edges': len(tree.edges())
                }
                tree_sizes.append(tree_size)
                
                dendropy_trees.append(tree)
                
            except Exception as e:
                print(f"Warning: Failed to parse tree {i} (ID: {record.get('id')}): {e}")
                # Debug: Show more details about the problematic tree
                newick_str = record.get('newick', '')
                print(f"  Tree length: {len(newick_str)} characters")
                print(f"  Tree starts: {newick_str[:100]}...")
                print(f"  Tree ends: ...{newick_str[-100:]}")
                continue
        
        total_end_time = time.time()
        total_parse_time = total_end_time - total_start_time
        
        # Calculate timing statistics
        timing_stats = {
            'total_parse_time': total_parse_time,
            'average_parse_time': np.mean(parse_times) if parse_times else 0,
            'min_parse_time': np.min(parse_times) if parse_times else 0,
            'max_parse_time': np.max(parse_times) if parse_times else 0,
            'trees_per_second': len(dendropy_trees) / total_parse_time if total_parse_time > 0 else 0,
            'individual_parse_times': parse_times,
            'tree_sizes': tree_sizes
        }
        
        print(f"Successfully parsed {len(dendropy_trees)} out of {len(tree_records)} trees")
        if tree_sizes:
            avg_newick_len = np.mean([s['newick_length'] for s in tree_sizes])
            avg_taxa = np.mean([s['num_taxa'] for s in tree_sizes])
            print(f"Average newick length: {avg_newick_len:.0f} characters")
            print(f"Average number of taxa: {avg_taxa:.0f}")
            print(f"Tree parsing rate: {timing_stats['trees_per_second']:.1f} trees/sec")
        
        return dendropy_trees, timing_stats
    
    def compute_pairwise_rf_distances(self, trees: List[dendropy.Tree], n_processes: int = 4) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute pairwise Robinson-Foulds distances between rooted trees using multiprocessing.
        
        Args:
            trees: List of DendroPy Tree objects
            n_processes: Number of parallel processes to use
            
        Returns:
            Tuple of (distance matrix, timing statistics)
        """
        n_trees = len(trees)
        if n_trees == 0:
            return np.array([]), {}
        
        # Initialize distance matrix
        rf_matrix = np.zeros((n_trees, n_trees), dtype=float)
        
        total_start_time = time.time()
        
        # Extract newick strings from trees for multiprocessing
        tree_newicks = []
        for tree in trees:
            newick_str = tree.as_string(schema='newick').strip()
            tree_newicks.append(newick_str)
        
        # Prepare work items (tree pairs to compare)
        work_items = []
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                work_items.append((tree_newicks[i], tree_newicks[j], i, j))
        
        comparisons_made = len(work_items)
        successful_comparisons = 0
        rf_times = []
        
        print(f"Computing {comparisons_made} RF distances using {n_processes} processes...")
        
        # Use multiprocessing to compute RF distances in parallel
        if n_processes > 1 and comparisons_made > 1:
            with mp.Pool(processes=n_processes) as pool:
                results = pool.map(_compute_rf_distance_worker, work_items)
        else:
            # Fallback to sequential processing
            results = [_compute_rf_distance_worker(item) for item in work_items]
        
        # Process results
        for i, j, rf_distance, computation_time in results:
            rf_times.append(computation_time)
            
            if not np.isnan(rf_distance):
                successful_comparisons += 1
                
            # Store in symmetric matrix
            rf_matrix[i, j] = rf_distance
            rf_matrix[j, i] = rf_distance
        
        total_end_time = time.time()
        total_rf_time = total_end_time - total_start_time
        
        # Calculate RF timing statistics
        rf_timing_stats = {
            'total_rf_time': total_rf_time,
            'average_rf_time': np.mean(rf_times) if rf_times else 0,
            'min_rf_time': np.min(rf_times) if rf_times else 0,
            'max_rf_time': np.max(rf_times) if rf_times else 0,
            'comparisons_per_second': successful_comparisons / total_rf_time if total_rf_time > 0 else 0,
            'individual_rf_times': rf_times,
            'comparisons_made': comparisons_made,
            'successful_comparisons': successful_comparisons,
            'failed_comparisons': comparisons_made - successful_comparisons,
            'processes_used': n_processes,
            'parallelization_efficiency': (sum(rf_times) / total_rf_time) if total_rf_time > 0 else 0
        }
        
        print(f"RF computations: {successful_comparisons}/{comparisons_made} successful")
        if rf_times:
            print(f"Average RF computation time: {rf_timing_stats['average_rf_time']:.6f} seconds")
            print(f"RF computation rate: {rf_timing_stats['comparisons_per_second']:.1f} comparisons/sec")
            print(f"Parallelization efficiency: {rf_timing_stats['parallelization_efficiency']:.2f}x")
        
        return rf_matrix, rf_timing_stats
    
    def compute_rf_distances_from_database(self, 
                                         file_source: str,
                                         sample_size: int,
                                         strategy: str = 'random',
                                         use_last_trees: bool = False,
                                         n_processes: int = 4) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Complete pipeline: sample trees from database and compute RF distances.
        
        Returns RF matrix, tree records, and detailed timing statistics.
        """
        # Sample trees from database
        tree_records = self.sample_trees_from_database(file_source, sample_size, strategy, use_last_trees)
        
        if not tree_records:
            return np.array([]), [], {}
        
        # Convert to DendroPy objects
        dendropy_trees, parse_timing = self.convert_to_dendropy_trees(tree_records)
        
        if not dendropy_trees:
            return np.array([]), tree_records, parse_timing
        
        # Compute RF distances
        rf_matrix, rf_timing = self.compute_pairwise_rf_distances(dendropy_trees, n_processes)
        
        # Combine timing statistics
        combined_timing = {
            'parse_timing': parse_timing,
            'rf_timing': rf_timing,
            'total_pipeline_time': parse_timing.get('total_parse_time', 0) + rf_timing.get('total_rf_time', 0)
        }
        
        return rf_matrix, tree_records, combined_timing
    
    def get_database_stats(self, file_source: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics for tree files."""
        stats = self.db_manager.get_database_stats()

        if file_source:
            tree_count = stats.get('trees_per_file', {}).get(file_source, 0)
            return {
                'file_source': file_source,
                'tree_count': tree_count
            }
        else:
            trees_per_file = stats.get('trees_per_file', {})
            return {
                'total_trees': stats.get('total_trees', 0),
                'files': [
                    {'file_source': fs, 'tree_count': tc}
                    for fs, tc in sorted(trees_per_file.items(), key=lambda x: -x[1])
                ]
            }
    
    def clear_taxon_namespace(self):
        """Clear the shared taxon namespace for memory cleanup."""
        self._taxon_namespace = None


def benchmark_rf_computation(file_source: str, 
                           sample_sizes: List[int],
                           strategy: str = 'random',
                           n_processes: int = 4) -> Dict[str, Any]:
    """Benchmark RF computation performance across different sample sizes."""
    calculator = RobinsonFouldsCalculator()
    results = []
    
    for sample_size in sample_sizes:
        print(f"Benchmarking RF computation for {sample_size} trees...")
        
        start_time = time.time()
        
        rf_matrix, tree_records, timing_stats = calculator.compute_rf_distances_from_database(
            file_source=file_source,
            sample_size=sample_size,
            strategy=strategy,
            n_processes=n_processes
        )
        
        end_time = time.time()
        
        computation_time = end_time - start_time
        trees_processed = len(tree_records)
        comparisons_made = (trees_processed * (trees_processed - 1)) // 2
        
        # Extract detailed timing information
        parse_timing = timing_stats.get('parse_timing', {})
        rf_timing = timing_stats.get('rf_timing', {})
        tree_sizes = parse_timing.get('tree_sizes', [])
        
        result = {
            'sample_size': sample_size,
            'trees_processed': trees_processed,
            'computation_time': computation_time,
            'comparisons_made': comparisons_made,
            'comparisons_per_second': comparisons_made / computation_time if computation_time > 0 else 0,
            'matrix_shape': rf_matrix.shape,
            'mean_rf_distance': np.nanmean(rf_matrix[rf_matrix > 0]) if rf_matrix.size > 0 else 0,
            'max_rf_distance': np.nanmax(rf_matrix) if rf_matrix.size > 0 else 0,
            'successful_comparisons': np.sum(~np.isnan(rf_matrix[rf_matrix > 0])) if rf_matrix.size > 0 else 0,
            
            # Detailed timing breakdown
            'parse_time': parse_timing.get('total_parse_time', 0),
            'rf_time': rf_timing.get('total_rf_time', 0),
            'trees_per_second_parsing': parse_timing.get('trees_per_second', 0),
            'avg_parse_time_per_tree': parse_timing.get('average_parse_time', 0),
            'avg_rf_time_per_comparison': rf_timing.get('average_rf_time', 0),
            'rf_comparisons_per_second': rf_timing.get('comparisons_per_second', 0),
            
            # Tree size statistics
            'avg_newick_length': np.mean([s['newick_length'] for s in tree_sizes]) if tree_sizes else 0,
            'avg_num_taxa': np.mean([s['num_taxa'] for s in tree_sizes]) if tree_sizes else 0,
            'avg_num_nodes': np.mean([s['num_nodes'] for s in tree_sizes]) if tree_sizes else 0,
            'avg_num_edges': np.mean([s['num_edges'] for s in tree_sizes]) if tree_sizes else 0,
            'min_newick_length': np.min([s['newick_length'] for s in tree_sizes]) if tree_sizes else 0,
            'max_newick_length': np.max([s['newick_length'] for s in tree_sizes]) if tree_sizes else 0,
            
            # Additional metrics
            'parse_time_percentage': (parse_timing.get('total_parse_time', 0) / computation_time * 100) if computation_time > 0 else 0,
            'rf_time_percentage': (rf_timing.get('total_rf_time', 0) / computation_time * 100) if computation_time > 0 else 0,
            'parallelization_efficiency': rf_timing.get('parallelization_efficiency', 0),
            'processes_used': rf_timing.get('processes_used', 1),
        }
        
        results.append(result)
        print(f"  Processed {trees_processed} trees in {computation_time:.2f}s")
        print(f"  Tree parsing: {result['parse_time']:.2f}s ({result['parse_time_percentage']:.1f}%)")
        print(f"  RF computation: {result['rf_time']:.2f}s ({result['rf_time_percentage']:.1f}%)")
        print(f"  {comparisons_made} comparisons at {result['comparisons_per_second']:.1f} comparisons/sec")
        print(f"  Using {result['processes_used']} processes, efficiency: {result['parallelization_efficiency']:.2f}x")
        if tree_sizes:
            print(f"  Average tree size: {result['avg_newick_length']:.0f} chars, {result['avg_num_taxa']:.0f} taxa")
    
    return {
        'file_source': file_source,
        'strategy': strategy,
        'results': results
    }