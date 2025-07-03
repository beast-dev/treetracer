#!/usr/bin/env python3
"""
Test script for DuckDB tree storage functionality.
Tests the db_manager and tree_service with the test.trees file.
Includes memory usage monitoring (requires psutil: pip install psutil).
"""

import os
import sys
import time
from typing import Dict, Any

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - memory monitoring disabled")

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.tree_service import get_tree_service
from db.db_manager import get_db_manager


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def get_memory_usage() -> float:
    """Get current process memory usage in MB for performance monitoring.
    
    Returns:
        Memory usage in megabytes, or 0.0 if psutil not available
    """
    if not PSUTIL_AVAILABLE:
        return 0.0
    
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def test_database_initialization():
    """Test database manager initialization."""
    print_section("DATABASE INITIALIZATION TEST")
    
    db_manager = get_db_manager()
    connection = db_manager.get_connection()
    
    print(f"Database connection established")
    print(f"Session ID: {db_manager.session_id}")
    print(f"Database path: {db_manager.db_path}")
    
    return db_manager


def test_nexus_file_loading():
    """Test loading the nexus file with memory monitoring."""
    print_section("NEXUS FILE LOADING TEST")
    
    # Get the test file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "test.trees")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return None
    
    print(f"Loading file: {test_file}")
    file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Memory monitoring
    initial_memory = get_memory_usage()
    if PSUTIL_AVAILABLE:
        print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Load the file
    tree_service = get_tree_service()
    start_time = time.time()
    
    result = tree_service.load_nexus_file(test_file, "test.trees")
    
    load_time = time.time() - start_time
    
    # Memory after loading
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    if result['success']:
        trees_loaded = result['trees_loaded']
        loading_rate = trees_loaded / load_time
        
        print(f"✓ Successfully loaded {trees_loaded} trees")
        print(f"✓ Loading time: {load_time:.2f} seconds")
        print(f"✓ Loading rate: {loading_rate:.0f} trees/second")
        print(f"✓ Data throughput: {file_size_mb / load_time:.1f} MB/second")
        
        if PSUTIL_AVAILABLE:
            print(f"✓ Final memory usage: {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
            
            # Memory efficiency analysis
            if trees_loaded > 0:
                mb_per_tree = memory_increase / trees_loaded
                print(f"✓ Memory per tree: {mb_per_tree:.3f} MB")
                
                if mb_per_tree < 0.1:
                    print("🎯 EXCELLENT: Memory efficient")
                elif mb_per_tree < 0.5:
                    print("✅ GOOD: Reasonable memory usage")
                else:
                    print("⚠️  HIGH: Consider memory optimizations")
        
        return result
    else:
        print(f"❌ Loading failed: {result['error']}")
        return None


def test_database_statistics(load_result: Dict[str, Any]):
    """Test database statistics and queries."""
    print_section("DATABASE STATISTICS TEST")
    
    tree_service = get_tree_service()
    summary = tree_service.get_database_summary()
    
    print(f"Total trees in database: {summary['total_trees']}")
    print(f"Number of files: {summary['number_of_files']}")
    print(f"Number of groups: {summary['number_of_groups']}")
    
    print_subsection("Files in Database")
    for file_name, count in summary['trees_per_file'].items():
        print(f"  {file_name}: {count} trees")
    
    if summary['trees_per_group']:
        print_subsection("Groups in Database")
        for group_name, count in list(summary['trees_per_group'].items())[:10]:  # Show first 10
            print(f"  {group_name}: {count} trees")
        if len(summary['trees_per_group']) > 10:
            print(f"  ... and {len(summary['trees_per_group']) - 10} more groups")


def test_sampling_strategies():
    """Test different sampling strategies."""
    print_section("SAMPLING STRATEGIES TEST")
    
    tree_service = get_tree_service()
    sample_sizes = [10, 50, 100]
    strategies = ['random', 'uniform']
    
    for strategy in strategies:
        print_subsection(f"Testing {strategy.upper()} sampling")
        
        for sample_size in sample_sizes:
            start_time = time.time()
            
            sample_result = tree_service.get_sample_for_analysis(
                sample_size=sample_size,
                strategy=strategy
            )
            
            sample_time = time.time() - start_time
            
            if sample_result['ready_for_analysis']:
                actual_size = sample_result['sample_size']
                print(f"✓ {strategy} sample of {actual_size} trees (requested {sample_size}) in {sample_time:.3f}s")
                
                # Check for duplicates
                tree_ids = [tree['id'] for tree in sample_result['trees']]
                unique_ids = set(tree_ids)
                if len(unique_ids) == len(tree_ids):
                    print(f"  ✓ No duplicate trees in sample")
                else:
                    print(f"  ⚠️  Found {len(tree_ids) - len(unique_ids)} duplicate trees")
            else:
                print(f"❌ {strategy} sampling failed")


def test_stratified_sampling():
    """Test stratified sampling if multiple groups exist."""
    print_section("STRATIFIED SAMPLING TEST")
    
    tree_service = get_tree_service()
    summary = tree_service.get_database_summary()
    
    if len(summary['trees_per_group']) < 2:
        print("Skipping stratified sampling test (need multiple groups)")
        return
    
    # Get the top groups by tree count
    sorted_groups = sorted(
        summary['trees_per_group'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]  # Take top 3 groups
    
    group_names = [group[0] for group in sorted_groups]
    print(f"Testing stratified sampling with groups: {group_names}")
    
    start_time = time.time()
    sample_result = tree_service.get_sample_for_analysis(
        group_names=group_names,
        sample_size=150,  # 50 per group on average
        strategy='stratified'
    )
    sample_time = time.time() - start_time
    
    if sample_result['ready_for_analysis']:
        print(f"Stratified sample of {sample_result['sample_size']} trees in {sample_time:.3f}s")
        
        if 'strata_info' in sample_result:
            print("  Strata breakdown:")
            for stratum in sample_result['strata_info']:
                print(f"    {stratum['stratum']}: {stratum['sampled_trees']}/{stratum['total_trees']} trees")
    else:
        print("L Stratified sampling failed")


def test_analysis_preparation():
    """Test preparation functions for RF and MDS analysis."""
    print_section("ANALYSIS PREPARATION TEST")
    
    tree_service = get_tree_service()
    
    # Get a small sample for testing
    sample_result = tree_service.get_sample_for_analysis(sample_size=5)
    
    if not sample_result['ready_for_analysis']:
        print("L Could not get sample for analysis preparation test")
        return
    
    trees = sample_result['trees']
    print(f"Using sample of {len(trees)} trees for preparation test")
    
    # Test RF preparation
    print_subsection("RF Analysis Preparation")
    newick_strings = tree_service.prepare_trees_for_rf_analysis(trees)
    print(f"Prepared {len(newick_strings)} newick strings for RF analysis")
    
    # Show first newick string (truncated)
    if newick_strings:
        newick_sample = newick_strings[0]
        if len(newick_sample) > 100:
            newick_sample = newick_sample[:100] + "..."
        print(f"  Sample newick: {newick_sample}")
    
    # Test MDS preparation
    print_subsection("MDS Analysis Preparation")
    mds_metadata = tree_service.prepare_metadata_for_mds(trees)
    print(f"Prepared metadata for {len(mds_metadata['tree_names'])} trees")
    print(f"  Tree names: {mds_metadata['tree_names'][:3]}...")
    print(f"  File sources: {set(mds_metadata['file_sources'])}")
    print(f"  Groups: {set(mds_metadata['group_names'])}")


def test_cleanup():
    """Test database cleanup with memory monitoring."""
    print_section("CLEANUP TEST")
    
    db_manager = get_db_manager()
    db_path = db_manager.db_path
    
    # Get final stats and memory usage
    stats = db_manager.get_database_stats()
    final_memory = get_memory_usage()
    
    print(f"Final Database Statistics:")
    print(f"  Total trees: {stats.get('total_trees', 0):,}")
    print(f"  Number of files: {len(stats.get('trees_per_file', {}))}")
    print(f"  Number of groups: {len(stats.get('trees_per_group', {}))}")
    
    # Database file size
    if db_path and os.path.exists(db_path):
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"  Database file size: {db_size_mb:.1f} MB")
    
    if PSUTIL_AVAILABLE:
        print(f"  Final memory usage: {final_memory:.1f} MB")
    
    print(f"Database file before cleanup: {os.path.exists(db_path) if db_path else 'N/A'}")
    
    # Test cleanup
    db_manager.cleanup()
    
    if db_path:
        file_exists_after = os.path.exists(db_path)
        print(f"Database file after cleanup: {file_exists_after}")
        
        if not file_exists_after:
            print("✓ Database file successfully cleaned up")
        else:
            print("⚠️  Database file still exists (might be expected on some systems)")
    else:
        print("✓ Cleanup completed")


def main():
    """Run all tests with memory monitoring."""
    print("TreeTracer DuckDB Functionality Test")
    print("=" * 60)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initial memory monitoring
    initial_memory = get_memory_usage()
    if PSUTIL_AVAILABLE:
        print(f"Initial system memory: {initial_memory:.1f} MB")
        print("Memory monitoring enabled")
    else:
        print("Memory monitoring disabled (psutil not available)")
    
    try:
        # Test 1: Database initialization
        db_manager = test_database_initialization()
        
        # Test 2: Load nexus file
        load_result = test_nexus_file_loading()
        if not load_result:
            print("\nL Cannot continue tests without successful file loading")
            return 1
        
        # Test 3: Database statistics
        test_database_statistics(load_result)
        
        # Test 4: Sampling strategies
        test_sampling_strategies()
        
        # Test 5: Stratified sampling
        test_stratified_sampling()
        
        # Test 6: Analysis preparation
        test_analysis_preparation()
        
        # Test 7: Cleanup
        test_cleanup()
        
        print_section("ALL TESTS COMPLETED")
        print("✓ DuckDB functionality tests passed")
        print(f"Test finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Final memory report
        if PSUTIL_AVAILABLE:
            final_memory = get_memory_usage()
            total_memory_increase = final_memory - initial_memory
            print(f"Final memory usage: {final_memory:.1f} MB (+{total_memory_increase:.1f} MB)")
        
        return 0
        
    except Exception as e:
        print(f"\nL Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Inform users about optional dependency
    if not PSUTIL_AVAILABLE:
        print("💡 Tip: Install psutil for memory monitoring: pip install psutil")
        print()
    
    sys.exit(main())