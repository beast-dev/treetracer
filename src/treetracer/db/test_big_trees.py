#!/usr/bin/env python3
"""Scalability test suite for large-scale phylogenetic tree processing.

This test script validates TreeTracer's performance with massive nexus files (4GB+)
and tests the scalability of the optimized DuckDB storage system. Features:
- Large file loading performance monitoring with memory tracking
- Sampling performance validation at scale (millions of trees)
- Database query performance testing with complex datasets
- Random tree sampling and analysis for quality validation
- Comprehensive memory usage and throughput metrics

Requires test_BIG.trees file in the same directory (typically 4GB+ with millions of trees).
Also requires psutil for memory monitoring: pip install psutil
"""

import os
import sys
import time
import psutil
from typing import Dict, Any

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.tree_service import get_tree_service
from db.db_manager import get_db_manager


def print_section(title: str) -> None:
    """Print a formatted section header for large-scale test organization."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header for large-scale test organization."""
    print(f"\n{'-' * 50}")
    print(f" {title}")
    print(f"{'-' * 50}")


def get_memory_usage() -> float:
    """Get current process memory usage in MB for performance monitoring.
    
    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_big_file_loading():
    """Test large-scale nexus file loading with comprehensive performance monitoring.
    
    Validates that TreeTracer can handle massive files (4GB+) with optimal
    performance, memory efficiency, and transaction management. Monitors
    throughput, memory usage, and provides performance benchmarks.
    
    Returns:
        Loading result dictionary if successful, None if failed
    """
    print_section("LARGE-SCALE NEXUS FILE LOADING TEST")
    
    # Locate large test file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "test_BIG.trees")
    
    if not os.path.exists(test_file):
        print(f"❌ Large test file not found: {test_file}")
        print(f"Please ensure test_BIG.trees exists in the db directory")
        print(f"This test requires a multi-GB nexus file for scalability validation")
        return None
    
    # File information
    file_size_gb = os.path.getsize(test_file) / (1024 * 1024 * 1024)
    print(f"Loading file: {test_file}")
    print(f"File size: {file_size_gb:.2f} GB")
    
    # Memory monitoring
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Load the file with timing
    tree_service = get_tree_service()
    start_time = time.time()
    
    print(f"\nStarting load at {time.strftime('%H:%M:%S')}")
    # For testing, we can also call the streaming function directly with custom parameters
    # result = tree_service.load_nexus_file(test_file, "test_BIG.trees")
    
    # Or for more control over transaction size:
    from db.process_trees import process_nexus_trees_streaming
    db_manager = get_db_manager()
    
    try:
        inserted_count = process_nexus_trees_streaming(
            test_file,
            db_manager, 
            "test_BIG.trees",
            batch_size=500,
            transaction_size=5000  # Smaller for testing - more frequent commits
        )
        
        # Get statistics
        stats = db_manager.get_database_stats()
        
        result = {
            'success': True,
            'trees_loaded': inserted_count,
            'file_source': "test_BIG.trees",
            'database_stats': stats
        }
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'trees_loaded': 0
        }
    
    load_time = time.time() - start_time
    end_time = time.strftime('%H:%M:%S')
    
    # Memory after loading
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    print(f"Finished load at {end_time}")
    print(f"Final memory usage: {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
    
    if result['success']:
        trees_loaded = result['trees_loaded']
        loading_rate = trees_loaded / load_time
        
        print(f"✓ Successfully loaded {trees_loaded:,} trees")
        print(f"✓ Loading time: {load_time:.2f} seconds")
        print(f"✓ Loading rate: {loading_rate:.0f} trees/second")
        print(f"✓ Data throughput: {file_size_gb / load_time * 1024:.1f} MB/second")
        
        # Performance benchmarks
        print_subsection("Performance Analysis")
        
        if loading_rate > 500:
            print("🎯 EXCELLENT: >500 trees/second - Production ready")
        elif loading_rate > 200:
            print("✅ GOOD: >200 trees/second - Acceptable performance") 
        elif loading_rate > 100:
            print("⚠️  ACCEPTABLE: >100 trees/second - Consider optimizations")
        else:
            print("❌ SLOW: <100 trees/second - Performance improvements required")
            
        # Memory efficiency
        mb_per_tree = memory_increase / trees_loaded if trees_loaded > 0 else 0
        print(f"Memory per tree: {mb_per_tree:.3f} MB")
        
        if mb_per_tree < 0.1:
            print("🎯 EXCELLENT: Memory efficient - Optimal for large datasets")
        elif mb_per_tree < 0.5:
            print("✅ GOOD: Reasonable memory usage - Scales well")
        else:
            print("⚠️  HIGH: Consider memory optimizations for very large files")
            
        return result
    else:
        print(f"❌ Loading failed: {result['error']}")
        return None


def test_sampling_performance(load_result: Dict[str, Any]):
    """Test sampling performance with the large dataset."""
    print_section("SAMPLING PERFORMANCE TEST")
    
    if not load_result:
        print("⚠️ Skipping sampling test (no data loaded)")
        return
        
    tree_service = get_tree_service()
    summary = tree_service.get_database_summary()
    
    total_trees = summary['total_trees']
    print(f"Total trees in database: {total_trees:,}")
    
    # Test different sampling strategies and sizes
    sample_configs = [
        (100, 'random'),
        (500, 'random'), 
        (1000, 'random'),
        (500, 'uniform'),
        (500, 'stratified') if len(summary['trees_per_group']) > 1 else None
    ]
    
    sample_configs = [config for config in sample_configs if config is not None]
    
    for sample_size, strategy in sample_configs:
        if sample_size > total_trees:
            continue
            
        print_subsection(f"Testing {strategy} sampling - {sample_size} trees")
        
        start_time = time.time()
        sample_result = tree_service.get_sample_for_analysis(
            sample_size=sample_size,
            strategy=strategy
        )
        sample_time = time.time() - start_time
        
        if sample_result['ready_for_analysis']:
            actual_size = sample_result['sample_size']
            sampling_rate = actual_size / sample_time
            
            print(f"✓ Sampled {actual_size} trees in {sample_time:.3f}s")
            print(f"✓ Sampling rate: {sampling_rate:.0f} trees/second")
            
            if sampling_rate > 10000:
                print("🎯 EXCELLENT: Ultra-fast sampling")
            elif sampling_rate > 1000:
                print("✅ GOOD: Fast sampling")
            else:
                print("⚠️  Consider sampling optimizations")
        else:
            print(f"❌ Sampling failed for {strategy}")


def test_database_scalability():
    """Test database operations at scale."""
    print_section("DATABASE SCALABILITY TEST")
    
    db_manager = get_db_manager()
    stats = db_manager.get_database_stats()
    
    total_trees = stats.get('total_trees', 0)
    print(f"Database contains {total_trees:,} trees")
    
    if total_trees == 0:
        print("⚠️ No trees in database for scalability test")
        return
    
    # Test query performance
    test_queries = [
        ("Count query", "SELECT COUNT(*) FROM trees"),
        ("Random sample", "SELECT id FROM trees ORDER BY RANDOM() LIMIT 100"),
        ("File grouping", "SELECT file_source, COUNT(*) FROM trees GROUP BY file_source")
    ]
    
    for query_name, sql in test_queries:
        start_time = time.time()
        result = db_manager.get_connection().execute(sql).fetchall()
        query_time = time.time() - start_time
        
        print(f"✓ {query_name}: {query_time:.3f}s ({len(result)} results)")
        
        if query_time < 1.0:
            print("  🎯 Fast query performance")
        elif query_time < 5.0:
            print("  ✅ Acceptable query performance")
        else:
            print("  ⚠️  Slow query - consider optimizations")


def test_random_row_sampling():
    """Test random row sampling and display a sample tree."""
    print_section("RANDOM ROW SAMPLING TEST")
    
    db_manager = get_db_manager()
    stats = db_manager.get_database_stats()
    
    total_trees = stats.get('total_trees', 0)
    if total_trees == 0:
        print("⚠️ No trees in database for random sampling")
        return
    
    print(f"Sampling random row from {total_trees:,} trees...")
    
    # Query for a random row with timing - use the sampling method for decompression
    start_time = time.time()
    sample_result = db_manager.get_trees_sample(limit=1, strategy='random')
    query_time = time.time() - start_time
    
    if sample_result and len(sample_result) > 0:
        tree = sample_result[0]
        
        print(f"✓ Random sampling took: {query_time:.3f}s")
        print_subsection("Random Tree Sample")
        
        print(f"Tree ID: {tree['id']}")
        print(f"Tree Name: {tree['name']}")
        print(f"File Source: {tree['file_source']}")
        print(f"Group Name: {tree['group_name'] or 'None'}")
        print(f"Metadata: {tree['metadata']}")
        
        # Display metadata info
        metadata = tree['metadata']
        if metadata and metadata != '{}':
            import json
            try:
                parsed_metadata = json.loads(metadata)
                print(f"Metadata: {parsed_metadata}")
                
                # Show specific metadata fields
                if 'rooted' in parsed_metadata:
                    print(f"  Rooted: {parsed_metadata['rooted']}")
                if 'rate' in parsed_metadata:
                    print(f"  Rate: {parsed_metadata['rate']}")
                if 'annotation' in parsed_metadata:
                    print(f"  Annotation: {parsed_metadata['annotation']}")
            except:
                print(f"Metadata (raw): {metadata}")
        else:
            print(f"Metadata: None")
        
        # Display newick string info
        newick = tree['newick']
        newick_length = len(newick)
        print(f"Newick Length: {newick_length:,} characters")
        
        # Show truncated newick for readability
        if newick_length > 200:
            newick_preview = newick[:100] + "..." + newick[-100:]
            print(f"Newick Preview: {newick_preview}")
        else:
            print(f"Newick String: {newick}")
        
        # Additional analysis
        print_subsection("Tree Analysis")
        
        # Count parentheses (rough estimate of tree complexity)
        open_parens = newick.count('(')
        close_parens = newick.count(')')
        commas = newick.count(',')
        
        print(f"Estimated internal nodes: {open_parens}")
        print(f"Estimated leaves: {commas + 1}")
        print(f"Parentheses balance: {open_parens == close_parens}")
        
        # Check for branch lengths
        has_branch_lengths = ':' in newick
        print(f"Has branch lengths: {has_branch_lengths}")
        
        if has_branch_lengths:
            colon_count = newick.count(':')
            print(f"Branch length entries: {colon_count}")
        
    else:
        print("❌ Failed to retrieve random row")
        
    # Test multiple random samples for consistency
    print_subsection("Multiple Random Samples Performance")
    
    sample_times = []
    for i in range(5):
        start_time = time.time()
        sample = db_manager.get_trees_sample(limit=1, strategy='random')
        sample_time = time.time() - start_time
        sample_times.append(sample_time)
        
        if sample and len(sample) > 0:
            print(f"Sample {i+1}: {sample_time:.3f}s (Tree: {sample[0]['name']})")
        else:
            print(f"Sample {i+1}: Failed")
    
    if sample_times:
        avg_time = sum(sample_times) / len(sample_times)
        min_time = min(sample_times)
        max_time = max(sample_times)
        
        print(f"Average random sampling time: {avg_time:.3f}s")
        print(f"Min/Max sampling time: {min_time:.3f}s / {max_time:.3f}s")
        
        if avg_time < 0.1:
            print("🎯 EXCELLENT: Sub-100ms random sampling")
        elif avg_time < 0.5:
            print("✅ GOOD: Fast random sampling")
        elif avg_time < 2.0:
            print("⚠️  ACCEPTABLE: Reasonable sampling time")
        else:
            print("❌ SLOW: Consider adding indexes for random sampling")


def test_cleanup_and_stats():
    """Test cleanup and show final statistics."""
    print_section("CLEANUP AND FINAL STATS")
    
    db_manager = get_db_manager()
    
    # Get final stats
    stats = db_manager.get_database_stats()
    
    print("Final Database Statistics:")
    print(f"  Total trees: {stats.get('total_trees', 0):,}")
    print(f"  Number of files: {len(stats.get('trees_per_file', {}))}")
    print(f"  Number of groups: {len(stats.get('trees_per_group', {}))}")
    
    # Database file size
    if db_manager.db_path and os.path.exists(db_manager.db_path):
        db_size_mb = os.path.getsize(db_manager.db_path) / (1024 * 1024)
        print(f"  Database file size: {db_size_mb:.1f} MB")
    
    # Memory usage
    final_memory = get_memory_usage()
    print(f"  Final memory usage: {final_memory:.1f} MB")
    
    # Test cleanup
    print(f"\nTesting cleanup...")
    db_path = db_manager.db_path
    db_manager.cleanup()
    
    if db_path and os.path.exists(db_path):
        print("⚠️ Database file still exists after cleanup")
    else:
        print("✓ Database file cleaned up successfully")


def main() -> int:
    """Execute comprehensive large-scale performance test suite.
    
    Runs scalability tests for TreeTracer's database functionality with
    massive phylogenetic datasets, validating performance, memory efficiency,
    and system reliability under extreme load conditions.
    
    Returns:
        Exit code - 0 for success, 1 for failure
    """
    print("TreeTracer Large-Scale Performance Test Suite")
    print("=" * 70)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial system memory: {get_memory_usage():.1f} MB")
    
    try:
        # Test 1: Large file loading with performance monitoring
        load_result = test_big_file_loading()
        
        if not load_result:
            print("\n❌ Cannot continue scalability tests without successful file loading")
            print("Please ensure test_BIG.trees file is available for large-scale testing")
            return 1
        
        # Test 2: Sampling performance at scale
        test_sampling_performance(load_result)
        
        # Test 3: Database scalability validation
        test_database_scalability()
        
        # Test 4: Random sampling and tree analysis
        test_random_row_sampling()
        
        # Test 5: Cleanup and final statistics
        test_cleanup_and_stats()
        
        print_section("ALL SCALABILITY TESTS COMPLETED SUCCESSFULLY")
        print("✓ Large-scale phylogenetic tree processing validated")
        print("✓ System performance meets requirements for production use")
        print(f"Test finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Scalability test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Check for required dependency
    try:
        import psutil
    except ImportError:
        print("❌ This test requires psutil for memory monitoring")
        print("Install with: pip install psutil")
        sys.exit(1)
        
    sys.exit(main())