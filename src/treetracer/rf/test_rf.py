"""Test Robinson-Foulds distance computation using test.trees file."""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from treetracer.rf.rf import RobinsonFouldsCalculator
from treetracer.db.process_trees import process_nexus_trees_streaming
from treetracer.db.tree_service import TreeService


def test_rf_computation():
    """Test RF distance computation with test.trees file."""
    print("Testing Robinson-Foulds distance computation...")
    
    # Path to test file
    test_file_path = os.path.join(os.path.dirname(__file__), 'test.trees')
    
    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found at {test_file_path}")
        return False
    
    print(f"Loading trees from: {test_file_path}")
    
    # Load trees into database
    file_source = "test_rf_group"
    
    try:
        # Get database manager
        from treetracer.db.tree_manager import get_tree_manager
        db_manager = get_tree_manager()
        
        # Process trees from nexus file
        tree_count = process_nexus_trees_streaming(
            nexus_file=test_file_path,
            db_manager=db_manager,
            file_source=file_source
        )
        
        print(f"Loaded {tree_count} trees into database")
        
        if tree_count == 0:
            print("No trees found in test file")
            return False
        
        # Initialize RF calculator
        calculator = RobinsonFouldsCalculator()
        
        # Get database stats
        stats = calculator.get_database_stats(file_source)
        print(f"Database stats: {stats}")
        
        # Sample 100 trees (or all if less than 100)
        sample_size = min(100, tree_count)
        print(f"Sampling {sample_size} trees for RF computation...")
        
        # Compute RF distances
        rf_matrix, tree_records = calculator.compute_rf_distances_from_database(
            file_source=file_source,
            sample_size=sample_size,
            strategy='random'
        )
        
        print(f"RF distance matrix shape: {rf_matrix.shape}")
        print(f"Trees processed: {len(tree_records)}")
        
        if rf_matrix.size > 0:
            # Calculate statistics
            valid_distances = rf_matrix[rf_matrix > 0]
            if len(valid_distances) > 0:
                print(f"Mean RF distance: {np.mean(valid_distances):.2f}")
                print(f"Max RF distance: {np.max(valid_distances):.2f}")
                print(f"Min RF distance: {np.min(valid_distances):.2f}")
                print(f"Std RF distance: {np.std(valid_distances):.2f}")
            
            # Check for NaN values
            nan_count = np.sum(np.isnan(rf_matrix))
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values in RF matrix")
            
            # Print sample of the matrix
            print("\nSample of RF distance matrix (first 5x5):")
            print(rf_matrix[:5, :5])
        
        print("RF computation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during RF computation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the RF test."""
    success = test_rf_computation()
    
    if success:
        print("\n✓ RF computation test passed!")
        sys.exit(0)
    else:
        print("\n✗ RF computation test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()