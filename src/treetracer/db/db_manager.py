"""DuckDB-based database manager for high-performance phylogenetic tree storage.

This module provides optimized database operations for handling massive phylogenetic
trees with large newick strings (100KB+). Features include:
- DuckDB performance tuning for large text data
- Chunked transaction management
- Cached ID generation to eliminate repeated MAX queries
- Temporary database with automatic cleanup
"""

import duckdb
import tempfile
import os
import atexit
import uuid
import json
from typing import List, Dict, Any, Optional


class TreeDatabaseManager:
    """High-performance database manager for phylogenetic tree storage.
    
    Optimized for massive tree datasets with large newick strings.
    Uses DuckDB with custom performance tuning and cached operations.
    """
    
    def __init__(self):
        self.connection = None
        self.db_path = None
        self.session_id = str(uuid.uuid4())[:8]
        
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self.connection is None:
            # Create temporary database file
            temp_dir = tempfile.gettempdir()
            self.db_path = os.path.join(temp_dir, f"treetracer_{self.session_id}.db")
            
            # Create connection
            self.connection = duckdb.connect(self.db_path)
            
            # Initialize schema
            self._initialize_schema()
            
            # Register cleanup
            atexit.register(self.cleanup)
            
        return self.connection
    
    def _initialize_schema(self):
        """Create the trees table and indexes."""
        # DuckDB memory-optimized tuning for phylogenetic trees
        try:
            # Memory-efficient settings
            self.connection.execute("SET memory_limit = '512MB'")                  # Reasonable memory limit
            self.connection.execute("SET threads = 1")                             # Optimal for your system
            self.connection.execute("SET preserve_insertion_order = true")         # Faster for sequential inserts
            
            # Performance optimizations
            self.connection.execute("SET enable_progress_bar = false")             # No progress overhead
            self.connection.execute("SET enable_profiling = 'no_output'")          # Disable profiling output
            self.connection.execute("SET checkpoint_threshold = '256MB'")           # More frequent, smaller checkpoints
            
            # Memory-efficient optimizations
            #self.connection.execute("SET force_compression = 'auto'")              # Enable compression to save memory
            self.connection.execute("SET temp_directory = '/tmp'")                 # Fast temp location
            #self.connection.execute("SET enable_object_cache = true")             # Disable aggressive caching
            
            print("✓ Database tuned for memory-efficient phylogenetic tree storage")
        except Exception as e:
            print(f"⚠️ Some tuning parameters not supported: {e}")
        
        schema_sql = """
        CREATE TABLE IF NOT EXISTS trees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            newick TEXT NOT NULL,
            file_source TEXT NOT NULL,
            group_name TEXT,
            metadata JSON
        );
        """
        
        self.connection.execute(schema_sql)
    
    
    def insert_trees_batch_raw(self, raw_trees_data: List[tuple]) -> int:
        """Insert trees with raw tuple data for maximum performance.
        
        Uses cached ID generation to avoid repeated MAX(id) queries,
        which was a major performance bottleneck.
        
        Args:
            raw_trees_data: List of tuples (name, newick, file_source, group_name, metadata)
            
        Returns:
            Number of trees inserted
        """
        if not raw_trees_data:
            return 0
            
        conn = self.get_connection()
        
        # Use cached ID generation to avoid repeated MAX queries
        insert_sql = """
        INSERT INTO trees (id, name, newick, file_source, group_name, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        # Initialize or use cached max ID to eliminate database queries
        if not hasattr(self, '_current_max_id'):
            max_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) FROM trees").fetchone()
            self._current_max_id = max_id_result[0] if max_id_result else 0
        
        # Generate sequential IDs from cached value
        insert_data = []
        for name, newick, file_source, group_name, metadata in raw_trees_data:
            self._current_max_id += 1
            # Convert metadata dict to JSON string for DuckDB storage
            metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata
            insert_data.append((
                self._current_max_id,
                name, newick, file_source, group_name, metadata_json
            ))
        
        # Execute batch insert (transactions handled at streaming level)
        conn.executemany(insert_sql, insert_data)
        return len(raw_trees_data)
    
    def get_trees_sample(self, 
                        filters: Optional[Dict[str, Any]] = None, 
                        limit: int = 500,
                        strategy: str = 'random') -> List[Dict[str, Any]]:
        """Get a sample of trees using various sampling strategies.
        
        Supports random, uniform, and stratified sampling with optional filtering.
        Optimized for fast sampling from large datasets.
        
        Args:
            filters: Optional filters - 'file_source' and/or 'group_name'
            limit: Maximum number of trees to return
            strategy: Sampling strategy - 'random', 'uniform', or 'stratified'
            
        Returns:
            List of tree dictionaries with keys: id, name, newick, file_source, group_name, metadata
        """
        conn = self.get_connection()
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if filters:
            if 'file_source' in filters:
                where_conditions.append("file_source = ?")
                params.append(filters['file_source'])
            if 'group_name' in filters:
                where_conditions.append("group_name = ?")
                params.append(filters['group_name'])
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Build sampling query based on strategy
        if strategy == 'random':
            order_clause = "ORDER BY RANDOM()"
        elif strategy == 'uniform':
            # For uniform sampling, we need to know total count first
            count_sql = f"SELECT COUNT(*) FROM trees {where_clause}"
            total_count = conn.execute(count_sql, params).fetchone()[0]
            if total_count <= limit:
                order_clause = "ORDER BY id"
            else:
                # Sample every nth tree
                step = max(1, total_count // limit)
                order_clause = f"ORDER BY id"
                where_conditions.append(f"(id % {step}) = 0")
                where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        else:  # stratified
            order_clause = "ORDER BY group_name, RANDOM()"
            
        query = f"""
        SELECT id, name, newick, file_source, group_name, metadata
        FROM trees 
        {where_clause}
        {order_clause}
        LIMIT ?
        """
        
        params.append(limit)
        result = conn.execute(query, params).fetchall()
        
        # Convert to list of dictionaries and parse JSON metadata
        columns = ['id', 'name', 'newick', 'file_source', 'group_name', 'metadata']
        trees = []
        for row in result:
            tree_dict = dict(zip(columns, row))
            # Parse JSON metadata back to dict
            if tree_dict['metadata']:
                try:
                    tree_dict['metadata'] = json.loads(tree_dict['metadata']) if isinstance(tree_dict['metadata'], str) else tree_dict['metadata']
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, keep as is
                    pass
            trees.append(tree_dict)
        return trees
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored trees."""
        conn = self.get_connection()
        
        stats = {}
        
        # Total tree count
        total_trees = conn.execute("SELECT COUNT(*) FROM trees").fetchone()[0]
        stats['total_trees'] = total_trees
        
        # Trees per file
        file_counts = conn.execute("""
            SELECT file_source, COUNT(*) as count 
            FROM trees 
            GROUP BY file_source
        """).fetchall()
        stats['trees_per_file'] = dict(file_counts)
        
        # Trees per group
        group_counts = conn.execute("""
            SELECT group_name, COUNT(*) as count 
            FROM trees 
            WHERE group_name IS NOT NULL
            GROUP BY group_name
        """).fetchall()
        stats['trees_per_group'] = dict(group_counts)
        
        return stats
    
    def clear_trees(self, file_source: Optional[str] = None) -> None:
        """Clear trees from database with optional file filtering.
        
        Args:
            file_source: If provided, only clear trees from this file source.
                        If None, clear all trees from database.
        """
        conn = self.get_connection()
        
        if file_source:
            conn.execute("DELETE FROM trees WHERE file_source = ?", [file_source])
        else:
            conn.execute("DELETE FROM trees")
            # Reset cached ID counter when clearing all data
            if hasattr(self, '_current_max_id'):
                delattr(self, '_current_max_id')
    
    def cleanup(self) -> None:
        """Clean up database connection and remove temporary file.
        
        Automatically called on program exit via atexit registration.
        Safe to call multiple times.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
        
        if self.db_path and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass  # File might be in use or already deleted


# Global instance
_db_manager = None

def get_db_manager() -> TreeDatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = TreeDatabaseManager()
    return _db_manager