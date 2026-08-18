"""
Statistical profiling functions for database columns.
Based on the BIRD paper's profiling methodology.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import re
from collections import Counter


@dataclass
class ColumnProfile:
    """Profile statistics for a single column."""
    
    # Basic info
    table_name: str
    column_name: str
    data_type: str
    
    # Counts
    total_records: int
    null_count: int
    non_null_count: int
    distinct_count: int
    
    # Shape (for data format detection)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    
    # Sample values
    top_k_values: List[tuple] = None  # [(value, count), ...] - top K frequent values
    sample_values: List[Any] = None   # Small sample for pattern detection (~100)
    indexed_values: List[Any] = field(default_factory=list)  # Large sample for LSH (up to N=10,000)
    
    # Data characteristics
    is_numeric: bool = False
    is_date_like: bool = False
    common_pattern: Optional[str] = None  # e.g., "YYYY-MM-DD", "NNN-NNN-NNNN"
    
    def __post_init__(self):
        if self.top_k_values is None:
            self.top_k_values = []
        if self.sample_values is None:
            self.sample_values = []


class ColumnProfiler:
    """Profiles individual database columns."""
    
    def __init__(self, top_k: int = 10, sample_size: int = 10000):
        """
        Initialize profiler.
        
        Args:
            top_k: Number of top frequent values to collect
            sample_size: Maximum number of distinct values to sample for LSH indexing
                        (BIRD paper uses N=10,000)
        """
        self.top_k = top_k
        self.sample_size = sample_size
    
    def profile_column(
        self,
        db,
        table_name: str,
        column_name: str,
        data_type: str,
    ) -> ColumnProfile:
        """
        Profile a single column.
        
        Args:
            db: Database connection
            table_name: Name of the table
            column_name: Name of the column
            data_type: Data type of the column
            
        Returns:
            ColumnProfile with statistics
        """
        print(f"DEBUG: Profiling {table_name}.{column_name}...", flush=True)
        # Get total records
        result = db.execute_query(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        total_records = result[0][0]
        
        # Get NULL count
        result = db.execute_query(
            f'SELECT COUNT(*) as cnt FROM "{table_name}" WHERE "{column_name}" IS NULL'
        )
        null_count = result[0][0]
        non_null_count = total_records - null_count
        
        # Get distinct count
        result = db.execute_query(
            f'SELECT COUNT(DISTINCT "{column_name}") as cnt FROM "{table_name}" '
            f'WHERE "{column_name}" IS NOT NULL'
        )
        distinct_count = result[0][0]
        
        # Initialize profile
        profile = ColumnProfile(
            table_name=table_name,
            column_name=column_name,
            data_type=data_type,
            total_records=total_records,
            null_count=null_count,
            non_null_count=non_null_count,
            distinct_count=distinct_count,
        )
        
        # If all NULL, return early
        if non_null_count == 0:
            return profile
        
        # Get top-k values (frequency ordered)
        profile.top_k_values = self._get_top_k_values(db, table_name, column_name)
        
        # Get small sample for pattern detection (100 values)
        profile.sample_values = self._get_sample_values(db, table_name, column_name, limit=100)
        
        # Get large sample for LSH indexing (up to N=10,000 distinct values)
        profile.indexed_values = self._get_indexed_values(db, table_name, column_name)
        
        # Detect data characteristics
        profile.is_numeric = self._is_numeric(profile.sample_values)
        profile.is_date_like = self._is_date_like(profile.sample_values)
        profile.common_pattern = self._detect_pattern(profile.sample_values)
        
        # Get min/max with Smart Fallback because of Oracle strickness
        if profile.is_numeric:
            # Try numeric profiling first
            min_val, max_val = self._get_numeric_range(db, table_name, column_name)
            
            if min_val is None:
                # It failed (probably ORA-01722). 
                # It's not actually numeric. Switch to text mode.
                profile.is_numeric = False
                profile.min_value, profile.max_value = self._get_string_range(
                    db, table_name, column_name
                )
            else:
                profile.min_value, profile.max_value = min_val, max_val
        else:
            # It was text from the start
            profile.min_value, profile.max_value = self._get_string_range(
                db, table_name, column_name
            )
        
        # Get length statistics for strings
        if not profile.is_numeric:
            profile.min_length, profile.max_length = self._get_length_range(
                db, table_name, column_name
            )
        
        return profile
    
    def _get_top_k_values(
            self, db, table_name: str, column_name: str
        ) -> List[tuple]:
            """Get top-k most frequent values."""
            try:
                # FIX: Used ROWNUM for older Oracle compatibility
                query = f"""
                    SELECT * FROM (
                        SELECT "{column_name}", COUNT(*) as cnt
                        FROM "{table_name}"
                        WHERE "{column_name}" IS NOT NULL
                        GROUP BY "{column_name}"
                        ORDER BY cnt DESC
                    ) WHERE ROWNUM <= {self.top_k}
                """
                results = db.execute_query(query)
                # FIX: Ensure index access (row[0], row[1]) for tuples
                return [(row[0], row[1]) for row in results]
            except Exception:
                return []
    
    def _get_sample_values(
            self, db, table_name: str, column_name: str, limit: int = 100
        ) -> List[Any]:
            """Get small sample of distinct values."""
            try:
                # FIX: Used ROWNUM for older Oracle compatibility
                query = f"""
                    SELECT * FROM (
                        SELECT DISTINCT "{column_name}"
                        FROM "{table_name}"
                        WHERE "{column_name}" IS NOT NULL
                    ) WHERE ROWNUM <= {limit}
                """
                results = db.execute_query(query)
                # FIX: Ensure index access (row[0]) for tuples
                return [row[0] for row in results]
            except Exception:
                return []
    
    def _get_indexed_values(
            self, db, table_name: str, column_name: str
        ) -> List[Any]:
            """Get up to N distinct values for LSH indexing."""
            try:
                # FIX: Used ROWNUM for older Oracle compatibility
                query = f"""
                    SELECT * FROM (
                        SELECT "{column_name}", COUNT(*) as cnt
                        FROM "{table_name}"
                        WHERE "{column_name}" IS NOT NULL
                        GROUP BY "{column_name}"
                        ORDER BY cnt DESC
                    ) WHERE ROWNUM <= {self.sample_size}
                """
                results = db.execute_query(query)
                # FIX: Ensure index access (row[0]) for tuples
                return [row[0] for row in results]
            except Exception:
                return []
    
    def _is_numeric(self, values: List[Any]) -> bool:
        """Check if values are numeric."""
        if not values:
            return False
        
        numeric_count = 0
        for val in values[:50]:  # Sample first 50
            try:
                float(str(val))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        return numeric_count / len(values[:50]) > 0.8
    
    def _is_date_like(self, values: List[Any]) -> bool:
        """Check if values look like dates."""
        if not values:
            return False
        
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        date_count = 0
        for val in values[:50]:
            val_str = str(val)
            for pattern in date_patterns:
                if re.match(pattern, val_str):
                    date_count += 1
                    break
        
        return date_count / len(values[:50]) > 0.8
    
    def _detect_pattern(self, values: List[Any]) -> Optional[str]:
        """Detect common pattern in values."""
        if not values:
            return None
        
        # Convert to strings and get first few samples
        samples = [str(v) for v in values[:20]]
        
        # Check if all same length
        lengths = [len(s) for s in samples]
        if len(set(lengths)) == 1:
            length = lengths[0]
            
            # Check character types
            all_numeric = all(s.isdigit() for s in samples)
            all_alpha = all(s.isalpha() for s in samples)
            
            if all_numeric:
                return f"{'N' * length}"  # e.g., "NNNN"
            elif all_alpha:
                return f"{'A' * length}"  # e.g., "AAAA"
            else:
                # Detect pattern like "NNN-NNN-NNNN"
                pattern = self._extract_pattern(samples[0])
                if all(self._extract_pattern(s) == pattern for s in samples):
                    return pattern
        
        return None
    
    def _extract_pattern(self, s: str) -> str:
        """Extract pattern from a string (e.g., '123-456' -> 'NNN-NNN')."""
        pattern = []
        for char in s:
            if char.isdigit():
                pattern.append('N')
            elif char.isalpha():
                pattern.append('A')
            else:
                pattern.append(char)
        return ''.join(pattern)
    
    def _get_numeric_range(
        self, db, table_name: str, column_name: str
    ) -> tuple:
        """Get min/max for numeric columns."""
        try:
            query = f"""
                SELECT 
                    MIN(CAST("{column_name}" AS NUMBER)) as min_val,
                    MAX(CAST("{column_name}" AS NUMBER)) as max_val
                FROM "{table_name}"
                WHERE "{column_name}" IS NOT NULL
            """
            result = db.execute_query(query)
            return result[0][0], result[0][1]
        except Exception:
            return None, None
    
    def _get_string_range(
        self, db, table_name: str, column_name: str
    ) -> tuple:
        """Get min/max for string columns (alphabetically)."""
        try:
            query = f"""
                SELECT 
                    MIN("{column_name}") as min_val,
                    MAX("{column_name}") as max_val
                FROM "{table_name}"
                WHERE "{column_name}" IS NOT NULL
            """
            result = db.execute_query(query)
            return result[0][0], result[0][1]
        except Exception:
            return None, None
    
    def _get_length_range(
        self, db, table_name: str, column_name: str
    ) -> tuple:
        """Get min/max length for string columns."""
        try:
            query = f"""
                SELECT 
                    MIN(LENGTH("{column_name}")) as min_len,
                    MAX(LENGTH("{column_name}")) as max_len
                FROM "{table_name}"
                WHERE "{column_name}" IS NOT NULL
            """
            result = db.execute_query(query)
            return result[0][0], result[0][1]
        except Exception:
            return None, None