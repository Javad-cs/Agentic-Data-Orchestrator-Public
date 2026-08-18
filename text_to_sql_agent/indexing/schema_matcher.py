"""
Schema-specific adapter for literal matching.
Maps literals to (table, column, value) tuples with type information.
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import re
from profiling.statistics import ColumnProfile
from .lsh_matcher import LexicalLSHMatcher, MatchResult


# SQL numeric types (normalized, uppercase)
NUMERIC_TYPES = {
    "INT", "INTEGER", "SMALLINT", "BIGINT", "TINYINT",
    "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL",
}


@dataclass
class SchemaMatch:
    """A match between a literal and a schema value."""
    table: str
    column: str
    value: str  # String representation of matched value
    score: float
    column_type: str  # SQL type (e.g., "INTEGER", "TEXT")
    is_numeric: bool  # Whether column is numeric
    
    def __repr__(self):
        return f"SchemaMatch({self.table}.{self.column}='{self.value}', score={self.score:.3f})"


class SchemaLiteralMatcher:
    """
    Matches user literals to database schema values.
    Adapter over LexicalLSHMatcher for schema linking.
    """
    
    def __init__(
        self,
        skip_likely_ids: bool = False,
        skip_constants: bool = True,
        **matcher_kwargs
    ):
        """
        Initialize schema matcher.
        
        Args:
            skip_likely_ids: Whether to skip columns that look like auto-increment IDs
                            (Default False - users do query IDs!)
            skip_constants: Whether to skip constant columns (1 distinct value in large tables)
                           (Default True - wastes resources, no value)
            **matcher_kwargs: Passed to LexicalLSHMatcher (including normalize_separators)
        """
        self.matcher = LexicalLSHMatcher(**matcher_kwargs)
        self.skip_likely_ids = skip_likely_ids
        self.skip_constants = skip_constants
        
        # Track indexed columns with type info (use Set to avoid duplicates)
        self.indexed_columns: Set[Tuple[str, str]] = set()
        self.column_types: Dict[Tuple[str, str], str] = {}
        self.column_is_numeric: Dict[Tuple[str, str], bool] = {}
    
    def _looks_like_auto_increment_id(self, profile: ColumnProfile) -> bool:
        """
        Check if column looks like an auto-incrementing surrogate key.
        
        Only returns True if ALL of these are true:
        - Numeric
        - High cardinality (>90% unique)
        - Column name matches ID pattern
        - Table is large enough (>1000 rows)
        
        This prevents false positives like zip_code, year, etc.
        """
        # Must be numeric
        if not profile.is_numeric:
            return False
        
        # Must have high cardinality
        if profile.total_records == 0:
            return False
        uniqueness = profile.distinct_count / profile.total_records
        if uniqueness < 0.9:
            return False
        
        # Must be a large table (skip heuristic only for large tables)
        if profile.total_records < 1000:
            return False
        
        # Column name must strongly suggest it's an ID
        col_lower = profile.column_name.lower()
        
        # Patterns that indicate an ID column
        id_patterns = [
            r'^id$',              # Exact "id"
            r'_id$',              # Ends with "_id" (user_id, order_id)
            r'^id_',              # Starts with "id_" (id_user, id_order)
            r'.*_?id$',           # Various ID patterns
        ]
        
        if any(re.match(pattern, col_lower) for pattern in id_patterns):
            return True
        
        return False
    
    def should_index_column(self, profile: ColumnProfile) -> bool:
        """
        Decide if a column should be indexed.
        
        The profiler already handles sampling (N=10,000), so we just decide
        whether to include the column at all.
        """
        # Skip if all NULL (no values to index)
        if profile.distinct_count == 0:
            return False
        
        # Skip constant columns in non-trivial tables
        if self.skip_constants:
            if profile.distinct_count == 1 and profile.total_records > 20:
                # Single value repeated many times - not useful
                return False
        
        # Optionally skip likely auto-increment IDs
        if self.skip_likely_ids:
            if self._looks_like_auto_increment_id(profile):
                return False
        
        return True
    
    def _is_numeric_type(self, data_type: str, profile: ColumnProfile) -> bool:
        """
        Determine if column is numeric based on declared type and observations.
        
        Args:
            data_type: SQL type string
            profile: Column profile with observed characteristics
            
        Returns:
            True if column should be treated as numeric for SQL
        """
        # Normalize and tokenize type
        # e.g., "INT(11)" → {"INT"}, "POINT" → {"POINT"}
        t = (data_type or "").upper()
        tokens = set(re.findall(r'[A-Z]+', t))
        
        # Check if any token is a numeric type
        # This prevents "POINT" from matching "INT" (substring bug)
        if tokens & NUMERIC_TYPES:
            return True
        
        # Fallback: if observed as numeric and not obviously text-ish
        text_types = {"TEXT", "CHAR", "CLOB", "VARCHAR", "STRING"}
        if profile.is_numeric and not (tokens & text_types):
            return True
        
        return False
    
    def index_column_from_profile(
        self,
        profile: ColumnProfile,
        use_frequency_order: bool = True,
    ) -> int:
        """
        Index a column from its profile.
        
        Args:
            profile: Column profile (profiler has already sampled values)
            use_frequency_order: Whether to prioritize frequent values
            
        Returns:
            Number of values indexed
        """
        if not self.should_index_column(profile):
            return 0
        
        # Get values to index
        # Trust the profiler - it already capped at sample_size (N=10,000)
        values = []
        
        # Priority 1: Use indexed_values from LSH profiling
        if hasattr(profile, 'indexed_values') and profile.indexed_values:
            values = list(profile.indexed_values)
        # Priority 2: Fallback to sample_values
        elif profile.sample_values:
            values = list(profile.sample_values)
        
        # Optional: Bring frequent values to the front for better recall
        if use_frequency_order and profile.top_k_values:
            top_values = [str(val) for val, _ in profile.top_k_values]
            
            # Prepend top values while avoiding duplicates (stable order)
            seen = set()
            values = [
                x for x in (top_values + values) 
                if not (x in seen or seen.add(x))
            ]
        
        if not values:
            return 0
        
        # Use "|" separator for consistent parsing
        prefix = f"{profile.table_name}|{profile.column_name}"
        
        # Index values (profiler already limited them, no need to re-sample)
        count = self.matcher.index_values(values, prefix=prefix)
        
        if count > 0:
            col_key = (profile.table_name, profile.column_name)
            self.indexed_columns.add(col_key)
            
            # Store type information for SQL generation
            self.column_types[col_key] = profile.data_type
            self.column_is_numeric[col_key] = self._is_numeric_type(
                profile.data_type, profile
            )
        
        return count
    
    def find_matching_fields(
        self,
        literal: str,
        top_k: Optional[int] = None,
    ) -> List[SchemaMatch]:
        """
        Find schema fields that contain the given literal.
        
        Args:
            literal: User literal (e.g., "Fresno County")
            top_k: Return top k results
            
        Returns:
            List of SchemaMatch, sorted by score
        """
        results = self.matcher.query(literal, top_k=top_k, exact_rerank=True)
        
        schema_matches = []
        for result in results:
            # Parse key: "table|column|norm_hash|orig_hash"
            parts = result.key.split('|')
            if len(parts) < 4:
                continue
            
            table = parts[0]
            column = parts[1]
            # parts[2] is normalized hash
            # parts[3] is original hash
            
            col_key = (table, column)
            
            # Get type information
            col_type = self.column_types.get(col_key, "TEXT")
            is_numeric = self.column_is_numeric.get(col_key, False)
            
            schema_matches.append(SchemaMatch(
                table=table,
                column=column,
                value=result.original_value,
                score=result.score,
                column_type=col_type,
                is_numeric=is_numeric,
            ))
        
        return schema_matches
    
    def __repr__(self):
        return f"SchemaLiteralMatcher({len(self.matcher)} values, {len(self.indexed_columns)} columns)"