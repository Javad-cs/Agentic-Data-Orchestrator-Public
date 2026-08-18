"""
Table-level profiling - aggregates column statistics.
"""

from typing import List, Optional
from dataclasses import dataclass

from .statistics import ColumnProfile


@dataclass
class TableProfile:
    """
    Aggregate profile for a database table.
    
    Combines column profiles to provide table-level view.
    """
    
    # Basic info
    table_name: str
    total_rows: int
    total_columns: int
    
    # Column profiles
    column_profiles: List[ColumnProfile]
    
    # Key column candidates (detected via heuristics, not guaranteed)
    primary_key_candidates: List[str]  # Likely PKs based on uniqueness + naming
    foreign_key_candidates: List[str]  # Likely FKs based on naming patterns
    first_column: str  # Fallback PK candidate 
    
    def get_column_names(self) -> List[str]:
        """Get list of all column names."""
        return [col.column_name for col in self.column_profiles]
    
    def get_column_profile(self, column_name: str) -> Optional[ColumnProfile]:
        """Get profile for specific column."""
        for col in self.column_profiles:
            if col.column_name == column_name:
                return col
        return None
    
    def has_numeric_columns(self) -> bool:
        """Check if table has any numeric columns."""
        return any(col.is_numeric for col in self.column_profiles)
    
    def has_date_columns(self) -> bool:
        """Check if table has any date-like columns."""
        return any(col.is_date_like for col in self.column_profiles)
    
    def __repr__(self):
        return (
            f"TableProfile({self.table_name}, "
            f"{self.total_rows} rows, {self.total_columns} columns)"
        )


class TableProfiler:
    """
    Profiles tables by aggregating column profiles.
    
    Does NOT re-profile columns - uses existing ColumnProfile objects.
    """
    
    def profile_table(
        self,
        db,
        table_name: str,
        column_profiles: List[ColumnProfile]
    ) -> TableProfile:
        """
        Create table profile from column profiles.
        
        Args:
            db: Database connection
            table_name: Table name
            column_profiles: Pre-computed column profiles for this table
            
        Returns:
            TableProfile with aggregated information
        """
        if not column_profiles:
            raise ValueError(f"No column profiles provided for table {table_name}")
        
        # Get total rows from first column (all columns have same count)
        total_rows = column_profiles[0].total_records
        total_columns = len(column_profiles)
        
        # Detect keys
        primary_key_candidates = self._detect_primary_keys(column_profiles)
        foreign_key_candidates = self._detect_foreign_keys(column_profiles)
        
        # First column (often PK even if not explicitly marked)
        first_column = column_profiles[0].column_name
        
        return TableProfile(
            table_name=table_name,
            total_rows=total_rows,
            total_columns=total_columns,
            column_profiles=column_profiles,
            primary_key_candidates=primary_key_candidates,
            foreign_key_candidates=foreign_key_candidates,
            first_column=first_column
        )
    
    def _detect_primary_keys(self, column_profiles: List[ColumnProfile]) -> List[str]:
        """
        Detect likely primary key columns using heuristics.
        
        These are candidates, not guaranteed PKs.
        We cannot query SQLite's schema metadata reliably in BIRD dataset.
        
        Heuristics:
        - Column name contains 'id' or 'key' or 'pk'
        - All values are unique (distinct_count == non_null_count)
        - No NULL values
        
        Returns:
            List of column names that are likely primary keys
        """
        primary_key_candidates = []
        
        for col in column_profiles:
            # Check uniqueness (strong signal for PK)
            is_unique = (
                col.distinct_count == col.non_null_count and
                col.null_count == 0
            )
            
            # Check name hints
            col_lower = col.column_name.lower()
            has_key_name = (
                'id' in col_lower or
                'key' in col_lower or
                col_lower == 'pk' or
                col_lower.endswith('_id')
            )
            
            # If unique AND has key-like name, likely PK
            if is_unique and has_key_name:
                primary_key_candidates.append(col.column_name)
            
            # If named exactly "id" and is unique, definitely PK
            elif col.column_name.lower() == 'id' and is_unique:
                primary_key_candidates.append(col.column_name)
        
        return primary_key_candidates

    def _detect_foreign_keys(self, column_profiles: List[ColumnProfile]) -> List[str]:
        """
        Detect likely foreign key columns using heuristics.
        
        These are candidates, not guaranteed FKs.
        
        Heuristics:
        - Column name ends with '_id' but is not the first column
        - Column name contains 'fk'
        - NOT a primary key candidate
        
        Returns:
            List of column names that are likely foreign keys
        """
        foreign_key_candidates = []
        
        # Get PKs to exclude them
        primary_keys_set = set(self._detect_primary_keys(column_profiles))
        
        for col in column_profiles:
            col_lower = col.column_name.lower()
            
            # Skip if it's a PK
            if col.column_name in primary_keys_set:
                continue
            
            # Check FK name patterns
            is_fk_pattern = (
                col_lower.endswith('_id') or
                'fk' in col_lower or
                col_lower.startswith('ref_')
            )
            
            if is_fk_pattern:
                foreign_key_candidates.append(col.column_name)
        
        return foreign_key_candidates