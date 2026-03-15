"""
Data types for schema linking.
"""

from dataclasses import dataclass
from typing import Optional, Literal, List
from enum import Enum


@dataclass
class FocusedField:
    """
    A field selected by the focused schema builder.
    
    Keeps separate scores from FAISS (semantic) and LSH (literal) 
    for transparency and paper alignment.
    """
    table: str
    column: str
    
    # Separate scores (paper-aligned: two indexes, two signals)
    faiss_score: Optional[float] = None      # Semantic similarity (0-1)
    lsh_score: Optional[float] = None        # Jaccard similarity (0-1)
    
    # Source tracking
    selected_by: Literal["faiss", "lsh", "both"] = "faiss"
    
    # For debugging
    description: str = ""
    
    def __repr__(self):
        scores = []
        if self.faiss_score is not None:
            scores.append(f"faiss={self.faiss_score:.3f}")
        if self.lsh_score is not None:
            scores.append(f"lsh={self.lsh_score:.3f}")
        
        score_str = ", ".join(scores) if scores else "no scores"
        return f"FocusedField({self.table}.{self.column}, {score_str}, via={self.selected_by})"


class SchemaVariant(str, Enum):
    """
    The 5 schema representation variants from the BIRD paper.
    
    Focused schema = semantically relevant fields (FAISS + LSH)
    Full schema = all fields in database
    
    Minimal = short LLM description only
    Maximal = long LLM description only
    Full = SME description + long LLM description
    """
    FOCUSED_MINIMAL = "focused_minimal"
    FOCUSED_MAXIMAL = "focused_maximal"
    FOCUSED_FULL = "focused_full"
    FULL_MINIMAL = "full_minimal"
    FULL_MAXIMAL = "full_maximal"
    
    def is_focused(self) -> bool:
        """Whether this variant uses focused schema."""
        return self.value.startswith("focused_")
    
    def is_full_schema(self) -> bool:
        """Whether this variant uses full schema."""
        return self.value.startswith("full_")
    
    def profile_type(self) -> str:
        """Get profile type: minimal, maximal, or full."""
        if "minimal" in self.value:
            return "minimal"
        elif "maximal" in self.value:
            return "maximal"
        elif "full" in self.value:
            return "full"
        return "minimal"


@dataclass
class SchemaRepresentation:
    """
    A schema representation ready for LLM consumption.
    """
    variant: SchemaVariant
    fields: List[FocusedField]
    text: str  # Formatted text for LLM
    
    def __repr__(self):
        return f"SchemaRepresentation({self.variant.value}, {len(self.fields)} fields)"
    
    
# Newly added for coordinator to be able to break query into subqueries
@dataclass
class ColumnSummary:
    """
    Column information for Slow Lane planner.
    Minimal info needed for tool routing decisions.
    """
    name: str
    data_type: str
    description: str  # From FieldMetadata.short_description
    
    # Highlighting flags (why this column is shown)
    is_lsh_matched: bool = False  # Matched query literally
    is_pk_candidate: bool = False  # Changed from is_primary_key
    is_fk_candidate: bool = False  # Changed from is_foreign_key
    is_first_column: bool = False
    
    def __repr__(self):
        flags = []
        if self.is_lsh_matched:
            flags.append("LSH")
        if self.is_pk_candidate:  # Updated
            flags.append("PK")
        if self.is_fk_candidate:  # Updated
            flags.append("FK")
        if self.is_first_column:
            flags.append("1st")
        
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"{self.name} ({self.data_type}){flag_str}"


@dataclass
class TableSummary:
    """
    Minimal table summary for Slow Lane planner.
    
    Includes only information needed for tool routing decisions.
    NOT for SQL generation (that's SchemaRepresentation's job).
    """
    table_name: str
    description: str  # From TableMetadata (1 sentence)
    
    # Matching scores (why this table was selected)
    faiss_score: Optional[float] = None  # Table description similarity
    has_lsh_match: bool = False  # Any field name matched query
    
    # Columns to show (smart selection with safety valve)
    columns: List[ColumnSummary] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
    
    def get_lsh_matched_columns(self) -> List[ColumnSummary]:
        """Get columns that matched the query literally."""
        return [col for col in self.columns if col.is_lsh_matched]
    
    def get_key_columns(self) -> List[ColumnSummary]:
        """Get primary/foreign key candidate columns."""
        return [col for col in self.columns if col.is_pk_candidate or col.is_fk_candidate]
    
    def format_for_planner(self) -> str:
        """
        Format table summary as text for planner prompt.
        
        Example output:
```
        employees (FAISS: 0.85)
          Description: Stores employee records including compensation and assignments
          
          Columns (8 total):
          [*] salary (numeric) - annual compensation [LSH matched]
          [K] id (int) - employee identifier [PK]
          [F] department_id (int) - department reference [FK]
              name (text) - employee full name
              ... (4 more columns)
```
        """
        lines = []
        
        # Header with matching info
        header = f"{self.table_name}"
        if self.faiss_score:
            header += f" (FAISS: {self.faiss_score:.2f})"
        elif self.has_lsh_match:
            header += " (LSH matched)"
        lines.append(header)
        
        # Description
        lines.append(f"  Description: {self.description}")
        lines.append("")
        
        # Columns
        lines.append(f"  Columns ({len(self.columns)} total):")
        
        for col in self.columns:
            # Marker based on flags
            marker = "   "
            if col.is_lsh_matched:
                marker = "[*]"
            elif col.is_pk_candidate: 
                marker = "[K]"
            elif col.is_fk_candidate:
                marker = "[F]"
            
            col_line = f"  {marker} {col.name} ({col.data_type})"
            
            if col.description:
                col_line += f" - {col.description}"
            
            # Add flags
            flags = []
            if col.is_lsh_matched:
                flags.append("LSH matched")
            if col.is_pk_candidate: 
                flags.append("PK candidate")
            if col.is_fk_candidate:
                flags.append("FK candidate")
            
            if flags:
                col_line += f" [{', '.join(flags)}]"
            
            lines.append(col_line)
        
        return "\n".join(lines)

    def __repr__(self):
        match_info = f"FAISS={self.faiss_score:.2f}" if self.faiss_score else "LSH" if self.has_lsh_match else "?"
        return f"TableSummary({self.table_name}, {len(self.columns)} cols, {match_info})"