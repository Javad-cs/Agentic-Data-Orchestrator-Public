"""
Types for SQL generation and refinement loop.
Implements Algorithm 1 from BIRD paper.
"""

from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional
from enum import Enum

from schema_linking import SchemaVariant


@dataclass
class SQLParseResult:
    """Result of parsing SQL query."""
    sql: str
    referenced_fields: Set[Tuple[str, str]]  # {(table, column), ...}
    literals: Set[str]  # Literals from WHERE clauses (set for deduplication)
    is_valid: bool
    parse_error: Optional[str] = None


@dataclass
class LiteralMatch:
    """Match between SQL literal and database field."""
    literal: str
    table: str
    column: str
    score: float  # LSH similarity score
    

@dataclass
class IterationResult:
    """
    Result of one refinement iteration.
    Tracks SQL generation, literal matching, and schema augmentation.
    """
    variant: SchemaVariant
    iteration: int
    
    # Generated SQL
    sql: str
    is_valid_sql: bool
    
    # Extracted from SQL
    fields_used: Set[Tuple[str, str]]  # FieldsQ in paper
    literals_used: Set[str]  # LitsQ in paper (set for deduplication)
    
    # Literal matching
    literal_matches: List[LiteralMatch]  # Fields containing literals
    missing_literals: List[str]  # MissingLits in paper (ordered for prompting)
    
    # Schema augmentation
    augmented_fields: Set[Tuple[str, str]]  # LitFieldsQ in paper
    
    # Debug info
    schema_used: str  # Full schema text sent to LLM
    llm_prompt: str
    llm_response: str
    

@dataclass
class VariantResult:
    """
    Result for one schema variant (all iterations).
    """
    variant: SchemaVariant
    iterations: List[IterationResult]
    
    # Final fields discovered
    final_fields: Set[Tuple[str, str]]
    final_literals: Set[str]
    
    # Success metrics
    num_refinements: int
    converged: bool  # True if no new fields found


@dataclass
class Algorithm1Result:
    """
    Final output of Algorithm 1.
    Union of fields across all 5 schema variants.
    """
    question: str
    
    # Results per variant
    variant_results: List[VariantResult]
    
    # Final union
    final_fields: Set[Tuple[str, str]]  # Union of all FieldsQ
    final_literals: Set[str]  # Union of all LitsQ
    
    # Statistics
    total_iterations: int
    total_sql_generated: int
    
    def get_fields_by_variant(self, variant: SchemaVariant) -> Set[Tuple[str, str]]:
        """Get fields discovered by specific variant."""
        for vr in self.variant_results:
            if vr.variant == variant:
                return vr.final_fields
        return set()
    
    def get_iteration_by_variant(self, variant: SchemaVariant, iteration: int) -> Optional[IterationResult]:
        """Get specific iteration result."""
        for vr in self.variant_results:
            if vr.variant == variant and iteration < len(vr.iterations):
                return vr.iterations[iteration]
        return None