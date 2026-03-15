"""
Focused schema builder - combines FAISS and LSH for schema linking.
Paper-aligned: Uses semantic similarity (FAISS) + literal matching (LSH).
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .types import FocusedField, TableSummary
from profiling import TableMetadata
from .literal_extractor import LiteralExtractor, extract_literals
from indexing import FieldIndex, SemanticMatch, SchemaLiteralMatcher, SchemaMatch


@dataclass
class FocusedSchemaConfig:
    """
    Configuration for focused schema building.
    
    Paper-aligned: Prioritizes recall over precision.
    Fields are selected by threshold, not top-k limits.
    """
    
    # FAISS semantic search
    faiss_threshold: float = 0.2  # Minimum cosine similarity (lower = more recall)
    faiss_max_candidates: int = 100  # Max candidates to retrieve (optimization only)
    
    # LSH literal matching  
    lsh_threshold: float = 0.3  # Minimum Jaccard similarity
    lsh_max_candidates: int = 50  # Max candidates per literal (optimization only)
    
    # Merging strategy
    merge_strategy: str = "union"  # "union", "intersection", "faiss_only", "lsh_only"
    
    # Note: No max_fields limit! 
    # Paper says "recall > precision" - include all fields above thresholds


class FocusedSchemaBuilder:
    """
    Build focused schema by combining FAISS semantic search and LSH literal matching.
    
    Paper-aligned: The focused schema selection uses both:
    1. FAISS over long field descriptions (semantic)
    2. LSH over field values (literal matching)
    """
    
    def __init__(
        self,
        field_index: FieldIndex,
        literal_matcher: SchemaLiteralMatcher,
        literal_extractor: Optional[LiteralExtractor] = None,
        config: Optional[FocusedSchemaConfig] = None
    ):
        """
        Initialize focused schema builder.
        
        Args:
            field_index: FAISS index over field descriptions
            literal_matcher: LSH matcher over field values
            literal_extractor: Optional literal extractor (creates default if None)
            config: Configuration (uses defaults if None)
        """
        self.field_index = field_index
        self.literal_matcher = literal_matcher
        self.literal_extractor = literal_extractor or LiteralExtractor(use_llm_fallback=False)
        self.config = config or FocusedSchemaConfig()
    
    def build(
        self, 
        question: str,
        return_scores: bool = True
    ) -> List[FocusedField]:
        """
        Build focused schema for a question.
        
        Args:
            question: Natural language question
            return_scores: Whether to include scores in results
            
        Returns:
            List of FocusedField with scores and source tracking
        """
        # Step 1: FAISS semantic search
        faiss_matches = self._search_faiss(question)
        
        # Step 2: LSH literal matching
        lsh_matches = self._search_lsh(question)
        
        # Step 3: Merge results
        focused_fields = self._merge_matches(faiss_matches, lsh_matches)
        
        # Return ALL fields above thresholds (paper: recall > precision)
        return focused_fields
    
    def _search_faiss(self, question: str) -> Dict[Tuple[str, str], float]:
        """
        Search FAISS index for semantically relevant fields.
        
        Returns ALL fields above threshold (not limited by top-k).
        
        Returns:
            Dict mapping (table, column) to cosine similarity score
        """
        # Retrieve candidates (optimization - get more than we need)
        matches = self.field_index.search(
            question, 
            top_k=self.config.faiss_max_candidates
        )
        
        # Filter by threshold (this is what matters, not top-k)
        field_scores = {}
        for match in matches:
            if match.score >= self.config.faiss_threshold:
                key = (match.table, match.column)
                field_scores[key] = match.score
        
        return field_scores
    
    def _search_lsh(self, question: str) -> Dict[Tuple[str, str], float]:
        """
        Search LSH index for literal matches.
        
        Returns ALL fields above threshold for any extracted literal.
        
        Returns:
            Dict mapping (table, column) to best Jaccard similarity score
        """
        # Extract literals from question
        literals = self.literal_extractor.extract_simple(question)
        
        if not literals:
            return {}
        
        # Query LSH for each literal
        field_scores: Dict[Tuple[str, str], float] = {}
        
        for literal in literals:
            # Get candidates for this literal
            matches = self.literal_matcher.find_matching_fields(
                literal,
                top_k=self.config.lsh_max_candidates
            )
            
            # Filter by threshold (key part!)
            for match in matches:
                if match.score >= self.config.lsh_threshold:
                    key = (match.table, match.column)
                    # Keep maximum score if field matches multiple literals
                    if key not in field_scores or match.score > field_scores[key]:
                        field_scores[key] = match.score
        
        return field_scores
    
    def _merge_matches(
        self,
        faiss_matches: Dict[Tuple[str, str], float],
        lsh_matches: Dict[Tuple[str, str], float]
    ) -> List[FocusedField]:
        """
        Merge FAISS and LSH results according to merge strategy.
        
        Returns:
            List of FocusedField sorted by relevance
        """
        focused_fields = []
        
        if self.config.merge_strategy == "union":
            # Union: Include field if it appears in either FAISS or LSH
            all_keys = set(faiss_matches.keys()) | set(lsh_matches.keys())
            
            for table, column in all_keys:
                faiss_score = faiss_matches.get((table, column))
                lsh_score = lsh_matches.get((table, column))
                
                # Determine source
                if faiss_score and lsh_score:
                    selected_by = "both"
                elif faiss_score:
                    selected_by = "faiss"
                else:
                    selected_by = "lsh"
                
                focused_fields.append(FocusedField(
                    table=table,
                    column=column,
                    faiss_score=faiss_score,
                    lsh_score=lsh_score,
                    selected_by=selected_by
                ))
        
        elif self.config.merge_strategy == "intersection":
            # Intersection: Include only if it appears in BOTH
            common_keys = set(faiss_matches.keys()) & set(lsh_matches.keys())
            
            for table, column in common_keys:
                focused_fields.append(FocusedField(
                    table=table,
                    column=column,
                    faiss_score=faiss_matches[(table, column)],
                    lsh_score=lsh_matches[(table, column)],
                    selected_by="both"
                ))
        
        elif self.config.merge_strategy == "faiss_only":
            # Only use FAISS results
            for (table, column), score in faiss_matches.items():
                focused_fields.append(FocusedField(
                    table=table,
                    column=column,
                    faiss_score=score,
                    lsh_score=None,
                    selected_by="faiss"
                ))
        
        elif self.config.merge_strategy == "lsh_only":
            # Only use LSH results
            for (table, column), score in lsh_matches.items():
                focused_fields.append(FocusedField(
                    table=table,
                    column=column,
                    faiss_score=None,
                    lsh_score=score,
                    selected_by="lsh"
                ))
        
        # Sort by combined relevance
        # Priority: both > faiss > lsh
        # Within each, sort by max(faiss_score, lsh_score)
        def sort_key(field: FocusedField) -> tuple:
            # Primary: selection source (both=3, faiss=2, lsh=1)
            source_priority = {"both": 3, "faiss": 2, "lsh": 1}
            
            # Secondary: maximum score
            max_score = max(
                field.faiss_score or 0.0,
                field.lsh_score or 0.0
            )
            
            return (source_priority[field.selected_by], max_score)
        
        focused_fields.sort(key=sort_key, reverse=True)
        
        return focused_fields
    
    def get_statistics(self, focused_fields: List[FocusedField]) -> Dict[str, int]:
        """Get statistics about focused schema."""
        return {
            "total_fields": len(focused_fields),
            "faiss_only": sum(1 for f in focused_fields if f.selected_by == "faiss"),
            "lsh_only": sum(1 for f in focused_fields if f.selected_by == "lsh"),
            "both": sum(1 for f in focused_fields if f.selected_by == "both"),
            "unique_tables": len(set(f.table for f in focused_fields))
        }
    
    def __repr__(self):
        return (
            f"FocusedSchemaBuilder("
            f"faiss={len(self.field_index)} fields, "
            f"lsh={len(self.literal_matcher.matcher)} values, "
            f"strategy={self.config.merge_strategy})"
        )
        
    def get_relevant_tables(
            self,
            question: str,
            table_metadata_map: Dict[str, 'TableMetadata'],
            max_tables: int = 5
        ) -> List['TableSummary']:
            """
            Get table summaries for Slow Lane planner.
            
            Convenience method that wraps TableRetriever for easy access.
            
            Args:
                question: User query
                table_metadata_map: Pre-computed table metadata
                max_tables: Maximum tables to return
                
            Returns:
                List of TableSummary objects for planner
                
            Example:
    ```python
                # Get relevant tables for planner
                table_summaries = builder.get_relevant_tables(
                    question="What is average employee salary?",
                    table_metadata_map=metadata_map,
                    max_tables=5
                )
                
                # Format for planner prompt
                for summary in table_summaries:
                    print(summary.format_for_planner())
    ```
            """
            from .table_retriever import TableRetriever
            
            retriever = TableRetriever(
                field_index=self.field_index,
                literal_matcher=self.literal_matcher,
                table_metadata_map=table_metadata_map,
                config=self.config
            )
            
            return retriever.get_relevant_tables(question, max_tables)