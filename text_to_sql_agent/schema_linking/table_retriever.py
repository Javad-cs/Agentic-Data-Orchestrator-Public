"""
Table retrieval for Slow Lane planner.

Uses dual filtering (FAISS + LSH) to find relevant tables,
then performs smart column selection with safety valve.
"""

from typing import List, Dict, Set, Optional
from .types import TableSummary, ColumnSummary
from .focused_schema import FocusedSchemaConfig
from indexing import FieldIndex, SchemaLiteralMatcher
from profiling import TableMetadata


class TableRetriever:
    """
    Retrieves relevant tables for Slow Lane planner.
    
    Dual filtering approach:
    1. FAISS on table descriptions (semantic similarity)
    2. LSH on field names (literal matching)
    
    Include table if EITHER passes threshold (recall-focused).
    """
    
    def __init__(
        self,
        field_index: FieldIndex,
        literal_matcher: SchemaLiteralMatcher,
        table_metadata_map: Dict[str, TableMetadata],
        config: Optional[FocusedSchemaConfig] = None
    ):
        """
        Initialize table retriever.
        
        Args:
            field_index: Existing FAISS index over field descriptions
            literal_matcher: Existing LSH matcher over field values
            table_metadata_map: Map of table_name -> TableMetadata
            config: Configuration (uses defaults if None)
        """
        self.field_index = field_index
        self.literal_matcher = literal_matcher
        self.table_metadata_map = table_metadata_map
        self.config = config or FocusedSchemaConfig()
    
    def get_relevant_tables(
        self,
        question: str,
        max_tables: int = 5
    ) -> List[TableSummary]:
        """
        Get relevant tables for planner.
        
        Process:
        1. FAISS search on table descriptions
        2. LSH search on field names (grouped by table)
        3. Merge results (union - include if EITHER passes)
        4. For each table: smart column selection
        5. Return top-K tables sorted by relevance
        
        Args:
            question: User query
            max_tables: Maximum tables to return
            
        Returns:
            List of TableSummary objects for planner
        """
        # Step 1: FAISS on table descriptions
        faiss_tables = self._search_tables_by_description(question)
        
        # Step 2: LSH on field names (grouped by table)
        lsh_tables = self._search_tables_by_field_names(question)
        
        # Step 3: Merge (union)
        table_summaries = self._merge_and_build_summaries(
            question,
            faiss_tables,
            lsh_tables
        )
        
        # Step 4: Sort by relevance and limit
        table_summaries = self._sort_by_relevance(table_summaries)
        
        return table_summaries[:max_tables]
    
    def _search_tables_by_description(self, question: str) -> Dict[str, float]:
        """
        Search tables by semantic similarity of table descriptions.
        
        Uses FAISS to embed and search table descriptions.
        
        Returns:
            Dict mapping table_name -> FAISS score
        """
        # We need to search table descriptions, not field descriptions
        # Since we only have FieldIndex, we'll build a temporary mapping
        
        # For now, we'll use field-level FAISS and aggregate to table level
        # This is a reasonable approximation: if table's fields are relevant,
        # the table is relevant
        
        from indexing import SemanticMatch
        
        # Search fields with FAISS
        field_matches = self.field_index.search(
            question,
            top_k=100  # Get many candidates
        )
        
        # Group by table and take max score per table
        table_scores: Dict[str, float] = {}
        
        for match in field_matches:
            if match.score >= self.config.faiss_threshold:
                table = match.table
                # Keep maximum score for this table
                if table not in table_scores or match.score > table_scores[table]:
                    table_scores[table] = match.score
        
        return table_scores
    
    def _search_tables_by_field_names(self, question: str) -> Dict[str, Set[str]]:
        """
        Search tables by literal matching of field names.
        
        Uses LSH to find fields whose names appear in query.
        
        Returns:
            Dict mapping table_name -> set of matched column names
        """
        from .literal_extractor import extract_literals
        
        # Extract literals from question
        literals = extract_literals(question)
        
        if not literals:
            return {}
        
        # Find matching fields for each literal
        table_matched_columns: Dict[str, Set[str]] = {}
        
        for literal in literals:
            matches = self.literal_matcher.find_matching_fields(
                literal,
                top_k=50
            )
            
            for match in matches:
                if match.score >= self.config.lsh_threshold:
                    table = match.table
                    column = match.column
                    
                    if table not in table_matched_columns:
                        table_matched_columns[table] = set()
                    
                    table_matched_columns[table].add(column)
        
        return table_matched_columns
    
    def _merge_and_build_summaries(
        self,
        question: str,
        faiss_tables: Dict[str, float],
        lsh_tables: Dict[str, Set[str]]
    ) -> List[TableSummary]:
        """
        Merge FAISS and LSH results and build TableSummary objects.
        
        Union strategy: Include table if it appears in EITHER source.
        
        Args:
            question: Original query (for column selection)
            faiss_tables: table_name -> FAISS score
            lsh_tables: table_name -> set of matched columns
            
        Returns:
            List of TableSummary objects
        """
        # Get all unique tables
        all_tables = set(faiss_tables.keys()) | set(lsh_tables.keys())
        
        summaries = []
        
        for table_name in all_tables:
            # Skip if no metadata
            if table_name not in self.table_metadata_map:
                continue
            
            table_meta = self.table_metadata_map[table_name]
            
            # Get scores
            faiss_score = faiss_tables.get(table_name)
            lsh_matched_columns = lsh_tables.get(table_name, set())
            
            # Select columns to show
            columns = self._select_columns_for_table(
                table_meta,
                lsh_matched_columns
            )
            
            # Build TableSummary
            summary = TableSummary(
                table_name=table_name,
                description=table_meta.description,
                faiss_score=faiss_score,
                has_lsh_match=len(lsh_matched_columns) > 0,
                columns=columns
            )
            
            summaries.append(summary)
        
        return summaries
    
    def _select_columns_for_table(
        self,
        table_meta: TableMetadata,
        lsh_matched_columns: Set[str],
        max_columns: int = 50
    ) -> List[ColumnSummary]:
        """
        Smart column selection with safety valve.
        
        Logic:
        1. If total_columns <= 50: show ALL columns
        2. If total_columns > 50: show top 50 by priority:
           - LSH matched columns (priority 1)
           - Primary key candidates (priority 2)
           - Foreign key candidates (priority 3)
           - First column (priority 4)
           - Remaining by distinctness (priority 5)
        
        Args:
            table_meta: Table metadata
            lsh_matched_columns: Columns that matched query literally
            max_columns: Maximum columns to show (safety valve)
            
        Returns:
            List of ColumnSummary objects
        """
        profile = table_meta.profile
        total_columns = profile.total_columns
        
        # SAFETY VALVE: If <= 50 columns, show ALL
        if total_columns <= max_columns:
            return self._build_all_columns(table_meta, lsh_matched_columns)
        
        # SMART SELECTION: For large tables (> 50 columns)
        return self._build_smart_columns(
            table_meta,
            lsh_matched_columns,
            max_columns
        )
    
    def _build_all_columns(
        self,
        table_meta: TableMetadata,
        lsh_matched_columns: Set[str]
    ) -> List[ColumnSummary]:
        """Build ColumnSummary for ALL columns."""
        profile = table_meta.profile
        columns = []
        
        for i, col_profile in enumerate(profile.column_profiles):
            col_name = col_profile.column_name
            
            col_summary = ColumnSummary(
                name=col_name,
                data_type=col_profile.data_type,
                description=table_meta.column_summaries.get(col_name, ""),
                is_lsh_matched=col_name in lsh_matched_columns,
                is_pk_candidate=col_name in profile.primary_key_candidates,  # Updated
                is_fk_candidate=col_name in profile.foreign_key_candidates,  # Updated
                is_first_column=(i == 0)
            )
            
            columns.append(col_summary)
        
        return columns
    
    def _build_smart_columns(
        self,
        table_meta: TableMetadata,
        lsh_matched_columns: Set[str],
        max_columns: int
    ) -> List[ColumnSummary]:
        """
        Smart column selection for large tables (> 50 columns).
        
        Priority order:
        1. LSH matched columns
        2. Primary key candidates
        3. Foreign key candidates
        4. First column
        5. Top N by distinctness
        """
        profile = table_meta.profile
        
        # Categorize columns
        priority_1 = []  # LSH matched
        priority_2 = []  # PK
        priority_3 = []  # FK
        priority_4 = []  # First column
        priority_5 = []  # Others (sorted by distinctness)
        
        for i, col_profile in enumerate(profile.column_profiles):
            col_name = col_profile.column_name
            
            col_summary = ColumnSummary(
                name=col_name,
                data_type=col_profile.data_type,
                description=table_meta.column_summaries.get(col_name, ""),
                is_lsh_matched=col_name in lsh_matched_columns,
                is_pk_candidate=col_name in profile.primary_key_candidates,  # Updated
                is_fk_candidate=col_name in profile.foreign_key_candidates,  # Updated
                is_first_column=(i == 0)
            )
            
            # Prioritize
            if col_name in lsh_matched_columns:
                priority_1.append(col_summary)
            elif col_name in profile.primary_key_candidates:
                priority_2.append(col_summary)
            elif col_name in profile.foreign_key_candidates:
                priority_3.append(col_summary)
            elif i == 0:
                priority_4.append(col_summary)
            else:
                # Store with distinctness for sorting
                distinctness = col_profile.distinct_count / max(col_profile.non_null_count, 1)
                priority_5.append((distinctness, col_summary))
        
        # Sort priority_5 by distinctness (descending)
        priority_5.sort(key=lambda x: x[0], reverse=True)
        priority_5_cols = [col for _, col in priority_5]
        
        # Combine in priority order
        selected = []
        selected.extend(priority_1)
        selected.extend(priority_2)
        selected.extend(priority_3)
        selected.extend(priority_4)
        selected.extend(priority_5_cols)
        
        # Limit to max_columns
        return selected[:max_columns]
    
    def _sort_by_relevance(self, summaries: List[TableSummary]) -> List[TableSummary]:
        """
        Sort table summaries by relevance.
        
        Priority:
        1. FAISS score (with small boost if has LSH match)
        2. Has both signals (tiebreaker)
        3. Number of LSH matched columns (tiebreaker)
        """
        def sort_key(summary: TableSummary) -> tuple:
            # Base score from FAISS
            faiss_score = summary.faiss_score or 0.0
            
            # Small boost if has LSH match (encourages dual signals)
            if summary.has_lsh_match:
                faiss_score += 0.1
            
            # Tiebreakers
            num_lsh_matches = len([c for c in summary.columns if c.is_lsh_matched])
            has_both = (summary.faiss_score is not None) and summary.has_lsh_match
            
            return (faiss_score, has_both, num_lsh_matches)
        
        summaries.sort(key=sort_key, reverse=True)
        return summaries