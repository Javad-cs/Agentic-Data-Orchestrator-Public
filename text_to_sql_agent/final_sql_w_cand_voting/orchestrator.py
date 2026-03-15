"""
Voting Orchestrator (Phase 4).
Coordinated the Ensemble: Generation -> Validation -> Execution -> Voting.
Includes deterministic schema context and robust result normalization.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from core import BaseDatabase
from sql_generation.refinement_loop import Algorithm1Runner
from .few_shot_store import FewShotStore
from .candidate_generator import CandidateGenerator
from .sql_linter import SQLLinter
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SQLCandidate:
    sql: str
    description: str
    is_valid: bool = False
    execution_result: Any = None
    error_message: str = ""

class VotingOrchestrator:
    """
    Manages the 'Diversity & Voting' phase.
    """

    def __init__(
        self, 
        runner: Algorithm1Runner,
        few_shot_store: FewShotStore,
        candidate_generator: CandidateGenerator,
        database: BaseDatabase,
        num_candidates: int = 3
    ):
        self.runner = runner
        self.store = few_shot_store
        self.generator = candidate_generator
        self.db = database
        self.linter = SQLLinter(dialect=settings.db_dialect)
        self.num_candidates = num_candidates

    def solve(self, question: str, schema_variants: Dict, db_id: Optional[str] = None) -> str:
        """
        Main entry point.
        Args:
            question: User query
            schema_variants: Output from Schema Variant Generator
            db_id: (Optional) Database ID for filtering few-shot examples
        """
        logger.info(f"--- Starting Voting Flow for: {question} ---")

        # 1. Discovery Phase (Algorithm 1)
        logger.info("Step 1: Running Algorithm 1 for Field Discovery...")
        algo_result = self.runner.run(question, schema_variants)
        
        final_fields = algo_result.final_fields
        metadata_map = self.runner.metadata_map
        
        logger.info(f"Discovery complete. Found {len(final_fields)} relevant fields.")

        # 2. Retrieval Phase
        logger.info("Step 2: Retrieving Few-Shot Examples...")
        
        # Build Deterministic Schema Context for Masking
        # Format: # Table (Column, Column)
        schema_context = self._build_schema_context_str(final_fields)
        
        # Pass context to store so it can mask the question correctly
        # Filter by db_id if provided (critical for BIRD multi-db setup)
        scored_examples = self.store.retrieve(
            question, 
            schema_context=schema_context, 
            k=5, 
            filter_db_id=db_id
        )
        few_shots = [ex for ex, score in scored_examples]
        logger.info(f"Retrieved {len(few_shots)} few-shot examples.")

        # 3. Generation Phase (Diversity)
        logger.info(f"Step 3: Generating {self.num_candidates} diverse candidates...")
        generated_sqls = self.generator.generate_diverse_candidates(
            question=question,
            final_fields=final_fields,
            metadata_map=metadata_map,
            few_shots=few_shots,
            num_candidates=self.num_candidates
        )
        
        candidates = [
            SQLCandidate(sql=g.sql, description=g.description) 
            for g in generated_sqls
        ]

        if not candidates:
            logger.error("No candidates generated. Fallback to Algorithm 1 result.")
            last_best = self._extract_fallback_sql(algo_result)
            return last_best if last_best else "SELECT 'No SQL Generated'"

        # 4. Linting Phase
        logger.info("Step 4: Linting Candidates...")
        valid_candidates = []
        for cand in candidates:
            lint_res = self.linter.lint(cand.sql)
            
            if lint_res.is_valid:
                cand.is_valid = True
                if lint_res.needs_correction:
                    logger.warning(f"Candidate '{cand.description}' flagged: {lint_res.error_message}")
                valid_candidates.append(cand)
            else:
                logger.warning(f"Candidate rejected: {lint_res.error_message}")
        
        if not valid_candidates:
            logger.error("All candidates rejected by linter.")
            return "SELECT 'No valid SQL generated'"

        # 5. Execution & Voting Phase
        logger.info("Step 5: Execution and Voting...")
        final_sql = self._majority_vote(valid_candidates)
        return final_sql

    def _build_schema_context_str(self, final_fields: Set[Tuple[str, str]]) -> str:
        """
        Convert the discovered fields set into a deterministic string format.
        Format: # Table (Column, Column)
        """
        table_cols = {}
        for table, col in final_fields:
            if table not in table_cols:
                table_cols[table] = []
            table_cols[table].append(col)
            
        lines = []
        # FIX: Sort keys (tables) for determinism
        for table in sorted(table_cols.keys()):
            # FIX: Sort values (columns) for determinism
            cols = sorted(table_cols[table])
            cols_str = ", ".join(cols)
            lines.append(f"# {table} ({cols_str})")
            
        return "\n".join(lines)

    def _extract_fallback_sql(self, algo_result) -> Optional[str]:
        if algo_result.variant_results:
             try:
                 return algo_result.variant_results[-1].iterations[-1].sql
             except:
                 pass
        return None

    def _normalize_row(self, row: Any) -> Tuple:
        """Normalize a single row into a comparable tuple."""
        # Handle dictionary-like objects (e.g. sqlite3.Row or dicts from certain drivers)
        if hasattr(row, "keys"): 
            # Sort by keys to ensure dict {a:1, b:2} == {b:2, a:1}
            return tuple(str(row[k]) for k in sorted(row.keys()))
        elif isinstance(row, dict):
            return tuple(str(row[k]) for k in sorted(row.keys()))
        elif isinstance(row, (list, tuple)):
            return tuple(str(v) for v in row)
        else:
            # Scalar value
            return (str(row),)

    def _majority_vote(self, candidates: List[SQLCandidate]) -> str:
        results_map = {} 
        
        for cand in candidates:
            try:
                result_rows = self.db.execute_query(cand.sql) 
                
                # Robust Normalization
                if isinstance(result_rows, list):
                     # Normalize each row, then sort the rows to ignore result set order
                     # (e.g. [A, B] == [B, A] for set equality context)
                     normalized_set = sorted([self._normalize_row(r) for r in result_rows])
                     result_str = str(tuple(normalized_set))
                else:
                     result_str = str(result_rows)
                
                if result_str not in results_map:
                    results_map[result_str] = []
                results_map[result_str].append(cand.sql)
                cand.execution_result = result_rows
                
            except Exception as e:
                logger.warning(f"Execution failed for {cand.description}: {e}")
                cand.error_message = str(e)

        if not results_map:
            logger.error("All executions failed.")
            return candidates[0].sql

        # Find the result signature with the most votes
        best_sig = max(results_map, key=lambda k: len(results_map[k]))
        winning_sqls = results_map[best_sig]
        vote_count = len(winning_sqls)
        
        # Tie-breaker: Pick the shortest SQL among winners (Occam's Razor)
        winning_sqls.sort(key=len)
        final_winner = winning_sqls[0]
        
        logger.info(f"Winner selected with {vote_count}/{len(candidates)} votes. SQL: {final_winner}")
        return final_winner