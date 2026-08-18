"""
Candidate Generator (Phase 4).
Takes the discovered fields (from Algorithm 1) and generates diverse SQL candidates
using BIRD-style diversity techniques (shuffling + seeds).
Includes retry logic, observability logging, and tie-aware prompting.
"""

import random
import logging
from typing import List, Set, Dict, Optional
from dataclasses import dataclass

from core import BaseLLMClient, LLMMessage
from profiling.field_metadata import FieldMetadata
from .few_shot_store import FewShotExample

logger = logging.getLogger(__name__)

@dataclass
class GeneratedSQL:
    sql: str
    description: str

class CandidateGenerator:
    """
    Generates diverse SQL candidates for voting.
    Techniques:
    1. Schema Shuffling: Randomize field order in prompt.
    2. Seed: Vary LLM generation seed (if supported).
    """
    
    def __init__(self, llm_client: BaseLLMClient, rng_seed: Optional[int] = None):
        self.llm_client = llm_client
        # Use a local Random instance for reproducibility
        self.rng = random.Random(rng_seed)

    def generate_diverse_candidates(
        self,
        question: str,
        final_fields: Set[tuple],  # Set of (table, column)
        metadata_map: Dict[tuple, FieldMetadata],
        few_shots: List[FewShotExample],
        num_candidates: int = 3
    ) -> List[GeneratedSQL]:
        """
        Generate N diverse SQL candidates using the reduced schema.
        """
        candidates = []
        
        # Base list of fields
        # FIX: Sort the list first so that shuffling with a seed is deterministic
        base_fields_list = sorted(list(final_fields))
        
        # OBSERVABILITY: Log if schema is large, but do NOT truncate.
        if len(base_fields_list) > 80:
            logger.warning(
                f"Large schema context detected ({len(base_fields_list)} fields). "
                "Proceeding without truncation to preserve recall."
            )
        logger.info(f"[DEBUG]    Fields passed to candidate generator: {[f'{t}.{c}' for t, c in base_fields_list]}")

        
        for i in range(num_candidates):
            # 1. Diversity Technique: Field Shuffling
            shuffled_fields = base_fields_list[:]
            self.rng.shuffle(shuffled_fields)
            
            # 2. Build Prompt
            prompt = self._build_prompt(question, shuffled_fields, metadata_map, few_shots)
            
            # 3. Diversity Technique: Seed
            base_seed = self.rng.randint(0, 10_000)
            
            # RETRY LOOP: If output is empty, try again with a DIFFERENT seed
            for attempt in range(2):
                # VARY SEED ON RETRY to avoid deterministic failure
                current_seed = base_seed + attempt
                
                try:
                    logger.info(f"Generating candidate {i+1}/{num_candidates} (Seed: {current_seed}, Attempt: {attempt+1})")
                    
                    response = self.llm_client.generate(
                        [LLMMessage(role="user", content=prompt)],
                        max_tokens=2000, # Increased for safety
                        seed=current_seed,
                        temperature=None 
                    )
                    
                    sql = self._clean_sql(response.content)
                    if sql:
                        candidates.append(GeneratedSQL(
                            sql=sql,
                            description=f"Candidate {i+1} (Shuffle + Seed {current_seed})"
                        ))
                        break # Success
                    else:
                        logger.warning(f"Candidate {i+1} produced empty SQL. Retrying with new seed...")
                        
                except Exception as e:
                    logger.error(f"Failed to generate candidate {i+1} (Attempt {attempt+1}): {e}")
            else:
                logger.error(f"Candidate {i+1} failed after retries.")
        
        for idx, cand in enumerate(candidates):
            logger.info(f"[DEBUG]    Generated Candidate {idx+1} SQL: {cand.sql}")
                    
        return candidates

    def _build_prompt(
        self, 
        question: str, 
        fields_list: List[tuple], 
        metadata_map: Dict[tuple, FieldMetadata],
        few_shots: List[FewShotExample]
    ) -> str:
        """Construct the final prompt with few-shots and the shuffled schema."""
        
        prompt_parts = []
        
        # 1. Add Few-Shot Examples
        if few_shots:
            prompt_parts.append("<EXAMPLES>")
            prompt_parts.append("Reference Examples (structurally similar):")
            prompt_parts.append("")
            for ex in few_shots:
                prompt_parts.append("<SQL_EXAMPLE>")
                prompt_parts.append(f"Question: {ex.original_question}")
                prompt_parts.append("SQL:")
                prompt_parts.append(ex.sql.strip())
                prompt_parts.append("</SQL_EXAMPLE>")
            prompt_parts.append("</EXAMPLES>")
        
        # 2. Add Schema Context
        prompt_parts.append("<SCHEMA>")
        prompt_parts.append("Database Schema (Relevant Fields Only):")
        prompt_parts.append("")
        
        schema_lines = []
        for table, col in fields_list:
            meta = metadata_map.get((table, col))
            if meta:
                # Robust attribute access
                desc = (
                    getattr(meta, "sme_description", None)
                    or getattr(meta, "sme_desc", None)
                    or getattr(meta, "long_description", None)
                    or getattr(meta, "short_description", None)
                )
                
                if not desc:
                    desc = "No description available."
                
                # Robust type access
                col_type = "UNKNOWN"
                if meta.profile:
                    col_type = getattr(meta.profile, 'data_type', 
                                getattr(meta.profile, 'col_type', 
                                    getattr(meta.profile, 'type', "UNKNOWN")))
                
                schema_lines.append(f"Table {table}, Column {col}: {desc} (Type: {col_type})")
            else:
                # FALLBACK: If metadata is missing, still include the column!
                schema_lines.append(f"Table {table}, Column {col}: (No description available)")
            
        prompt_parts.extend(schema_lines)
        prompt_parts.append("</SCHEMA>")
        
        # 3. Add Question
        prompt_parts.append("\n<QUESTION>")
        prompt_parts.append(question)
        prompt_parts.append("</QUESTION>")
        
        # 4. Add Instructions
        prompt_parts.append("\n<INSTRUCTIONS>")
        prompt_parts.append("Generate a valid {settings.db_type} query for the question above using ONLY the provided schema.")
        prompt_parts.append("")
        prompt_parts.append("Output format rules (must follow):")
        prompt_parts.append("- Output MUST be a single {settings.db_type} SQL statement.")
        prompt_parts.append("- Do NOT include markdown fences/backticks, JSON, comments, or explanations.")
        prompt_parts.append("- End output immediately after the SQL statement.")
        prompt_parts.append("</INSTRUCTIONS>")
        
        # # Tie-Aware Policy
        # prompt_parts.append("IMPORTANT: If the question asks for the 'most', 'least', or 'highest' (superlatives), return ALL matching rows (ties) using a subquery, unless the question explicitly asks for a single result (e.g. 'top 1').")
        
        return "\n".join(prompt_parts)

    def _clean_sql(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            newline_idx = cleaned.find("\n")
            if newline_idx != -1:
                cleaned = cleaned[newline_idx+1:]
            end_idx = cleaned.rfind("```")
            if end_idx != -1:
                cleaned = cleaned[:end_idx]

        cleaned = cleaned.strip().rstrip(';')
        return cleaned.strip()