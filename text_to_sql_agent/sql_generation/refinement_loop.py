"""
Algorithm 1: Iterative Schema Refinement (BIRD Paper).

For each of 5 schema variants:
1. Generate SQL with LLM
2. Extract fields and literals from SQL
3. Check if literals are covered by referenced fields
4. If not, augment schema and retry (max iterations)
5. Collect all discovered fields

Returns union of fields across all variants.
"""
import re
from typing import Optional, Tuple

SQL_START_KEYWORDS = ("WITH", "SELECT", "INSERT", "UPDATE", "DELETE", "MERGE")

from typing import Set, Tuple, List, Dict, Optional

from core import BaseLLMClient, create_llm_client, LLMMessage
from config.settings import settings
from indexing import SchemaLiteralMatcher
from schema_linking import (
    SchemaVariant,
    SchemaRepresentation
)
from profiling.field_metadata import FieldMetadata

from .types import (
    Algorithm1Result,
    VariantResult,
    IterationResult,
    LiteralMatch,
    SQLParseResult
)
from .sql_parser import SQLParser
from .schema_augmenter import SchemaAugmenter, create_revision_prompt


class Algorithm1Runner:
    """
    Implements Algorithm 1 from BIRD paper.
    
    Discovers relevant fields by iterative SQL generation and refinement
    across all 5 schema variants.
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        literal_matcher: SchemaLiteralMatcher,
        metadata_map: Dict[Tuple[str, str], FieldMetadata],
        max_literal_refinements: int = 2,  # Paper's MaxRetry
        max_syntax_fixes: int = 1,  # Syntax repair attempts per iteration
        sql_dialect: str = settings.db_type
    ):
        """
        Initialize Algorithm 1 runner.
        
        Args:
            llm_client: LLM for SQL generation
            literal_matcher: LSH index for finding literal matches
            metadata_map: Field metadata for schema augmentation
            max_literal_refinements: Max iterations for literal-based refinement (paper: MaxRetry)
            max_syntax_fixes: Max syntax repair attempts per generation (engineering safety)
            sql_dialect: SQL dialect for parsing
        """
        self.llm_client = llm_client
        self.literal_matcher = literal_matcher
        self.metadata_map = metadata_map
        self.max_literal_refinements = max_literal_refinements
        self.max_syntax_fixes = max_syntax_fixes
        
        # Build schema mapping {table: {columns}} from metadata
        self.schema = self._build_schema_from_metadata(metadata_map)
        
        self.sql_parser = SQLParser(dialect=sql_dialect, schema=self.schema)
        self.schema_augmenter = SchemaAugmenter(metadata_map)
    
    def _build_schema_from_metadata(
        self, 
        metadata_map: Dict[Tuple[str, str], FieldMetadata]
    ) -> Dict[str, Set[str]]:
        """Build schema dict {table: {columns}} from metadata_map."""
        schema = {}
        for (table, column), _ in metadata_map.items():
            if table not in schema:
                schema[table] = set()
            schema[table].add(column)
        return schema
    
    def run(
        self,
        question: str,
        schema_representations: Dict[SchemaVariant, SchemaRepresentation]
    ) -> Algorithm1Result:
        """
        Run Algorithm 1 on a question.
        
        Args:
            question: Natural language question
            schema_representations: Pre-generated schema for each variant
            
        Returns:
            Algorithm1Result with discovered fields
        """
        variant_results = []
        
        # Process each schema variant
        for variant in SchemaVariant:
            if variant not in schema_representations:
                continue
            
            schema_rep = schema_representations[variant]
            
            print(f"\n  Processing variant: {variant.value}")
            
            # Run refinement loop for this variant
            vr = self._process_variant(question, schema_rep)
            variant_results.append(vr)
            
            print(f"    Iterations: {vr.num_refinements}, Fields found: {len(vr.final_fields)}")
        
        # Compute final union
        final_fields = set()
        final_literals = set()
        total_iterations = 0
        total_sql = 0
        
        for vr in variant_results:
            final_fields.update(vr.final_fields)
            final_literals.update(vr.final_literals)  # Both are sets
            total_iterations += len(vr.iterations)
            total_sql += len(vr.iterations)
        
        return Algorithm1Result(
            question=question,
            variant_results=variant_results,
            final_fields=final_fields,
            final_literals=final_literals,
            total_iterations=total_iterations,
            total_sql_generated=total_sql
        )
    
    def _process_variant(
        self,
        question: str,
        schema_rep: SchemaRepresentation
    ) -> VariantResult:
        """
        Process one schema variant with refinement loop.
        
        Implements steps 2a-2f from Algorithm 1.
        """
        iterations = []
        current_schema = schema_rep
        already_augmented = set()  # Track fields already added
        
        # Literal refinement loop (paper's MaxRetry)
        for literal_iter in range(self.max_literal_refinements + 1):
            # Step 2a: Generate SQL
            previous = iterations[-1] if iterations else None
            
            iter_result = self._generate_and_analyze(
                question=question,
                schema_rep=current_schema,
                variant=schema_rep.variant,
                iteration=literal_iter,
                previous_iteration=previous
            )
            
            iterations.append(iter_result)
            
            # If SQL is invalid, stop here (don't augment on bad SQL)
            if not iter_result.is_valid_sql:
                if literal_iter < self.max_literal_refinements:
                    print(f"      Invalid SQL after fix, retrying with same schema (attempt {literal_iter + 2}/{self.max_literal_refinements + 1})")
                    continue
                else:
                    print(f"      Invalid SQL, max literal refinements reached")
                    break
            
            # Check if refinement needed (only for valid SQL)
            if not iter_result.missing_literals or literal_iter >= self.max_literal_refinements:
                # No missing literals or max refinements reached
                break
            
            # Step 2e: Augment schema with NEW fields only
            new_aug_fields = iter_result.augmented_fields - already_augmented
            if not new_aug_fields:
                # No new fields to add, converged
                break
            
            current_schema = self.schema_augmenter.create_augmented_representation(
                current_schema,
                new_aug_fields
            )
            already_augmented.update(new_aug_fields)
        
        # Collect final fields and literals
        final_fields = set()
        final_literals = set()
        
        for it in iterations:
            final_fields.update(it.fields_used)
            final_literals.update(it.literals_used)  # Set.update works with sets
        
        return VariantResult(
            variant=schema_rep.variant,
            iterations=iterations,
            final_fields=final_fields,  # Union across all iterations (recall mode)
            final_literals=final_literals,  # Already a set
            num_refinements=len(iterations) - 1,
            converged=len(iterations[-1].missing_literals) == 0 if iterations else False
        )
    
    def _generate_and_analyze(
        self,
        question: str,
        schema_rep: SchemaRepresentation,
        variant: SchemaVariant,
        iteration: int,
        previous_iteration: Optional[IterationResult] = None
    ) -> IterationResult:
        """
        Generate SQL and analyze for missing literals.
        
        Implements steps 2a-2d from Algorithm 1.
        """
        # Step 2a: Generate SQL
        use_revision = (
            previous_iteration is not None
            and previous_iteration.is_valid_sql
            and len(previous_iteration.missing_literals) > 0
        )
        
        if use_revision:
            # Create revision prompt with previous SQL and missing literals
            # Build literal_fields map from previous iteration
            literal_fields = {}
            for match in previous_iteration.literal_matches:
                if match.literal not in literal_fields:
                    literal_fields[match.literal] = set()
                literal_fields[match.literal].add((match.table, match.column))
            
            revision_prompt = create_revision_prompt(
                question=question,
                original_sql=previous_iteration.sql,
                missing_literals=previous_iteration.missing_literals,
                literal_fields=literal_fields
            )
            
            # Add augmented schema
            prompt = f"{revision_prompt}\n\n{schema_rep.text}"
        else:
            # Initial generation
            prompt = self._create_sql_prompt(question, schema_rep.text)
        
        sql, llm_response = self._call_llm_for_sql(prompt)
        
        # Step 2b: Parse and attempt syntax fix if needed
        parse_result = self.sql_parser.parse(sql)
        
        # If invalid SQL, attempt ONE syntax fix per iteration
        if not parse_result.is_valid and parse_result.parse_error:
            print(f"        SQL parse failed: {parse_result.parse_error[:100]}")
            print(f"        Attempting syntax fix...")
            
            sql, llm_response = self._attempt_syntax_fix(
                question=question,
                schema_text=schema_rep.text,
                bad_sql=sql,
                parse_error=parse_result.parse_error
            )
            
            # Re-parse
            parse_result = self.sql_parser.parse(sql)
            
            if parse_result.is_valid:
                print(f"         Syntax fixed successfully")
            else:
                print(f"         Syntax fix failed, SQL still invalid")
        
        # Extract fields and literals (from final SQL, valid or not)
        fields_q = parse_result.referenced_fields
        lits_q = parse_result.literals
        
        # Step 2c-2d: Find literal matches
        lit_fields_q = set()
        missing_lits = []
        literal_matches = []
        
        for literal in lits_q:
            # Step 2d-i: Find fields containing this literal
            matches = self.literal_matcher.find_matching_fields(literal, top_k=5)
            
            # Check if any matching field is in FieldsQ
            matched_in_sql = False
            for match in matches:
                key = (match.table, match.column)
                
                literal_matches.append(
                    LiteralMatch(
                        literal=literal,
                        table=match.table,
                        column=match.column,
                        score=match.score
                    )
                )
                
                if key in fields_q:
                    matched_in_sql = True
            
            # Step 2d-ii: If no field in FieldsQ contains literal
            if not matched_in_sql and matches:
                missing_lits.append(literal)
                for match in matches:
                    lit_fields_q.add((match.table, match.column))
        
        return IterationResult(
            variant=variant,
            iteration=iteration,
            sql=sql,
            is_valid_sql=parse_result.is_valid,
            fields_used=fields_q,
            literals_used=lits_q,
            literal_matches=literal_matches,
            missing_literals=missing_lits,
            augmented_fields=lit_fields_q,
            schema_used=schema_rep.text,
            llm_prompt=prompt,
            llm_response=llm_response
        )
    
    def _create_sql_prompt(self, question: str, schema_text: str) -> str:
        """Create prompt for SQL generation."""
        return f"""{schema_text}

Question: {question}

Generate a valid {settings.db_type} query to answer this question.

Output format rules (must follow):
- Output MUST be a single {settings.db_type} SQL statement.
- Do NOT include markdown fences/backticks, JSON, comments, or explanations.
- Do NOT prefix with “SQL:” or any label.
"""
    
    def _call_llm_for_sql(self, prompt: str, is_syntax_fix: bool = False) -> Tuple[str, str]:
        """
        Call LLM to generate SQL.
        
        Args:
            prompt: Prompt for SQL generation
            is_syntax_fix: Whether this is a syntax repair attempt
            
        Returns:
            (sql, full_response) tuple
        """
        # Increase token limit for reasoning space (800-1000 recommended)
        max_tokens = 2000
        
        # Wrap prompt in LLMMessage
        messages = [LLMMessage(role="user", content=prompt)]
        llm_response = self.llm_client.generate(messages, max_tokens=max_tokens)
        
        # === DEBUG LOG ===
        print(f"        [DEBUG] Raw LLM Response: {repr(llm_response.content)}")
        
        # Extract text from LLMResponse
        response_text = llm_response.content
        
        # Extract SQL from response (remove markdown, explanations)
        sql = self._extract_sql_from_response(response_text)
        
        return sql, response_text
    
    def _attempt_syntax_fix(
        self,
        question: str,
        schema_text: str,
        bad_sql: str,
        parse_error: str
    ) -> Tuple[str, str]:
        """
        Attempt to fix syntax errors in SQL.
        
        One repair attempt per iteration to avoid wasting literal-refinement retries.
        
        Args:
            question: Original question
            schema_text: Schema context
            bad_sql: SQL with syntax error
            parse_error: Error message from parser
            
        Returns:
            (fixed_sql, llm_response) tuple
        """
        fix_prompt = f"""The following SQL query has a syntax error:

```sql
{bad_sql}
```

Parse error: {parse_error}

Question: {question}

Schema:
{schema_text}

Please fix the syntax error. 

Rules:
- Return ONLY the corrected SQL (no markdown fences, no explanations, no comments).
- Preserve the original intent and structure as much as possible.
- Only introduce new tables if they are strictly required by the existing query logic."""
        
        return self._call_llm_for_sql(fix_prompt, is_syntax_fix=True)
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL from an LLM response without corrupting CTEs (WITH ...)."""
        text = (response or "").strip()

        if not text:
            print("        [WARN] Empty LLM response received")
            return ""

        # 1) Prefer fenced code blocks (```sql ...``` or ``` ... ```)
        fenced = self._extract_first_sql_fence(text)
        if fenced:
            sql = fenced.strip()
            return self._final_cleanup_sql(sql, raw=response)

        # 2) No fences: find earliest SQL start keyword and slice from there
        sql = self._slice_from_earliest_sql_start(text)
        return self._final_cleanup_sql(sql, raw=response)


    def _extract_first_sql_fence(self, text: str) -> Optional[str]:
        """
        Return best candidate from fenced code blocks.
        Picks first block that looks like SQL, else longest block.
        """
        # Matches ```sql ... ``` OR ``` ... ```
        pattern = re.compile(r"```(?:\s*(\w+))?\s*\n([\s\S]*?)```", re.IGNORECASE)
        blocks = [(m.group(1) or "", m.group(2)) for m in pattern.finditer(text)]
        if not blocks:
            return None

        def looks_like_sql(s: str) -> bool:
            s2 = s.lstrip()
            up = s2[:50].upper()
            return any(up.startswith(k) for k in SQL_START_KEYWORDS)

        # Prefer a fence explicitly marked sql/oracle/etc + looks like SQL
        for lang, body in blocks:
            if lang.lower() in {"sql", "oracle", "postgres", "postgresql", "mysql", "sqlite", "tsql"} and looks_like_sql(body):
                return body

        # Otherwise, first block that looks like SQL
        for _, body in blocks:
            if looks_like_sql(body):
                return body

        # Otherwise, fallback to longest block (least bad)
        return max((b for _, b in blocks), key=len, default=None)


    def _slice_from_earliest_sql_start(self, text: str) -> str:
        """
        Find earliest occurrence among SQL_START_KEYWORDS and slice from it.
        IMPORTANT: uses *earliest index*, not keyword priority order.
        """
        upper = text.upper()

        best_idx = None
        for kw in SQL_START_KEYWORDS:
            i = upper.find(kw)
            if i != -1:
                if best_idx is None or i < best_idx:
                    best_idx = i

        if best_idx is None:
            # Nothing looks like SQL; return as-is (caller will log empties)
            return text

        return text[best_idx:]


    def _final_cleanup_sql(self, sql: str, raw: str) -> str:
        """Minimal safe cleanup. Never delete mid-query content."""
        if sql is None:
            sql = ""
        s = sql.strip()

        # If the whole thing is wrapped in a single pair of quotes, unwrap once.
        if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()

        # Remove leading "SQL:" / "Query:" only if it appears at the very beginning.
        s = re.sub(r"^\s*(SQL|QUERY)\s*:\s*", "", s, flags=re.IGNORECASE).strip()

        # Remove stray leading language token (rare but happens): "sql\nWITH ..."
        s = re.sub(r"^\s*sql\s*\n", "", s, flags=re.IGNORECASE).strip()

        # Optional: strip a single trailing semicolon (safe for single-statement pipelines)
        if s.endswith(";"):
            s = s[:-1].rstrip()

        if not s:
            print("        [WARN] Empty SQL after extraction. Raw response (first 300 chars):")
            print(f"        {repr((raw or '')[:300])}")

        return s   