"""
Heuristic SQL Linter / Validator.
Detects common BIRD anti-patterns and SQL logic flaws before execution.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Set
import sqlglot
from sqlglot import exp
from config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class LintResult:
    is_valid: bool              # Can this SQL execute? (Syntax check)
    needs_correction: bool      # Should we try to fix it? (Style/Heuristic check)
    error_message: Optional[str] = None

class SQLLinter:
    """
    Analyzes SQL for logical anti-patterns common in BIRD/Spider.
    Distinguishes between 'Syntax Errors' (invalid) and 'Anti-Patterns' (needs correction).
    """
    
    def __init__(self, dialect: str = settings.db_dialect):
        self.dialect = dialect

    def lint(self, sql: str) -> LintResult:
        """
        Run all heuristic checks on the SQL.
        """
        try:
            # 1. Basic Syntax Check (via sqlglot)
            expression = sqlglot.parse_one(sql, read=self.dialect)
        except Exception as e:
            return LintResult(
                is_valid=False, 
                needs_correction=True,
                error_message=f"Syntax Error: {str(e)}"
            )

        # 2. Run Heuristic Checks (Style / BIRD Patterns)

        # Check A: NULL value ordering (The "Top 1" Bug)
        # This IS risky because it returns NULLs instead of data. Keep it.
        if self._detect_risky_null_ordering(expression):
             return LintResult(
                 is_valid=True,
                 needs_correction=True,
                 error_message="Heuristic Warning: ORDER BY ASC with LIMIT 1 on potentially nullable column (missing IS NOT NULL)."
             )

        # Check B: String concatenation (BIRD prefers separate columns)
        # Keep this; separate columns are easier to evaluate.
        if self._detect_string_concatenation(expression):
            return LintResult(
                is_valid=True,
                needs_correction=True,
                error_message="Heuristic Warning: Detected string concatenation (||). Prefer returning separate columns."
            )

        # Check C: Nested subquery for MIN/MAX
        # REMOVED: This rule is harmful. BIRD questions often require returning ALL ties 
        # (e.g. "Who has the most strength?" -> multiple rows).
        # 'ORDER BY ... LIMIT 1' arbitrarily cuts data. We should NOT warn against nested queries.
        
        # if self._detect_nested_min_max(expression):
        #      return LintResult(
        #         is_valid=True,
        #         needs_correction=True,
        #         error_message="Heuristic Warning: Detected nested subquery for Min/Max. Prefer 'ORDER BY column LIMIT 1'."
        #     )

        return LintResult(is_valid=True, needs_correction=False)

    def _detect_risky_null_ordering(self, expression: exp.Expression) -> bool:
        """
        Heuristic: If query has ORDER BY <col> ASC and LIMIT 1,
        and there is no WHERE <col> IS NOT NULL check, flag it.
        """
        limit = expression.find(exp.Limit)
        if not limit:
            return False
            
        limit_expr = limit.expression
        if not (isinstance(limit_expr, exp.Literal) and limit_expr.this == "1"):
            return False

        orders = expression.find_all(exp.Order)
        risky_columns = []
        
        for order in orders:
            for ordered in order.expressions:
                is_desc = ordered.args.get("desc")
                if not is_desc:
                    if isinstance(ordered.this, exp.Column):
                        risky_columns.append(ordered.this.name)

        if not risky_columns:
            return False

        where = expression.find(exp.Where)
        checked_columns = set()
        
        if where:
            for node in where.walk():
                if isinstance(node, exp.Is) and isinstance(node.this, exp.Column):
                    if isinstance(node.expression, exp.Null):
                        if node.args.get("not"):
                            checked_columns.add(node.this.name)

        for col in risky_columns:
            if col not in checked_columns:
                return True
                
        return False

    def _detect_string_concatenation(self, expression: exp.Expression) -> bool:
        """Check for || operator (DPipe) or CONCAT function in projections."""
        for node in expression.find_all(exp.Select):
            for projection in node.expressions:
                for item in projection.walk():
                    if isinstance(item, exp.DPipe):
                        return True
                    if isinstance(item, exp.Concat):
                        return True
                    if isinstance(item, exp.Func) and item.sql_name().upper() == "CONCAT":
                        return True
        return False

    def _detect_nested_min_max(self, expression: exp.Expression) -> bool:
        """
        (Deprecated / Unused)
        Check if WHERE/HAVING clause uses a subquery containing MIN/MAX.
        """
        scopes = list(expression.find_all(exp.Where)) + list(expression.find_all(exp.Having))
        
        for scope in scopes:
            for subquery in scope.find_all(exp.Subquery):
                for agg in subquery.find_all(exp.Min, exp.Max):
                    return True
        return False