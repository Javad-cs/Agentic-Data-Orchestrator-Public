"""
SQL parser for extracting referenced fields and literals.
Uses sqlglot for robust SQL parsing with alias resolution.
Enhanced with Oracle database link (@DBNAME) support.
"""

from typing import Set, List, Tuple, Optional, Dict
import re
import sqlglot
from sqlglot import exp

from .types import SQLParseResult
from config.settings import settings


class SQLParser:
    """
    Parse SQL queries to extract:
    - Referenced tables and columns (FieldsQ)
    - Literals in WHERE clauses (LitsQ)
    
    Supports Oracle database links (TABLE@DBLINK syntax).
    """
    
    def __init__(self, dialect: str = settings.db_type, schema: Optional[Dict[str, Set[str]]] = None):
        """
        Initialize parser.
        
        Args:
            dialect: SQL dialect (sqlite, postgres, mysql, oracle, etc.)
            schema: Optional schema mapping {table: {columns}} for resolving unqualified columns
        """
        self.dialect = dialect
        self.schema = schema or {}
        self._dblink_map = {}  # Store database link mappings for restoration
        
        # Performance optimization (Fix #4): Pre-compute uppercase schema
        self._schema_upper = {}
        for table, columns in self.schema.items():
            self._schema_upper[table] = {c.upper() for c in columns}
    
    def _preprocess_oracle_dblinks(self, sql: str) -> str:
        """
        Replace Oracle database links (@DBNAME) with placeholders before parsing.
        
        Oracle syntax: TABLE@DBLINK (e.g., EQP_MST@LINKED_DATABASE)
        sqlglot doesn't recognize this syntax, so we temporarily replace it with
        valid identifiers that can be parsed, then restore them later.
        
        Args:
            sql: Original SQL with database links
            
        Returns:
            SQL with database links replaced by placeholders
        """
        if self.dialect != "oracle":
            return sql
        
        # Reset mapping for this parse
        self._dblink_map = {}
        
        # Pattern: word@word (e.g., EQP_MST@LINKED_DATABASE, DEPT_MST@LINKED_DATABASE)
        # Must be word boundary to avoid matching email addresses
        pattern = r'\b(\w+)@(\w+)\b'
        
        def replace_dblink(match):
            full_name = match.group(0)  # e.g., "EQP_MST@LINKED_DATABASE"
            table = match.group(1)       # e.g., "EQP_MST"
            dblink = match.group(2)      # e.g., "LINKED_DATABASE"
            
            # Create placeholder using underscore (valid SQL identifier)
            # e.g., "EQP_MST_DBLINK_LINKED_DATABASE"
            placeholder = f"{table}_DBLINK_{dblink}"
            
            # Store mapping for later restoration
            self._dblink_map[placeholder] = full_name
            
            return placeholder
        
        # Replace all database links with placeholders
        processed_sql = re.sub(pattern, replace_dblink, sql, flags=re.IGNORECASE)
        
        return processed_sql
    
    def _restore_oracle_dblinks(self, fields: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        """
        Restore Oracle database link syntax in extracted table names.
        
        Converts placeholders like "EQP_MST_DBLINK_LINKED_DATABASE" back to "EQP_MST@LINKED_DATABASE".
        
        Args:
            fields: Set of (table, column) tuples with placeholder table names
            
        Returns:
            Set of (table, column) tuples with original database link syntax
        """
        if not self._dblink_map:
            return fields
        
        restored = set()
        for table, column in fields:
            # Check if table is a placeholder we created
            # If so, restore original name; otherwise keep as-is
            original_table = self._dblink_map.get(table, table)
            restored.add((original_table, column))
        
        return restored
    
    def parse(self, sql: str) -> SQLParseResult:
        """
        Parse SQL query with Oracle database link support.
        
        Uses physical field extraction (filters CTEs, subqueries, derived columns).
        
        Args:
            sql: SQL query string (may contain @DBLINK syntax)
            
        Returns:
            SQLParseResult with extracted fields and literals
        """
        try:
            # Pre-process database links for parsing
            processed_sql = self._preprocess_oracle_dblinks(sql)
            
            # Try to parse with sqlglot
            parsed = sqlglot.parse_one(processed_sql, read=self.dialect)
            
            # Extract PHYSICAL fields only
            fields = self._extract_physical_fields(parsed)
            
            # Restore database link syntax in fields
            fields = self._restore_oracle_dblinks(fields)
            
            # Extract literals
            literals = self._extract_literals(parsed)
            
            return SQLParseResult(
                sql=sql,
                referenced_fields=fields,
                literals=literals,
                is_valid=True
            )
            
        except Exception as e:
            # Parsing failed - return as invalid
            return SQLParseResult(
                sql=sql,
                referenced_fields=set(),
                literals=set(),
                is_valid=False,
                parse_error=str(e)
            )
    
    def _extract_physical_fields(self, parsed: exp.Expression) -> Set[Tuple[str, str]]:
        """
        Extract ONLY fields that resolve to physical base tables.
        
        Strategy:
        1. Build scope context (identify CTEs, subqueries, real tables)
        2. For each column reference, recursively resolve to base tables
        3. Validate against schema (strict mode)
        4. Return only (table, column) pairs that exist in schema
        
        Returns:
            Set of (table, column) tuples - all validated against schema
        """
        # Build the scope/context tree
        scope_context = self._build_scope_context(parsed)
        
        # Extract all column references
        all_columns = list(parsed.find_all(exp.Column))
        
        # Resolve each column to physical fields
        physical_fields = set()
        
        for col in all_columns:
            table_ref = col.table  # May be alias, CTE name, or real table
            column_name = col.name
            
            if not column_name:
                continue
            
            # Resolve this reference to physical base tables
            resolved = self._resolve_column_to_physical(
                table_ref=table_ref,
                column_name=column_name,
                scope_context=scope_context
            )
            
            physical_fields.update(resolved)
        
        return physical_fields
    
    def _build_scope_context(self, parsed: exp.Expression) -> Dict[str, any]:
        """
        Build a scope context that tracks:
        - CTEs: name -> definition AST
        - Subqueries: alias -> definition AST
        - Real tables: name -> name (with aliases resolved)
        - Table aliases: alias -> real table name
        
        Fix #3: Uses scope-aware CTE detection (not global)
        
        Returns:
            Dict with keys: 'ctes', 'subqueries', 'table_aliases', 'real_tables'
        """
        context = {
            'ctes': {},           # CTE_name (upper) -> AST node
            'subqueries': {},     # alias (upper) -> AST node  
            'table_aliases': {},  # alias (upper) -> real_table_name (RESTORED @DBLINK)
            'real_tables': set()  # Set of real table names (RESTORED @DBLINK)
        }
        
        # Extract CTEs (WITH clauses) - scope-aware (Fix #3)
        # Only get CTEs at the top level of this parsed expression
        cte_names = set()
        if hasattr(parsed, 'ctes') and parsed.ctes:
            for cte in parsed.ctes:
                cte_name = cte.alias_or_name
                if cte_name:
                    cte_name_str = cte_name if isinstance(cte_name, str) else str(cte_name)
                    cte_name_upper = cte_name_str.upper()
                    context['ctes'][cte_name_upper] = cte.this  # Store the SELECT inside WITH
                    cte_names.add(cte_name_upper)
        
        # Extract ALL tables using brute-force, then filter CTEs
        for table_node in parsed.find_all(exp.Table):
            table_name = table_node.name
            if not table_name:
                continue
            
            # Normalize table name (keep placeholder for now)
            table_name_str = table_name if isinstance(table_name, str) else str(table_name)
            table_name_upper = table_name_str.upper()
            
            # Skip if it's a CTE reference
            if table_name_upper in cte_names:
                continue
            
            # Restore @DBLINK ONCE here (single restoration point)
            restored_table = self._dblink_map.get(table_name_str, table_name_str)
            restored_table = restored_table.upper()
            
            # It's a real table - store with restored name
            context['real_tables'].add(restored_table)
            
            # Handle alias (proper string extraction)
            alias = table_node.alias
            if alias:
                # Extract string from alias
                if isinstance(alias, str):
                    alias_str = alias
                elif hasattr(alias, 'name'):
                    alias_str = alias.name
                elif hasattr(alias, 'this'):
                    alias_str = str(alias.this)
                else:
                    alias_str = str(alias)
                
                # Store with uppercase key, restored table name
                context['table_aliases'][alias_str.upper()] = restored_table
            else:
                # Map table to itself for consistency (uppercase key)
                context['table_aliases'][table_name_upper] = restored_table
        
        # Extract subqueries
        for subq in parsed.find_all(exp.Subquery):
            alias = subq.alias_or_name
            if alias:
                alias_str = alias if isinstance(alias, str) else (
                    alias.name if hasattr(alias, 'name') else str(alias)
                )
                context['subqueries'][alias_str.upper()] = subq.this
        
        return context
    
    def _resolve_column_to_physical(
        self,
        table_ref: Optional[str],
        column_name: str,
        scope_context: Dict
    ) -> Set[Tuple[str, str]]:
        """
        Resolve a column reference to physical base table(s).
        
        Args:
            table_ref: Table reference. None if unqualified.
            column_name: Column name
            scope_context: Scope context
            
        Returns:
            Set of (table, column) tuples that exist in schema
        """
        results = set()
        
        # Normalize inputs
        table_ref_upper = table_ref.upper() if table_ref else None
        column_name_upper = column_name.upper()
        
        # Case 1: Qualified column
        if table_ref_upper:
            # Check if it's a CTE
            if table_ref_upper in scope_context['ctes']:
                cte_select = scope_context['ctes'][table_ref_upper]
                resolved = self._resolve_through_select(column_name_upper, cte_select, scope_context)
                results.update(resolved)
            
            # Check if it's a subquery alias
            elif table_ref_upper in scope_context['subqueries']:
                subquery_select = scope_context['subqueries'][table_ref_upper]
                resolved = self._resolve_through_select(column_name_upper, subquery_select, scope_context)
                results.update(resolved)
            
            # Check if it's a table alias
            elif table_ref_upper in scope_context['table_aliases']:
                real_table = scope_context['table_aliases'][table_ref_upper]  # Already restored
                if self._is_valid_field_fast(real_table, column_name_upper):
                    results.add((real_table, column_name_upper))
            
            # Otherwise, assume it's a real table (unaliased)
            else:
                restored_table = self._dblink_map.get(table_ref, table_ref)
                if self._is_valid_field_fast(restored_table, column_name_upper):
                    results.add((restored_table, column_name_upper))
        
        # Case 2: Unqualified column
        else:
            resolved_tables = self._resolve_unqualified_column(
                column_name_upper,
                scope_context['real_tables'],
                max_candidates=2
            )
            
            for table in resolved_tables:
                results.add((table, column_name_upper))
        
        return results
    
    def _resolve_through_select(
        self,
        column_name: str,  # Already uppercase
        select_node: exp.Select,
        scope_context: Dict
    ) -> Set[Tuple[str, str]]:
        """
        Resolve a column through a SELECT (from CTE or subquery).
        
        Args:
            column_name: Column to find (uppercase)
            select_node: The SELECT AST node
            scope_context: Scope context
            
        Returns:
            Set of physical (table, column) pairs
        """
        results = set()
        
        # Get the select expressions (only immediate select list)
        expressions = select_node.args.get('expressions', [])
        
        # Check for SELECT *
        for expr in expressions:
            if isinstance(expr, exp.Star):
                # SELECT * found - expand using schema
                return self._expand_select_star(column_name, select_node, scope_context)
        
        # Check each projection in the select list
        for expr in expressions:
            if isinstance(expr, exp.Alias):
                # Aliased projection
                proj_alias = expr.alias
                proj_expr = expr.this
                
                # Normalize alias
                if proj_alias:
                    proj_alias_str = proj_alias if isinstance(proj_alias, str) else (
                        proj_alias.name if hasattr(proj_alias, 'name') else str(proj_alias)
                    )
                    proj_alias_upper = proj_alias_str.upper()
                else:
                    continue
                
                if proj_alias_upper == column_name:
                    # Found the projection
                    source_columns = self._extract_columns_from_expression(proj_expr, scope_context)
                    results.update(source_columns)
            
            elif isinstance(expr, exp.Column):
                # Unnamed column projection
                expr_name_upper = expr.name.upper() if expr.name else None
                if expr_name_upper == column_name:
                    resolved = self._resolve_column_to_physical(
                        table_ref=expr.table,
                        column_name=expr.name,
                        scope_context=scope_context
                    )
                    results.update(resolved)
        
        return results
    
    def _expand_select_star(
        self,
        column_name: str,  # Already uppercase
        select_node: exp.Select,
        scope_context: Dict
    ) -> Set[Tuple[str, str]]:
        """
        Handle SELECT * by expanding using schema.
        
        Fix #1 & #2: Only scan IMMEDIATE FROM/JOIN relations (no recursion into subqueries)
        
        Args:
            column_name: Column being requested (uppercase)
            select_node: SELECT node with * projection
            scope_context: Scope context
            
        Returns:
            Set of physical (table, column) pairs
        """
        results = set()
        from_tables = set()
        cte_names = set(scope_context['ctes'].keys())
        
        # Extract immediate FROM relation (Fix #2 - no recursion)
        if select_node.args.get('from'):
            from_relation = select_node.args['from'].this
            self._extract_immediate_table(from_relation, from_tables, cte_names)
        
        # Extract immediate JOIN relations (Fix #1 - only join.this, not find_all)
        for join in select_node.args.get('joins', []):
            if join.this:
                self._extract_immediate_table(join.this, from_tables, cte_names)
        
        # For each table, check if it has this column
        for table in from_tables:
            if self._is_valid_field_fast(table, column_name):
                results.add((table, column_name))
        
        return results
    
    def _extract_immediate_table(
        self,
        relation: exp.Expression,
        from_tables: Set[str],
        cte_names: Set[str]
    ):
        """
        Extract table name from an immediate relation (Table or aliased Table).
        
        Does NOT recurse into Subquery - treats it as opaque.
        
        Args:
            relation: FROM/JOIN relation expression
            from_tables: Set to add table names to (mutated)
            cte_names: Set of CTE names to skip
        """
        if isinstance(relation, exp.Table):
            table_name = relation.name
            if table_name:
                table_str = table_name if isinstance(table_name, str) else str(table_name)
                table_upper = table_str.upper()
                
                # Skip CTEs
                if table_upper in cte_names:
                    return
                
                # Restore @DBLINK
                restored_table = self._dblink_map.get(table_str, table_str)
                restored_table = restored_table.upper()
                from_tables.add(restored_table)
        
        # If it's a Subquery, don't descend - it's a separate scope
        # (We already handle subqueries via scope_context['subqueries'])
    
    def _extract_columns_from_expression(
        self,
        expr: exp.Expression,
        scope_context: Dict
    ) -> Set[Tuple[str, str]]:
        """
        Extract all base columns from an expression (handles computed columns).
        
        Args:
            expr: Expression AST node
            scope_context: Scope context
            
        Returns:
            Set of physical (table, column) pairs
        """
        results = set()
        
        # Find all Column nodes in the expression
        for col in expr.find_all(exp.Column):
            resolved = self._resolve_column_to_physical(
                table_ref=col.table,
                column_name=col.name,
                scope_context=scope_context
            )
            results.update(resolved)
        
        return results
    
    def _is_valid_field_fast(self, table: str, column: str) -> bool:
        """
        Validate field using pre-computed uppercase schema (Fix #4).
        
        ASSUMES table is already restored (@DBLINK syntax).
        ASSUMES column is already uppercase.
        
        Args:
            table: Table name (already restored)
            column: Column name (already uppercase)
            
        Returns:
            True if field exists in schema
        """
        if not self._schema_upper:
            return True
        
        # Check if table exists
        if table not in self._schema_upper:
            return False
        
        # Check if column exists (already uppercase)
        return column in self._schema_upper[table]
    
    def _resolve_unqualified_column(
        self, 
        column: str,  # Already uppercase
        available_tables: Set[str],  # Already restored
        max_candidates: int = 2,
        debug: bool = False
    ) -> List[str]:
        """
        Resolve unqualified column to table(s) using 3-tier policy.
        
        Tier 1 - Unique owner: 1 candidate → return it
        Tier 2 - Small ambiguity: 2-K candidates → return ALL
        Tier 3 - Large ambiguity: >K candidates → return NONE (skip)
        
        Args:
            column: Column name (uppercase)
            available_tables: Tables in this query (restored @DBLINK)
            max_candidates: K value for tier 2 (default: 2)
            debug: Whether to log skipped columns
            
        Returns:
            List of resolved table names (validated against schema)
        """
        if not self._schema_upper:
            return []
        
        # Find which available tables have this column
        candidates = []
        for table in available_tables:
            if column in self._schema_upper.get(table, set()):
                candidates.append(table)
        
        # Sort for deterministic ordering
        candidates = sorted(candidates)
        
        # Apply 3-tier policy
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            # Tier 1: Unique owner
            return candidates
        elif len(candidates) <= max_candidates:
            # Tier 2: Small ambiguity
            return candidates
        else:
            # Tier 3: Large ambiguity - skip
            if debug:
                print(f"[DEBUG] Skipped unqualified column '{column}' - too many candidates ({len(candidates)})")
            return []
    
    # ========== UTILITY METHODS ==========
    
    def _extract_literals(self, parsed: exp.Expression) -> Set[str]:
        """
        Extract literal values from ALL WHERE clauses.
        
        Returns:
            Set of unique literals
        """
        literals = set()
        
        for where in parsed.find_all(exp.Where):
            for lit in where.find_all(exp.Literal):
                value = lit.this
                
                if value and value.upper() not in ('NULL', 'TRUE', 'FALSE'):
                    value = value.strip("'\"")
                    if value:
                        literals.add(value)
        
        return literals
    
    def extract_tables(self, sql: str) -> Set[str]:
        """
        Extract just table names (for quick checks).
        
        Args:
            sql: SQL query
            
        Returns:
            Set of table names (with @DBLINK syntax preserved)
        """
        try:
            processed_sql = self._preprocess_oracle_dblinks(sql)
            parsed = sqlglot.parse_one(processed_sql, read=self.dialect)
            tables = set()
            
            for table_node in parsed.find_all(exp.Table):
                if table_node.name:
                    original_name = self._dblink_map.get(table_node.name, table_node.name)
                    tables.add(original_name)
            
            return tables
        except:
            return set()
    
    def is_valid_sql(self, sql: str) -> bool:
        """
        Check if SQL is syntactically valid.
        
        Args:
            sql: SQL query
            
        Returns:
            True if valid
        """
        try:
            processed_sql = self._preprocess_oracle_dblinks(sql)
            sqlglot.parse_one(processed_sql, read=self.dialect)
            return True
        except:
            return False


def extract_fields_and_literals(
    sql: str, 
    dialect: str = settings.db_type,
    schema: Optional[Dict[str, Set[str]]] = None
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """
    Convenience function to extract fields and literals.
    
    Args:
        sql: SQL query
        dialect: SQL dialect
        schema: Optional schema {table: {columns}}
        
    Returns:
        (fields, literals) tuple
    """
    parser = SQLParser(dialect=dialect, schema=schema)
    result = parser.parse(sql)
    return result.referenced_fields, result.literals