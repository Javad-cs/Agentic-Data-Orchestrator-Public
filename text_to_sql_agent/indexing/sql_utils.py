"""
SQL rendering utilities for safe literal and identifier formatting.
"""

from typing import Optional


def escape_sql_identifier(identifier: str, dialect: str = "sqlite") -> str:
    """
    Escape a SQL identifier (table/column name) for safe use in queries.
    
    Handles dotted names (table.column) by escaping each part separately.
    Escapes quote characters within identifiers to prevent SQL injection.
    
    Args:
        identifier: Table or column name, optionally dotted (e.g., "table.column")
        dialect: SQL dialect ("sqlite", "postgres", "mysql")
        
    Returns:
        Escaped identifier
        
    Examples:
        ("age", "sqlite") → "`age`"
        ("user", "sqlite") → "`user`"
        ("users.age", "sqlite") → "`users`.`age`"
        ("my`table", "sqlite") → "`my``table`"  (escaped backtick)
        ("my\"col", "postgres") → "\"my\"\"col\""  (escaped quote)
    """
    # Determine quote characters for dialect
    if dialect in ("sqlite", "mysql"):
        quote = "`"
    elif dialect == "postgres" or dialect == "oracle":
        quote = '"'
    else:
        # Generic: use double quotes
        quote = '"'
    
    # Split on dots and escape each part
    parts = identifier.split(".")
    escaped_parts = []
    
    for part in parts:
        if not part:  # Skip empty parts
            continue
        
        # Escape quote characters within the identifier
        if quote == "`":
            # Backtick: escape ` as ``
            escaped = part.replace("`", "``")
        else:
            # Double quote: escape " as ""
            escaped = part.replace('"', '""')
        
        escaped_parts.append(f"{quote}{escaped}{quote}")
    
    return ".".join(escaped_parts)


def format_sql_literal(
    value: str,
    is_numeric: bool,
    column_type: Optional[str] = None,
) -> str:
    """
    Format a value as a SQL literal with proper quoting.
    
    Args:
        value: String representation of value
        is_numeric: Whether the column is numeric
        column_type: SQL column type (for validation)
        
    Returns:
        SQL-safe literal
        
    Examples:
        ("25", True) → "25"
        ("District 01", False) → "'District 01'"
        ("O'Brien", False) → "'O''Brien'"
        ("N/A", True) → "'N/A'" (fallback for invalid numeric)
    """
    if is_numeric:
        # Try to validate as numeric
        cleaned = value.strip()
        try:
            # Check if it's actually a valid number
            float(cleaned)
            return cleaned
        except ValueError:
            # Not a valid number - fall back to quoting
            # This handles "N/A", "-", etc. in numeric columns
            escaped = cleaned.replace("'", "''")
            return f"'{escaped}'"
    else:
        # Text - single quote and escape
        escaped = value.replace("'", "''")
        return f"'{escaped}'"


def build_where_clause(
    table: str,
    column: str,
    value: str,
    is_numeric: bool,
    dialect: str = "sqlite",
) -> str:
    """
    Build a WHERE clause for a literal match.
    
    Args:
        table: Table name
        column: Column name
        value: Matched value
        is_numeric: Whether column is numeric
        dialect: SQL dialect
        
    Returns:
        WHERE clause string
        
    Examples:
        ("users", "age", "25", True, "sqlite") → "`users`.`age` = 25"
        ("schools", "district", "Fresno County", False, "sqlite") 
            → "`schools`.`district` = 'Fresno County'"
        ("my.table", "col", "val", False, "sqlite") 
            → "`my`.`table`.`col` = 'val'"  (handles dotted table name)
    """
    table_esc = escape_sql_identifier(table, dialect)
    column_esc = escape_sql_identifier(column, dialect)
    value_lit = format_sql_literal(value, is_numeric)
    
    return f"{table_esc}.{column_esc} = {value_lit}"