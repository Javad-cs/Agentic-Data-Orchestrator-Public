import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Unit tests for SQL parser.
Tests field and literal extraction from SQL queries.
"""

from sql_generation import SQLParser


def test_sql_parser():
    """Test SQL parsing capabilities."""
    
    print("=" * 80)
    print("SQL Parser Unit Tests")
    print("=" * 80 + "\n")
    
    parser = SQLParser(dialect="sqlite")
    
    # Test 1: Simple SELECT
    print("Test 1: Simple SELECT")
    sql1 = "SELECT superhero_name FROM superhero WHERE alignment = 'Good'"
    result1 = parser.parse(sql1)
    
    print(f"SQL: {sql1}")
    print(f"Valid: {result1.is_valid}")
    print(f"Fields: {result1.referenced_fields}")
    print(f"Literals: {result1.literals}")
    assert result1.is_valid
    assert ('superhero', 'superhero_name') in result1.referenced_fields
    assert 'Good' in result1.literals
    print(" Passed\n")
    
    # Test 2: JOIN query with aliases
    print("Test 2: JOIN with aliases - ALIAS RESOLUTION")
    sql2 = """
    SELECT T1.superhero_name 
    FROM superhero AS T1 
    INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id 
    WHERE T2.attribute_value = 100
    """
    result2 = parser.parse(sql2)
    
    print(f"SQL: {sql2.strip()}")
    print(f"Valid: {result2.is_valid}")
    if not result2.is_valid:
        print(f"Parse Error: {result2.parse_error}")
    print(f"Fields (should show REAL tables, not aliases): {result2.referenced_fields}")
    print(f"Literals: {result2.literals}")
    assert result2.is_valid, f"SQL parsing failed: {result2.parse_error}"
    assert '100' in result2.literals
    # Check alias resolution worked
    assert ('superhero', 'superhero_name') in result2.referenced_fields
    assert ('superhero', 'id') in result2.referenced_fields
    assert ('hero_attribute', 'hero_id') in result2.referenced_fields
    assert ('hero_attribute', 'attribute_value') in result2.referenced_fields
    # Make sure aliases NOT in results
    assert ('T1', 'superhero_name') not in result2.referenced_fields
    assert ('T2', 'hero_id') not in result2.referenced_fields
    print(" Passed - Aliases resolved correctly!\n")
    
    # Test 3: Complex WHERE
    print("Test 3: Multiple literals in WHERE")
    sql3 = """
    SELECT full_name 
    FROM superhero 
    WHERE height_cm > 180 
      AND publisher_id IN (SELECT id FROM publisher WHERE publisher_name = 'Marvel Comics')
    """
    result3 = parser.parse(sql3)
    
    print(f"SQL: {sql3.strip()}")
    print(f"Valid: {result3.is_valid}")
    print(f"Fields: {result3.referenced_fields}")
    print(f"Literals: {result3.literals}")
    assert result3.is_valid
    assert 'Marvel Comics' in result3.literals
    assert '180' in result3.literals
    print(" Passed\n")
    
    # Test 4: Invalid SQL
    print("Test 4: Invalid SQL")
    sql4 = "SELECT FROM WHERE"
    result4 = parser.parse(sql4)
    
    print(f"SQL: {sql4}")
    print(f"Valid: {result4.is_valid}")
    print(f"Error: {result4.parse_error[:100] if result4.parse_error else 'None'}")
    assert not result4.is_valid
    print(" Passed\n")
    
    # Test 5: No literals
    print("Test 5: No literals (aggregation)")
    sql5 = "SELECT MAX(height_cm) FROM superhero"
    result5 = parser.parse(sql5)
    
    print(f"SQL: {sql5}")
    print(f"Valid: {result5.is_valid}")
    print(f"Fields: {result5.referenced_fields}")
    print(f"Literals: {result5.literals}")
    assert result5.is_valid
    assert len(result5.literals) == 0
    print(" Passed\n")
    
    print("=" * 80)
    print(" All SQL Parser Tests Passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_sql_parser()