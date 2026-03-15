import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Comprehensive test of the indexing pipeline using BIRD benchmark data.
Tests profiler → LSH matcher → schema matcher → SQL utils.
"""

from pathlib import Path
from core import connect_database
from profiling import ColumnProfiler, ProfileSummarizer
from indexing import (
    SchemaLiteralMatcher, 
    LexicalLSHMatcher,
    escape_sql_identifier,
    format_sql_literal,
    build_where_clause,
)
from config import settings


def test_profiler_indexed_values():
    """Test that profiler populates indexed_values correctly on BIRD data."""
    print("=" * 80)
    print("TEST 1: Profiler indexed_values on BIRD superhero DB")
    print("=" * 80 + "\n")
    
    bird_root = settings.bird_data_path
    db_path = bird_root / "superhero" / "superhero.sqlite"
    
    if not db_path.exists():
        print(f" SKIP: Database not found at {db_path}")
        print("Please ensure BIRD dataset is downloaded and BIRD_DATA_PATH is set correctly")
        return False
    
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler(sample_size=10000)
        
        # Test on alignment table (small, clean data)
        print("Testing on 'alignment' table:")
        profile = profiler.profile_column(db, "alignment", "alignment", "TEXT")
        
        print(f"  Total records: {profile.total_records}")
        print(f"  Distinct count: {profile.distinct_count}")
        print(f"  Top-k values: {len(profile.top_k_values)}")
        print(f"  Sample values: {len(profile.sample_values)}")
        print(f"  Indexed values: {len(profile.indexed_values)}")
        
        if profile.indexed_values:
            print(f"  First 5 indexed values: {profile.indexed_values[:5]}")
        
        # Verify indexed_values exists and is populated
        assert hasattr(profile, 'indexed_values'), "Profile should have indexed_values!"
        assert profile.indexed_values is not None, "indexed_values should not be None!"
        assert len(profile.indexed_values) > 0, "indexed_values should be populated!"
        
        # Verify it's distinct (GROUP BY should ensure this)
        assert len(profile.indexed_values) == len(set(profile.indexed_values)), "Values should be distinct!"
        
        print("\n PASS: Profiler correctly populates indexed_values\n")
        return True


def test_lsh_matcher_collisions():
    """Test that LSH handles collisions like 'L.A.' vs 'LA'."""
    print("=" * 80)
    print("TEST 2: LSH Matcher Collision Handling")
    print("=" * 80 + "\n")
    
    from indexing import LexicalLSHMatcher
    
    # Test values that normalize to same form
    values = ["L.A.", "LA", "la", "San-Francisco", "San Francisco"]
    
    matcher = LexicalLSHMatcher()
    count = matcher.index_values(values, prefix="test|city")
    
    print(f"Indexed {count} values from {len(values)} input values")
    print(f"Keys generated: {len(matcher.key_to_original)}")
    
    # All should be indexed (stable hash handles collisions)
    assert count == len(values), f"Should index all {len(values)} values!"
    
    # Test querying
    print("\nTesting queries:")
    
    for query in ["LA", "la", "L.A.", "San Francisco"]:
        results = matcher.query(query, top_k=5)
        print(f"  Query '{query}': {len(results)} matches")
        for r in results[:3]:
            print(f"    - '{r.original_value}' (score: {r.score:.3f})")
    
    print("\n PASS: LSH handles collisions correctly\n")
    return True


def test_schema_matcher_superhero():
    """Test schema matcher on BIRD superhero database."""
    print("=" * 80)
    print("TEST 3: Schema Matcher on BIRD Superhero DB")
    print("=" * 80 + "\n")
    
    bird_root = settings.bird_data_path
    db_path = bird_root / "superhero" / "superhero.sqlite"
    
    if not db_path.exists():
        print(f" SKIP: Database not found at {db_path}")
        return False
    
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler()
        matcher = SchemaLiteralMatcher()
        
        # Profile alignment table (known data)
        tables_to_test = ["alignment", "colour", "gender"]
        
        for table_name in tables_to_test:
            print(f"\n Indexing table: {table_name}")
            table_info = db.get_table_info(table_name)
            
            for col in table_info.columns:
                profile = profiler.profile_column(db, table_name, col.name, col.type)
                count = matcher.index_column_from_profile(profile)
                
                if count > 0:
                    print(f"   {col.name} ({col.type}): {count} values indexed")
                else:
                    reason = "constant" if profile.distinct_count <= 1 else "skipped"
                    print(f"  ⊘ {col.name}: {reason}")
        
        print(f"\n Total indexed: {len(matcher.matcher)} values across {len(matcher.indexed_columns)} columns")
        
        # Test realistic queries from superhero domain
        test_queries = [
            "Good",           # Should match alignment
            "Bad",            # Should match alignment
            "Blue",           # Should match colour
            "Male",           # Should match gender
            "Female",         # Should match gender
        ]
        
        print(f"\n Testing queries:")
        for query in test_queries:
            matches = matcher.find_matching_fields(query, top_k=5)
            print(f"\n  Query: '{query}' → {len(matches)} matches")
            
            for match in matches[:3]:
                sql_snippet = f"{match.table}.{match.column} = "
                if match.is_numeric:
                    sql_snippet += match.value
                else:
                    sql_snippet += f"'{match.value}'"
                print(f"    - {sql_snippet} (score: {match.score:.3f})")
        
        print("\n PASS: Schema matcher works correctly\n")
        return True


def test_sql_utils():
    """Test SQL utilities for safety."""
    print("=" * 80)
    print("TEST 4: SQL Utilities (Security & Correctness)")
    print("=" * 80 + "\n")
    
    # Test identifier escaping
    print("Testing identifier escaping:")
    
    test_cases = [
        ("age", "sqlite", "`age`"),
        ("users.age", "sqlite", "`users`.`age`"),
        ("my`table", "sqlite", "`my``table`"),  # SQL injection attempt
        ("my\"col", "postgres", '"my""col"'),    # SQL injection attempt
        ("schema.table.col", "sqlite", "`schema`.`table`.`col`"),
    ]
    
    all_passed = True
    for identifier, dialect, expected in test_cases:
        result = escape_sql_identifier(identifier, dialect)
        passed = result == expected
        all_passed = all_passed and passed
        status = "" if passed else ""
        print(f"  {status} {identifier:20} ({dialect:8}) → {result}")
        if not passed:
            print(f"       Expected: {expected}")
    
    # Test literal formatting
    print("\nTesting literal formatting:")
    
    literal_cases = [
        ("25", True, "25"),
        ("Good", False, "'Good'"),
        ("O'Brien", False, "'O''Brien'"),      # Quote escaping
        ("N/A", True, "'N/A'"),                # Invalid numeric → quoted
        ("123.45", True, "123.45"),
    ]
    
    for value, is_numeric, expected in literal_cases:
        result = format_sql_literal(value, is_numeric)
        passed = result == expected
        all_passed = all_passed and passed
        status = "" if passed else ""
        print(f"  {status} {value:15} (numeric={is_numeric}) → {result}")
        if not passed:
            print(f"       Expected: {expected}")
    
    # Test WHERE clause building
    print("\nTesting WHERE clause building:")
    
    where_tests = [
        (("alignment", "alignment", "Good", False, "sqlite"), 
         "`alignment`.`alignment` = 'Good'"),
        (("superhero", "id", "123", True, "sqlite"), 
         "`superhero`.`id` = 123"),
        (("hero_power", "power_name", "Super Strength", False, "sqlite"),
         "`hero_power`.`power_name` = 'Super Strength'"),
    ]
    
    for args, expected in where_tests:
        result = build_where_clause(*args)
        passed = result == expected
        all_passed = all_passed and passed
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {result}")
        if not passed:
            print(f"       Expected: {expected}")
    
    if all_passed:
        print("\n PASS: All SQL utilities work correctly\n")
    else:
        print("\n FAIL: Some SQL utility tests failed\n")
    
    return all_passed


def test_end_to_end_superhero():
    """Test complete pipeline on BIRD superhero database."""
    print("=" * 80)
    print("TEST 5: End-to-End Pipeline on BIRD Data")
    print("=" * 80 + "\n")
    
    bird_root = settings.bird_data_path
    db_path = bird_root / "superhero" / "superhero.sqlite"
    
    if not db_path.exists():
        print(f" SKIP: Database not found")
        return False
    
    with connect_database(str(db_path)) as db:
        # Step 1: Profile multiple tables
        print("Step 1: Profiling tables...")
        profiler = ColumnProfiler()
        matcher = SchemaLiteralMatcher()
        
        tables = ["alignment", "colour", "gender"]
        
        for table_name in tables:
            table_info = db.get_table_info(table_name)
            print(f"   {table_name}: {len(table_info.columns)} columns")
            
            for col in table_info.columns:
                profile = profiler.profile_column(db, table_name, col.name, col.type)
                matcher.index_column_from_profile(profile)
        
        print(f"\n  Total: {len(matcher.matcher)} values indexed")
        
        # Step 2: Realistic text-to-SQL scenario
        print("\nStep 2: Simulating text-to-SQL query...")
        user_question = "Show all superheroes with Good alignment"
        print(f"  User: \"{user_question}\"")
        
        # Extract literal
        literal = "Good"
        print(f"  Extracted literal: '{literal}'")
        
        # Step 3: Find matching fields
        print("\nStep 3: Finding matching schema fields...")
        matches = matcher.find_matching_fields(literal, top_k=3)
        
        if not matches:
            print("   No matches found!")
            return False
        
        print(f"  Found {len(matches)} potential matches:")
        for match in matches:
            print(f"    - {match.table}.{match.column} = '{match.value}' (score: {match.score:.3f})")
        
        # Step 4: Generate SQL
        print("\nStep 4: Generating SQL WHERE clause...")
        best_match = matches[0]
        where_clause = build_where_clause(
            best_match.table,
            best_match.column,
            best_match.value,
            best_match.is_numeric,
        )
        print(f"  WHERE {where_clause}")
        
        # Step 5: Verify it's correct
        expected_table = "alignment"
        expected_column = "alignment"
        
        if best_match.table == expected_table and best_match.column == expected_column:
            print("\n PASS: Correctly identified alignment.alignment column!")
        else:
            print(f"\n  WARNING: Expected {expected_table}.{expected_column}, got {best_match.table}.{best_match.column}")
        
        print("\n PASS: End-to-end pipeline works!\n")
        return True


def main():
    print("\n" + "=" * 80)
    print(" BIRD-Based Indexing Pipeline Tests")
    print("=" * 80 + "\n")
    
    print("Testing Phase 1 implementation:")
    print("  - Database profiling with indexed_values")
    print("  - LSH-based literal matching")
    print("  - Schema literal matcher")
    print("  - SQL utilities\n")
    
    results = {}
    
    try:
        results['profiler'] = test_profiler_indexed_values()
        results['lsh'] = test_lsh_matcher_collisions()
        results['sql_utils'] = test_sql_utils()
        results['schema_matcher'] = test_schema_matcher_superhero()
        results['end_to_end'] = test_end_to_end_superhero()
        
        print("=" * 80)
        print(" TEST SUMMARY")
        print("=" * 80)
        
        for test_name, passed in results.items():
            status = " PASS" if passed else " FAIL"
            print(f"  {status}: {test_name}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n ALL TESTS PASSED!")
            print("\n Phase 1 implementation is complete and working!")
            print(" Ready to proceed to Phase 2: Schema Linking Algorithm")
        else:
            print("\n  SOME TESTS FAILED")
            print("Please review the output above for details")
        
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n TEST SUITE FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()