import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Test table retrieval for Slow Lane planner.

Tests the complete pipeline:
1. Column profiling → ColumnProfile
2. Column summarization → FieldMetadata
3. Table profiling → TableProfile
4. Table summarization → TableMetadata
5. Table retrieval → TableSummary
"""

import asyncio
from pathlib import Path

from core import connect_database
from profiling import (
    ColumnProfiler,
    ProfileSummarizer,
    TableProfiler,
    TableSummarizer
)
from schema_linking import (
    FocusedSchemaBuilder,
    FocusedSchemaConfig,
    TableRetriever
)
from indexing import FieldIndex, SchemaLiteralMatcher
from profiling.metadata_enricher import MetadataEnricher
from config import settings


def test_column_profiling():
    """Test 1: Column profiling works correctly."""
    print("\n" + "="*70)
    print("TEST 1: Column Profiling")
    print("="*70)
    
    db_name = "financial"
    db_path = settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
    
    if not db_path.exists():
        print(f" Database not found: {db_path}")
        print("   Please update db_name to a database you have")
        return None
    
    with connect_database(str(db_path)) as db:
        tables = db.get_tables()
        print(f"Found {len(tables)} tables: {tables[:3]}...")
        
        # Profile first table
        table_name = tables[0]
        table_info = db.get_table_info(table_name)
        
        profiler = ColumnProfiler()
        
        print(f"\nProfiling table: {table_name}")
        column_profiles = []
        
        for col in table_info.columns[:3]:  # First 3 columns
            profile = profiler.profile_column(
                db, 
                table_name, 
                col.name, 
                col.type
            )
            column_profiles.append(profile)
            
            print(f"   {col.name} ({col.type}): "
                  f"{profile.distinct_count} distinct, "
                  f"{profile.null_count} nulls")
        
        print(f"\n Column profiling works")
        return db_name, table_name, column_profiles


def test_column_summarization(db_name, table_name, column_profiles):
    """Test 2: Column summarization generates descriptions."""
    print("\n" + "="*70)
    print("TEST 2: Column Summarization")
    print("="*70)
    
    summarizer = ProfileSummarizer(use_cache=True)
    
    print(f"\nGenerating descriptions for {table_name} columns...")
    field_metadata_list = []
    
    for profile in column_profiles:
        print(f"\n  Column: {profile.column_name}")
        metadata = summarizer.summarize(profile)
        field_metadata_list.append(metadata)
        
        print(f"  Short: {metadata.short_description[:100]}...")
        if metadata.long_description:
            print(f"  Long: {metadata.long_description[:100]}...")
    
    print(f"\n Column summarization works")
    return field_metadata_list


def test_table_profiling(db_name, table_name, column_profiles):
    """Test 3: Table profiling aggregates column profiles."""
    print("\n" + "="*70)
    print("TEST 3: Table Profiling")
    print("="*70)
    
    db_path = settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
    
    with connect_database(str(db_path)) as db:
        table_profiler = TableProfiler()
        
        print(f"\nProfiling table: {table_name}")
        table_profile = table_profiler.profile_table(
            db,
            table_name,
            column_profiles
        )
        
        print(f"  Total rows: {table_profile.total_rows}")
        print(f"  Total columns: {table_profile.total_columns}")
        print(f"  PK candidates: {table_profile.primary_key_candidates}")
        print(f"  FK candidates: {table_profile.foreign_key_candidates}")
        print(f"  First column: {table_profile.first_column}")
        
        print(f"\n Table profiling works")
        return table_profile


def test_table_summarization(table_profile, field_metadata_list):
    """Test 4: Table summarization generates table description."""
    print("\n" + "="*70)
    print("TEST 4: Table Summarization")
    print("="*70)
    
    print(f"\n  Column descriptions being passed:")
    for fm in field_metadata_list:
        print(f"    - {fm.profile.column_name}: {fm.short_description[:80]}...")
    
    table_summarizer = TableSummarizer(use_cache=True)
    
    print(f"\nGenerating description for {table_profile.table_name}...")
    table_metadata = table_summarizer.summarize(
        table_profile,
        field_metadata_list
    )
    
    print(f"\n  Description: {table_metadata.description}")
    print(f"  Column summaries: {len(table_metadata.column_summaries)} columns")
    
    print(f"\n Table summarization works")
    return table_metadata


def test_table_retrieval_end_to_end():
    """Test 5: End-to-end table retrieval for planner."""
    print("\n" + "="*70)
    print("TEST 5: End-to-End Table Retrieval")
    print("="*70)
    
    db_name = "financial"
    db_path = settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
    
    if not db_path.exists():
        print(f" Database not found: {db_path}")
        return
    
    # Step 1: Profile all columns and tables
    print("\n[1/5] Profiling database...")
    
    with connect_database(str(db_path)) as db:
        col_profiler = ColumnProfiler()
        all_column_profiles = []
        
        for table in db.get_tables()[:3]:  # First 3 tables for speed
            table_info = db.get_table_info(table)
            for col in table_info.columns:
                profile = col_profiler.profile_column(db, table, col.name, col.type)
                all_column_profiles.append(profile)
        
        print(f"  Profiled {len(all_column_profiles)} columns")
    
    # Step 2: Summarize columns
    print("\n[2/5] Summarizing columns...")
    
    col_summarizer = ProfileSummarizer(use_cache=True)
    field_metadata_list = []
    
    for profile in all_column_profiles:
        metadata = col_summarizer.summarize(profile)
        field_metadata_list.append(metadata)
    
    print(f"  Summarized {len(field_metadata_list)} columns")
    
    # Step 3: Build indices (FAISS + LSH)
    print("\n[3/5] Building FAISS and LSH indices...")
    
    # Enrich metadata
    enricher = MetadataEnricher(settings.bird_root_path)
    field_metadata_list = enricher.enrich_batch(db_name, field_metadata_list)
    
    # Build FAISS index
    field_index = FieldIndex()
    field_index.build_from_metadata(
        field_metadata_list,
        use_full_description=True,
        show_progress=False
    )
    
    # Build LSH index
    literal_matcher = SchemaLiteralMatcher(threshold=0.3)
    for metadata in field_metadata_list:
        literal_matcher.index_column_from_profile(metadata.profile)
    
    print(f"  FAISS index: {len(field_index)} fields")
    print(f"  LSH index: {len(literal_matcher.matcher)} values")
    
    # Step 4: Profile and summarize tables
    print("\n[4/5] Profiling and summarizing tables...")
    
    table_metadata_map = {}
    table_profiler = TableProfiler()
    table_summarizer = TableSummarizer(use_cache=True)
    
    # Group field metadata by table
    from collections import defaultdict
    fields_by_table = defaultdict(list)
    for fm in field_metadata_list:
        fields_by_table[fm.profile.table_name].append(fm)
    
    with connect_database(str(db_path)) as db:
        for table_name, field_metas in fields_by_table.items():
            # Get column profiles for this table
            col_profiles = [fm.profile for fm in field_metas]
            
            # Profile table
            table_profile = table_profiler.profile_table(db, table_name, col_profiles)
            
            # Summarize table
            table_metadata = table_summarizer.summarize(table_profile, field_metas)
            
            table_metadata_map[table_name] = table_metadata
    
    print(f"  Created metadata for {len(table_metadata_map)} tables")
    
    # Step 5: Retrieve relevant tables for query
    print("\n[5/5] Retrieving relevant tables for query...")
    
    test_queries = [
        "How many accounts do we have?",  # Should find 'account' table
        "Show me client information",     # Should find 'client' table
        "What types of cards are available?"  # Should find 'card' table
    ]
    
    config = FocusedSchemaConfig(
        faiss_threshold=0.2,
        lsh_threshold=0.3
    )
    
    builder = FocusedSchemaBuilder(
        field_index=field_index,
        literal_matcher=literal_matcher,
        config=config
    )
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        
        table_summaries = builder.get_relevant_tables(
            question=query,
            table_metadata_map=table_metadata_map,
            max_tables=3
        )
        
        print(f"  Found {len(table_summaries)} relevant tables:")
        
        for i, summary in enumerate(table_summaries, 1):
            match_info = f"FAISS={summary.faiss_score:.2f}" if summary.faiss_score else "LSH"
            print(f"\n    {i}. {summary.table_name} ({match_info})")
            print(f"       {summary.description}")
            print(f"       Columns: {len(summary.columns)} total")
            
            # Show matched columns
            matched = [c.name for c in summary.columns if c.is_lsh_matched]
            if matched:
                print(f"       LSH matched: {', '.join(matched)}")
            
            # Show key columns
            keys = [c.name for c in summary.columns if c.is_pk_candidate or c.is_fk_candidate]
            if keys:
                print(f"       Keys: {', '.join(keys)}")
    
    print(f"\n End-to-end retrieval works")
    
    # Test formatting for planner
    print("\n" + "="*70)
    print("FORMATTED OUTPUT FOR PLANNER:")
    print("="*70)
    
    query = test_queries[0]
    table_summaries = builder.get_relevant_tables(
        question=query,
        table_metadata_map=table_metadata_map,
        max_tables=2
    )
    
    print(f"\nQuery: '{query}'\n")
    print("RELEVANT TABLES:\n")
    
    for summary in table_summaries:
        print(summary.format_for_planner())
        print()


def test_large_table_safety_valve():
    """Test 6: Safety valve for tables with > 50 columns."""
    print("\n" + "="*70)
    print("TEST 6: Large Table Safety Valve")
    print("="*70)
    
    print("\nThis test requires a table with > 50 columns.")
    print("Skipping for now (will implement when we have such a table)")
    print("\n️  Test skipped")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TABLE RETRIEVAL TEST SUITE")
    print("="*70)
    
    # Test 1-4: Individual components
    result = test_column_profiling()
    
    if result:
        db_name, table_name, column_profiles = result
        
        field_metadata_list = test_column_summarization(
            db_name, table_name, column_profiles
        )
        
        table_profile = test_table_profiling(
            db_name, table_name, column_profiles
        )
        
        table_metadata = test_table_summarization(
            table_profile, field_metadata_list
        )
    
    # Test 5: End-to-end
    test_table_retrieval_end_to_end()
    
    # Test 6: Safety valve
    test_large_table_safety_valve()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()