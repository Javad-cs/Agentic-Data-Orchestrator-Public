"""
End-to-end example of Phase 2: Schema Linking Pipeline.

This demonstrates the complete workflow:
1. Profile database columns
2. Generate LLM summaries  
3. Enrich with SME descriptions
4. Build FAISS index (semantic)
5. Build LSH index (literal)
6. Extract literals from question
7. Build focused schema (FAISS + LSH)
8. Generate 5 schema variants
"""

from pathlib import Path
from core import connect_database
from profiling import ColumnProfiler, ProfileSummarizer
from profiling.field_metadata import FieldMetadata
from profiling.metadata_enricher import MetadataEnricher
from indexing import FieldIndex, SchemaLiteralMatcher
from schema_linking import (
    FocusedSchemaBuilder,
    FocusedSchemaConfig,
    SchemaVariantGenerator,
    SchemaVariant,
    extract_literals
)
from config import settings


def run_phase2_example():
    """Run complete Phase 2 pipeline on superhero database."""
    
    print("=" * 80)
    print("Phase 2: Schema Linking Pipeline - Complete Superhero Database")
    print("=" * 80 + "\n")
    
    db_name = "superhero"
    db_path = settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
    
    if not db_path.exists():
        print(f" Database not found: {db_path}")
        return
    
    # Variables for summary
    all_tables = []
    by_table = {}
    
    # Step 1: Profile database
    print("Step 1: Profiling ALL tables in superhero database...")
    
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler()
        
        # Get ALL tables
        all_tables = db.get_tables()
        print(f"  Found {len(all_tables)} tables: {', '.join(all_tables)}")
        
        profiles = []
        for table in all_tables:
            table_info = db.get_table_info(table)
            print(f"  Profiling table: {table} ({len(table_info.columns)} columns)")
            
            for col in table_info.columns:
                profile = profiler.profile_column(db, table, col.name, col.type)
                profiles.append(profile)
        
        print(f"   Profiled {len(profiles)} columns from {len(all_tables)} tables\n")
    
    # Step 2: Generate LLM summaries
    print("Step 2: Generating LLM summaries...")
    print("  Note: This makes 1 LLM call per column (can be slow).")
    print("  For production, consider: async batching or caching.")
    
    summarizer = ProfileSummarizer()
    metadata_list = []
    
    # Generate summaries for ALL profiled fields
    for i, profile in enumerate(profiles):
        print(f"  Summarizing field {i+1}/{len(profiles)}: {profile.table_name}.{profile.column_name}")
        metadata = summarizer.summarize(profile)
        metadata_list.append(metadata)
    
    print(f"   Generated summaries for {len(metadata_list)} fields\n")
    
    # Step 3: Enrich with SME descriptions
    print("Step 3: Enriching with SME descriptions...")
    enricher = MetadataEnricher(settings.bird_root_path)
    metadata_list = enricher.enrich_batch(db_name, metadata_list)
    
    sme_count = sum(1 for m in metadata_list if m.sme_description)
    print(f"   Added SME descriptions to {sme_count}/{len(metadata_list)} fields\n")
    
    # Step 4: Build FAISS index
    print("Step 4: Building FAISS semantic index...")
    field_index = FieldIndex()
    indexed_faiss = field_index.build_from_metadata(
        metadata_list,
        use_full_description=True,
        show_progress=False
    )
    print(f"   Indexed {indexed_faiss} fields in FAISS\n")
    
    # Step 5: Build LSH index
    print("Step 5: Building LSH literal index...")
    # Note: SchemaLiteralMatcher filters columns internally (skips constants by default)
    literal_matcher = SchemaLiteralMatcher(
        threshold=0.3,
        skip_constants=True,   # Skip columns with 1 distinct value (paper-aligned)
        skip_likely_ids=False  # Include IDs (users query by ID!)
    )
    
    indexed_lsh_count = 0
    for metadata in metadata_list:
        count = literal_matcher.index_column_from_profile(metadata.profile)
        indexed_lsh_count += count
    
    print(f"   Indexed {indexed_lsh_count} values from {len(literal_matcher.indexed_columns)} columns in LSH\n")
    
    # Step 6-7: Test BIRD questions
    questions = [
        ("Who is the dumbest superhero?", "superhero, hero_attribute, attribute"),
        ("What is the hero's full name with the highest attribute in strength?", "superhero, hero_attribute, attribute"),
        ("Who is the tallest superhero?", "superhero")
    ]

    config = FocusedSchemaConfig(
        faiss_threshold=0.2,
        lsh_threshold=0.3,
        merge_strategy="union"
    )

    builder = FocusedSchemaBuilder(
        field_index=field_index,
        literal_matcher=literal_matcher,
        config=config
    )

    for i, (question, expected_tables) in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print(f"Expected tables: {expected_tables}")
        print('='*80)
        
        focused_fields = builder.build(question)
        stats = builder.get_statistics(focused_fields)
        
        print(f" Found {stats['total_fields']} fields from {stats['unique_tables']} tables")
        print(f"   FAISS: {stats['faiss_only']}, LSH: {stats['lsh_only']}, Both: {stats['both']}")
        
        for field in focused_fields[:10]:
            f_score = f"{field.faiss_score:.2f}" if field.faiss_score else "N/A"
            l_score = f"{field.lsh_score:.2f}" if field.lsh_score else "N/A"
            print(f"  - {field.table}.{field.column} (FAISS={f_score}, LSH={l_score})")

    print("\n" + "=" * 80)
    print(" Phase 2 Complete - Tested on 3 BIRD Questions!")
    print("=" * 80)

#     # Step 8: Generate schema variants
#     # Summary
#     print("\n" + "=" * 80)
#     print(" Phase 2 Pipeline Complete on Full Superhero Database!")
#     print("=" * 80)
#     print(f"""
# Summary:
# - Database: superhero ({len(all_tables)} tables)
# - Profiled: {len(profiles)} columns from ALL tables
# - LLM summaries: {len(metadata_list)} fields
# - SME enriched: {sme_count}/{len(metadata_list)} fields
# - FAISS indexed: {indexed_faiss} fields
# - LSH indexed: {indexed_lsh_count} values from {len(literal_matcher.indexed_columns)} columns
# - Focused schema: {len(focused_fields)} fields from {stats['unique_tables']} tables

# Tables in focused schema: {', '.join(sorted(by_table.keys()))}

# Ready for Phase 3: SQL Generation!
# """)


if __name__ == "__main__":
    try:
        run_phase2_example()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()