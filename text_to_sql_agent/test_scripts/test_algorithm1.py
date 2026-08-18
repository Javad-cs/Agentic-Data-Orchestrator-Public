import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Test Algorithm 1: Iterative Schema Refinement
Tests on real BIRD questions to verify field discovery.
"""

from pathlib import Path
from core import connect_database, create_llm_client
from profiling import ColumnProfiler, ProfileSummarizer
from profiling.field_metadata import FieldMetadata
from profiling.metadata_enricher import MetadataEnricher
from indexing import FieldIndex, SchemaLiteralMatcher
from schema_linking import (
    SchemaVariant,
    SchemaVariantGenerator,
    FocusedSchemaBuilder,
    FocusedSchemaConfig
)
from sql_generation import Algorithm1Runner
from config import settings


def test_algorithm1():
    """
    Test Algorithm 1 on BIRD superhero database.
    
    Tests:
    1. Question: "Who is the dumbest superhero?"
       Expected tables: superhero, hero_attribute, attribute
    
    2. Question: "What is the hero's full name with the highest attribute in strength?"
       Expected tables: superhero, hero_attribute, attribute
    
    3. Question: "Who is the tallest superhero?"
       Expected tables: superhero
    """
    
    print("=" * 80)
    print("Algorithm 1 Test: Field Discovery via Iterative Refinement")
    print("=" * 80 + "\n")
    
    db_name = "superhero"
    db_path = settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
    
    if not db_path.exists():
        print(f" Database not found: {db_path}")
        return
    
    # Step 1: Profile database (reuse from Phase 2)
    print("Step 1: Profiling database...")
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler()
        all_tables = db.get_tables()
        
        profiles = []
        for table in all_tables:
            table_info = db.get_table_info(table)
            for col in table_info.columns:
                profile = profiler.profile_column(db, table, col.name, col.type)
                profiles.append(profile)
        
        print(f"   Profiled {len(profiles)} columns from {len(all_tables)} tables\n")
    
    # Step 2: Generate LLM summaries (with caching)
    print("Step 2: Generating LLM summaries...")
    print("  (Using cache to avoid re-summarizing - first run will be slow)")
    
    summarizer = ProfileSummarizer(use_cache=True)  # Enable caching
    metadata_list = []
    
    cached_count = 0
    for i, profile in enumerate(profiles):
        # Check if cached (by trying to load)
        cache_key = summarizer._get_cache_key(profile)
        is_cached = summarizer._load_from_cache(cache_key) is not None
        
        status = "[CACHED]" if is_cached else "[NEW]"
        print(f"  {status} Summarizing {i+1}/{len(profiles)}: {profile.table_name}.{profile.column_name}")
        
        metadata = summarizer.summarize(profile)
        metadata_list.append(metadata)
        
        if is_cached:
            cached_count += 1
    
    print(f"   Generated summaries: {cached_count} from cache, {len(profiles) - cached_count} new\n")
    # Step 3: Enrich with SME
    print("Step 3: Enriching with SME descriptions...")
    enricher = MetadataEnricher(settings.bird_root_path)
    metadata_list = enricher.enrich_batch(db_name, metadata_list)
    sme_count = sum(1 for m in metadata_list if m.sme_description)
    print(f"   {sme_count}/{len(metadata_list)} fields enriched with SME descriptions\n")
    
    # Step 4: Build indexes
    print("Step 4: Building indexes...")
    
    # FAISS
    field_index = FieldIndex()
    field_index.build_from_metadata(metadata_list, use_full_description=True, show_progress=False)
    
    # LSH
    literal_matcher = SchemaLiteralMatcher(threshold=0.3, skip_constants=True)
    for metadata in metadata_list:
        literal_matcher.index_column_from_profile(metadata.profile)
    
    print(f"   Indexes built: FAISS + LSH\n")
    
    # Step 5: Build focused schemas
    print("Step 5: Building focused schemas...")
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
    
    # Step 6: Test questions
    test_questions = [
        ("Who is the dumbest superhero?", {"superhero", "hero_attribute", "attribute"}),
        ("What is the hero's full name with the highest attribute in strength?", {"superhero", "hero_attribute", "attribute"}),
        ("Who is the tallest superhero?", {"superhero"})
    ]
    
    # Create metadata map for schema generation
    metadata_map = {
        (m.profile.table_name, m.profile.column_name): m
        for m in metadata_list
    }
    
    variant_generator = SchemaVariantGenerator(metadata_map, debug=False)
    
    # Initialize Algorithm 1
    llm_client = create_llm_client()
    
    runner = Algorithm1Runner(
        llm_client=llm_client,
        literal_matcher=literal_matcher,
        metadata_map=metadata_map,
        max_literal_refinements=2,
        max_syntax_fixes=1
    )
    
    print("Step 6: Running Algorithm 1 on test questions...\n")
    
    for i, (question, expected_tables) in enumerate(test_questions, 1):
        print("=" * 80)
        print(f"Question {i}: {question}")
        print(f"Expected tables: {', '.join(sorted(expected_tables))}")
        print("=" * 80)
        
        # Build focused schema
        focused_fields = builder.build(question)
        print(f"\nFocused schema: {len(focused_fields)} fields")
        
        # Generate all 5 schema variants
        schema_representations = variant_generator.generate_all(
            focused_fields=focused_fields,
            include_scores=False
        )
        
        # Run Algorithm 1
        print(f"\nRunning Algorithm 1 (5 variants, max 2 refinements each)...")
        result = runner.run(question, schema_representations)
        
        # Analyze results
        print(f"\n{'─' * 80}")
        print("Algorithm 1 Results:")
        print(f"{'─' * 80}")
        print(f"Total SQL generated: {result.total_sql_generated}")
        print(f"Total iterations: {result.total_iterations}")
        print(f"Final fields discovered: {len(result.final_fields)}")
        print(f"Final literals found: {len(result.final_literals)}")
        
        # Show SQL generated (for debugging)
        print(f"\n{'─' * 80}")
        print("Generated SQL by Variant:")
        print(f"{'─' * 80}")
        for vr in result.variant_results:
            print(f"\n{vr.variant.value}:")
            for iter_idx, iteration in enumerate(vr.iterations):
                status = " VALID" if iteration.is_valid_sql else " INVALID"
                print(f"  Iteration {iter_idx} ({status}):")
                # Show SQL (truncated if too long)
                sql_preview = iteration.sql.replace('\n', ' ')[:200]
                if len(iteration.sql) > 200:
                    sql_preview += "..."
                print(f"    SQL: {sql_preview}")
                print(f"    Fields: {iteration.fields_used}")
                print(f"    Literals: {iteration.literals_used}")
                if iteration.missing_literals:
                    print(f"    Missing literals: {iteration.missing_literals}")
        
        # Show fields by table
        fields_by_table = {}
        for table, col in result.final_fields:
            if table not in fields_by_table:
                fields_by_table[table] = []
            fields_by_table[table].append(col)
        
        print(f"\nDiscovered tables: {', '.join(sorted(fields_by_table.keys()))}")
        
        # Check coverage
        discovered_tables = set(fields_by_table.keys())
        coverage = expected_tables & discovered_tables
        missing = expected_tables - discovered_tables
        extra = discovered_tables - expected_tables
        
        print(f"\n Covered: {', '.join(sorted(coverage)) if coverage else 'None'}")
        if missing:
            print(f" Missing: {', '.join(sorted(missing))}")
        if extra:
            print(f"○ Extra: {', '.join(sorted(extra))}")
        
        # Show per-variant breakdown
        print(f"\nPer-variant breakdown:")
        for vr in result.variant_results:
            variant_tables = set(t for t, c in vr.final_fields)
            print(f"  {vr.variant.value:20s}: {len(vr.iterations)} iterations, "
                  f"{len(vr.final_fields)} fields, "
                  f"tables: {', '.join(sorted(variant_tables))}")
            
            # Show convergence
            if vr.converged:
                print(f"                           Converged (no missing literals)")
            else:
                print(f"                          ○ Stopped at max refinements")
        
        print()
    
    print("=" * 80)
    print(" Algorithm 1 Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_algorithm1()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()