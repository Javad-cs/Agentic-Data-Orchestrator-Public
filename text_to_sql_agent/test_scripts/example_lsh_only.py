"""
Simplified Phase 2 demo - LSH only (no FAISS required).
Tests literal matching without semantic search.
"""

from pathlib import Path
from core import connect_database
from profiling import ColumnProfiler, ProfileSummarizer
from profiling.field_metadata import FieldMetadata
from profiling.metadata_enricher import MetadataEnricher
from indexing import SchemaLiteralMatcher
from schema_linking import extract_literals
from config import settings


def run_lsh_only_demo():
    """Run LSH-only demo (no FAISS needed)."""
    
    print("=" * 80)
    print("Phase 2 Demo: LSH Literal Matching (No FAISS)")
    print("=" * 80 + "\n")
    
    db_name = "superhero"
    db_path = settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
    
    if not db_path.exists():
        print(f" Database not found: {db_path}")
        return
    
    # Step 1: Profile database
    print("Step 1: Profiling superhero database...")
    
    tables_to_profile = ["alignment", "colour", "superhero", "publisher"]
    
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler()
        
        profiles = []
        for table in tables_to_profile:
            table_info = db.get_table_info(table)
            print(f"  Profiling table: {table} ({len(table_info.columns)} columns)")
            
            for col in table_info.columns:
                profile = profiler.profile_column(db, table, col.name, col.type)
                profiles.append(profile)
        
        print(f"   Profiled {len(profiles)} columns from {len(tables_to_profile)} tables\n")
    
    # Step 2: Enrich with SME descriptions
    print("Step 2: Enriching with SME descriptions...")
    enricher = MetadataEnricher(settings.bird_root_path)
    
    metadata_list = []
    for profile in profiles:
        metadata = FieldMetadata(profile=profile)
        metadata_list.append(metadata)
    
    metadata_list = enricher.enrich_batch(db_name, metadata_list)
    
    sme_count = sum(1 for m in metadata_list if m.sme_description)
    print(f"   Added SME descriptions to {sme_count}/{len(metadata_list)} fields\n")
    
    # Step 3: Build LSH index
    print("Step 3: Building LSH literal index...")
    literal_matcher = SchemaLiteralMatcher(
        threshold=0.3,
        skip_constants=True,
        skip_likely_ids=False
    )
    
    indexed_lsh_count = 0
    for metadata in metadata_list:
        count = literal_matcher.index_column_from_profile(metadata.profile)
        indexed_lsh_count += count
    
    print(f"   Indexed {indexed_lsh_count} values from {len(literal_matcher.indexed_columns)} columns\n")
    
    # Step 4: Test questions
    test_questions = [
        "Show superheroes with Good alignment",
        "Find characters with blue eyes",
        "List heroes from Marvel Comics",
    ]
    
    print("Step 4: Testing LSH literal matching...")
    print("-" * 80)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        # Extract literals
        literals = extract_literals(question)
        print(f"  Extracted literals: {literals}")
        
        # Find matching fields
        all_matches = {}
        for literal in literals:
            matches = literal_matcher.find_matching_fields(literal, top_k=5)
            
            if matches:
                print(f"  Matches for '{literal}':")
                for match in matches:
                    print(f"    - {match.table}.{match.column} (score: {match.score:.3f})")
                    key = (match.table, match.column)
                    if key not in all_matches or match.score > all_matches[key]:
                        all_matches[key] = match.score
        
        if not all_matches:
            print("  No LSH matches found")
        else:
            print(f"  Total unique fields matched: {len(all_matches)}")
    
    print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print(" LSH Demo Complete!")
    print("=" * 80)
    print(f"""
Summary:
- Profiled: {len(profiles)} columns from {len(tables_to_profile)} tables
- SME enriched: {sme_count}/{len(metadata_list)} fields
- LSH indexed: {indexed_lsh_count} values from {len(literal_matcher.indexed_columns)} columns
- Literal matching works! 

To enable FAISS semantic search:
1. Install: conda install -c conda-forge faiss-cpu
2. Run: python example_phase2_pipeline.py
""")


if __name__ == "__main__":
    try:
        run_lsh_only_demo()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()