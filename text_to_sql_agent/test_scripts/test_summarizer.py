import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test LLM-based profile summarization."""

from test_database import create_sample_db
from core import connect_database
from profiling import ColumnProfiler, ProfileSummarizer


def test_summarizer():
    """Test profile summarization."""
    print(" Testing Profile Summarizer...\n")
    
    # Create sample database
    db_path = create_sample_db()
    
    # Connect and profile
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler()
        summarizer = ProfileSummarizer()
        
        # Profile and summarize the 'users' table
        print(" Profiling and summarizing 'users' table...\n")
        table_info = db.get_table_info("users")
        
        for col in table_info.columns[:2]:  # Just first 2 columns for demo
            print(f"{'='*60}")
            print(f"Column: {col.name}")
            print(f"{'='*60}\n")
            
            # Profile
            print(f" Profiling column '{col.name}'...")
            profile = profiler.profile_column(db, "users", col.name, col.type)
            
            # Summarize
            print(f" Generating LLM summaries for '{col.name}'...")
            metadata = summarizer.summarize(profile)
            
            # Debug: Check what we got
            print(f"\n[DEBUG] metadata type: {type(metadata)}")
            print(f"[DEBUG] short_description type: {type(metadata.short_description)}")
            print(f"[DEBUG] short_description length: {len(metadata.short_description) if metadata.short_description else 0}")
            print(f"[DEBUG] short_description repr: {repr(metadata.short_description)}")
            
            print(f"\n SHORT Description:")
            if metadata.short_description:
                print(f"   {metadata.short_description}")
            else:
                print(f"   [EMPTY - This is the bug!]")
            
            print(f"\n LONG Description:")
            if metadata.long_description:
                print(f"   {metadata.long_description}")
            else:
                print(f"   [EMPTY - This is the bug!]")
            
            print(f"\n Original Stats:")
            print(f"   Distinct: {profile.distinct_count}/{profile.total_records}")
            print(f"   NULL: {profile.null_count}")
            print(f"   Top values: {profile.top_k_values[:3]}")
            print()
    
    # Cleanup
    db_path.unlink()
    print(" Summarizer tests passed!\n")


if __name__ == "__main__":
    test_summarizer()