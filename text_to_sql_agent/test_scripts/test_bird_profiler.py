import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test the profiler on real BIRD databases."""

import sys
from pathlib import Path
from core import connect_database
from profiling import ColumnProfiler, ProfileSummarizer
from config import settings


def profile_bird_database(db_path: str, max_tables: int = 3):
    """
    Profile a BIRD database.
    
    Args:
        db_path: Path to .sqlite file
        max_tables: Maximum number of tables to profile (for speed)
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        print(f" Database not found: {db_path}")
        return
    
    print(f" Profiling BIRD Database: {db_path.name}")
    print(f" Size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("="*80)
    
    with connect_database(str(db_path)) as db:
        # Get all tables
        tables = db.get_tables()
        print(f"\n Found {len(tables)} tables: {', '.join(tables)}\n")
        
        # Profile first N tables
        profiler = ColumnProfiler(top_k=10)
        summarizer = ProfileSummarizer()
        
        for i, table_name in enumerate(tables[:max_tables]):
            print(f"\n{'='*80}")
            print(f"TABLE {i+1}/{min(len(tables), max_tables)}: {table_name}")
            print(f"{'='*80}\n")
            
            # Get table info
            table_info = db.get_table_info(table_name)
            print(f"  Rows: {table_info.row_count:,}")
            print(f"  Columns: {len(table_info.columns)}")
            
            # Profile first 3 columns (for demo)
            for j, col in enumerate(table_info.columns[:3]):
                print(f"\n  Column {j+1}: {col.name} ({col.type})")
                print(f"  {'-'*60}")
                
                # Profile
                profile = profiler.profile_column(
                    db, table_name, col.name, col.type
                )
                
                print(f"    Total: {profile.total_records:,} | "
                      f"NULL: {profile.null_count:,} | "
                      f"Distinct: {profile.distinct_count:,}")
                
                if profile.min_value is not None:
                    print(f"    Range: {profile.min_value} to {profile.max_value}")
                
                if profile.top_k_values:
                    top_3 = profile.top_k_values[:3]
                    print(f"    Top values: {top_3}")
                
                # Generate LLM summary (optional - comment out if too slow)
                print(f"\n     Generating description...")
                metadata = summarizer.summarize(profile)
                
                if metadata.short_description:
                    print(f"     SHORT: {metadata.short_description}")
                
                if metadata.long_description:
                    print(f"\n     LONG: {metadata.long_description}")
            
            if len(table_info.columns) > 3:
                print(f"\n  ... and {len(table_info.columns) - 3} more columns")
    
    print(f"\n{'='*80}")
    print(" Profiling complete!")


if __name__ == "__main__":
    # Default to superhero database (smallest)
    bird_root = Path(settings.bird_data_path)
    
    # Test databases in order of size
    test_dbs = [
        "superhero/superhero.sqlite",
        "toxicology/toxicology.sqlite", 
        "student_club/student_club.sqlite",
    ]
    
    # Allow command line argument
    if len(sys.argv) > 1:
        db_file = sys.argv[1]
    else:
        db_file = test_dbs[0]  # Default to superhero
    
    db_path = bird_root / db_file
    
    if not db_path.exists():
        print(f" Not found: {db_path}")
        print(f"\n Available databases:")
        for db in test_dbs:
            full_path = bird_root / db
            if full_path.exists():
                size = full_path.stat().st_size / 1024 / 1024
                print(f"   - {db:40} ({size:.1f} MB)")
    else:
        profile_bird_database(db_path, max_tables=10)