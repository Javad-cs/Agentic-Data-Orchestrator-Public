import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test the column profiler."""

from test_database import create_sample_db
from core import connect_database
from profiling import ColumnProfiler


def test_profiler():
    """Test column profiling."""
    print(" Testing Column Profiler...\n")
    
    # Create sample database
    db_path = create_sample_db()
    
    # Connect and profile
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler(top_k=5)
        
        # Get all tables
        tables = db.get_tables()
        
        for table_name in tables:
            print(f" Profiling table: {table_name}")
            table_info = db.get_table_info(table_name)
            
            for col in table_info.columns:
                profile = profiler.profile_column(
                    db, table_name, col.name, col.type
                )
                
                print(f"\n  Column: {profile.column_name} ({profile.data_type})")
                print(f"    Total records: {profile.total_records}")
                print(f"    NULL: {profile.null_count}, Non-NULL: {profile.non_null_count}")
                print(f"    Distinct values: {profile.distinct_count}")
                
                if profile.min_value is not None:
                    print(f"    Range: {profile.min_value} to {profile.max_value}")
                
                if profile.min_length is not None:
                    print(f"    Length: {profile.min_length} to {profile.max_length} chars")
                
                if profile.common_pattern:
                    print(f"    Pattern: {profile.common_pattern}")
                
                if profile.top_k_values:
                    print(f"    Top values: {profile.top_k_values[:3]}")
                
                print(f"    Numeric: {profile.is_numeric}, Date-like: {profile.is_date_like}")
            
            print()
    
    # Cleanup
    db_path.unlink()
    print(" Profiler tests passed!\n")


if __name__ == "__main__":
    test_profiler()