import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Test script to verify SME parser can load BIRD descriptions correctly.
Run this to check if paths are configured properly.
"""

from pathlib import Path
from schema_linking.sme_parser import SMEParser
from config import settings


def test_sme_parser_paths():
    """Test that SME parser can find and load BIRD files."""
    
    print("=" * 80)
    print("Testing SME Parser Path Configuration")
    print("=" * 80 + "\n")
    
    bird_path = Path(settings.bird_root_path)
    print(f"BIRD_ROOT_PATH from .env: {bird_path}")
    print(f"Absolute path: {bird_path.resolve()}")
    print(f"Exists: {bird_path.exists()}\n")
    
    if not bird_path.exists():
        print(" BIRD_DATA_PATH does not exist!")
        print("\nExpected structure:")
        print("  dev_20240627/")
        print("  ├── dev.json")
        print("  ├── dev_tables.json")
        print("  └── dev_databases/")
        print("      └── superhero/")
        print("\nCurrent .env setting points to:")
        print(f"  {bird_path}")
        print("\nSuggested fix:")
        print("  BIRD_DATA_PATH=../bird_dataset/dev_20240627")
        return False
    
    # Check for dev_tables.json
    dev_tables_path = bird_path / "dev_tables.json"
    print(f"Checking for dev_tables.json...")
    print(f"  Path: {dev_tables_path}")
    print(f"  Exists: {dev_tables_path.exists()}")
    
    if not dev_tables_path.exists():
        print("\n dev_tables.json not found!")
        print("\nYour BIRD_DATA_PATH might be pointing to dev_databases/ instead of dev_20240627/")
        print("\nCurrent structure found:")
        if bird_path.exists():
            print(f"  Contents of {bird_path}:")
            for item in bird_path.iterdir():
                print(f"    - {item.name}")
        
        print("\nExpected to find: dev_tables.json")
        print("\nSuggested fix in .env:")
        print("  # OLD (wrong):")
        print("  # BIRD_DATA_PATH=../bird_dataset/dev_20240627/dev_databases")
        print("  ")
        print("  # NEW (correct):")
        print("  BIRD_DATA_PATH=../bird_dataset/dev_20240627")
        return False
    
    print("  Found!\n")
    
    # Check for dev_databases/superhero
    superhero_path = bird_path / "dev_databases" / "superhero"
    print(f"Checking for dev_databases/superhero...")
    print(f"  Path: {superhero_path}")
    print(f"  Exists: {superhero_path.exists()}")
    
    if not superhero_path.exists():
        print("\n superhero database not found!")
        return False
    
    print("  Found!\n")
    
    # Check for CSV descriptions
    csv_dir = superhero_path / "database_description"
    print(f"Checking for database_description/...")
    print(f"  Path: {csv_dir}")
    print(f"  Exists: {csv_dir.exists()}")
    
    if csv_dir.exists():
        csv_files = list(csv_dir.glob("*.csv"))
        print(f"   Found {len(csv_files)} CSV files:")
        for csv_file in csv_files[:5]:
            print(f"    - {csv_file.name}")
        if len(csv_files) > 5:
            print(f"    ... and {len(csv_files) - 5} more")
    else:
        print("    No database_description folder (will use dev_tables.json)")
    
    print("\n" + "=" * 80)
    print("Testing SME Parser Loading")
    print("=" * 80 + "\n")
    
    # Test loading
    parser = SMEParser(bird_path)
    
    print("Loading superhero database descriptions...")
    descriptions = parser.load_database_descriptions("superhero")
    
    print(f" Loaded {len(descriptions)} field descriptions\n")
    
    if descriptions:
        print("Sample descriptions:")
        for (table, column), desc in list(descriptions.items())[:3]:
            print(f"\n  {table}.{column} (source: {desc.source}):")
            preview = desc.description[:100] + "..." if len(desc.description) > 100 else desc.description
            print(f"    {preview}")
    
    print("\n" + "=" * 80)
    print(" SME Parser paths are configured correctly!")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        test_sme_parser_paths()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()