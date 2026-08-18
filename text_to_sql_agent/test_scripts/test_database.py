import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test database connection with a sample SQLite database."""

import sqlite3
from pathlib import Path
from core import connect_database


def create_sample_db():
    """Create a simple test database."""
    db_path = Path("test_sample.db")
    
    # Remove if exists
    if db_path.exists():
        db_path.unlink()
    
    # Create database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT NOT NULL,
            price REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Insert sample data
    cursor.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 30)")
    cursor.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com', 25)")
    cursor.execute("INSERT INTO users VALUES (3, 'Charlie', NULL, 35)")
    
    cursor.execute("INSERT INTO orders VALUES (1, 1, 'Laptop', 999.99)")
    cursor.execute("INSERT INTO orders VALUES (2, 1, 'Mouse', 29.99)")
    cursor.execute("INSERT INTO orders VALUES (3, 2, 'Keyboard', 79.99)")
    
    conn.commit()
    conn.close()
    
    print(f" Created sample database: {db_path}")
    return db_path


def test_database():
    """Test database connection and queries."""
    print(" Testing database connection...\n")
    
    # Create sample database
    db_path = create_sample_db()
    
    # Connect
    with connect_database(str(db_path)) as db:
        print(f" Connected: {db}\n")
        
        # Get tables
        tables = db.get_tables()
        print(f" Tables: {tables}\n")
        
        # Get table info
        for table_name in tables:
            info = db.get_table_info(table_name)
            print(f" Table: {info.name}")
            print(f"   Rows: {info.row_count}")
            print(f"   Columns:")
            for col in info.columns:
                fk = f" -> {col.foreign_key}" if col.foreign_key else ""
                pk = " [PK]" if col.primary_key else ""
                nullable = " NULL" if col.nullable else " NOT NULL"
                print(f"      - {col.name} ({col.type}){pk}{nullable}{fk}")
            print()
        
        # Test query
        print(" Testing query...")
        results = db.execute_query("SELECT name, email FROM users WHERE age > 25")
        print(f"   Results: {results}\n")
    
    print(" All database tests passed!\n")
    
    # Cleanup
    db_path.unlink()
    print(" Cleaned up test database")


if __name__ == "__main__":
    test_database()