"""
Database connection and query utilities.
Abstract interface for database operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[ColumnInfo]
    row_count: int


class BaseDatabase(ABC):
    """Abstract base class for database connections."""
    
    @abstractmethod
    def get_tables(self) -> List[str]:
        """Get list of all table names."""
        pass
    
    @abstractmethod
    def get_table_info(self, table_name: str) -> TableInfo:
        """Get detailed information about a table."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        pass
    
    @abstractmethod
    def close(self):
        """Close database connection."""
        pass


class SQLiteDatabase(BaseDatabase):
    """SQLite database implementation."""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    
    def get_tables(self) -> List[str]:
        """Get list of all table names."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_info(self, table_name: str) -> TableInfo:
        """Get detailed information about a table."""
        cursor = self.conn.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = []
        for row in cursor.fetchall():
            col = ColumnInfo(
                name=row[1],
                type=row[2],
                nullable=not row[3],
                primary_key=bool(row[5]),
            )
            columns.append(col)
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        fk_map = {}
        for row in cursor.fetchall():
            from_col = row[3]
            to_table = row[2]
            to_col = row[4]
            fk_map[from_col] = f"{to_table}.{to_col}"
        
        # Add foreign key info to columns
        for col in columns:
            if col.name in fk_map:
                col.foreign_key = fk_map[col.name]
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
        row_count = cursor.fetchone()[0]
        
        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=row_count,
        )
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as list of dicts."""
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        # Convert Row objects to dictionaries
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        
        return results
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __repr__(self) -> str:
        return f"SQLiteDatabase(path={self.db_path.name})"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Factory function
def connect_database(db_path: str, db_type: str = "sqlite") -> BaseDatabase:
    """
    Connect to a database.
    
    Args:
        db_path: Path to database file
        db_type: Type of database ("sqlite", could add "postgres", "mysql" later)
        
    Returns:
        Database connection instance
    """
    if db_type == "sqlite":
        return SQLiteDatabase(db_path)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")