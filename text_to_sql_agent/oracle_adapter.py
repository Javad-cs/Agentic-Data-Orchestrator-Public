"""
Oracle Database Adapter for Text-to-SQL Pipeline

This adapter implements the same interface as existing SQLite database
adapter.

Usage:
    from oracle_adapter import create_oracle_connection
    
    with create_oracle_connection() as db:
        tables = db.get_tables()
        results = db.execute_query("SELECT * FROM CIM_EQP_MST WHERE ROWNUM <= 5")
"""

import oracledb
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class ColumnInfo:
    """Column metadata"""
    name: str
    type: str
    length: Optional[int] = None
    nullable: bool = True
    primary_key: bool = False


@dataclass
class TableInfo:
    """Table metadata"""
    name: str
    columns: List[ColumnInfo]


class OracleDatabase:
    """
    Oracle database adapter compatible with BaseDatabase interface.
    
    Matches the interface of existing SQLite adapter so it can be
    used as a drop-in replacement in pipeline.
    """
    
    def __init__(self, user: str, password: str, dsn: str, encoding: str = "UTF-8"):
        self.user = user
        self.password = password
        self.dsn = dsn
        self.encoding = encoding
        self.connection = None
        self.cursor = None
        
        # Set encoding for Korean
        import os
        os.environ['NLS_LANG'] = '.UTF8'
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def connect(self):
        """Establish connection to Oracle database"""
        try:
            try:
                oracledb.init_oracle_client()
            except:
                pass
            
            self.connection = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            self.cursor = self.connection.cursor()
            print(f" Connected to Oracle database: {self.dsn}")
            
        except Exception as e:
            print(f" Failed to connect to Oracle: {e}")
            raise
    
    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database.
        
        Returns:
            List of table names
        """
        if not self.cursor:
            raise RuntimeError("Database not connected")
        
        self.cursor.execute("""
            SELECT table_name 
            FROM user_tables 
            ORDER BY table_name
        """)
        
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_table_info(self, table_name: str) -> TableInfo:
        """
        Get detailed information about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            TableInfo object with columns and metadata
        """
        if not self.cursor:
            raise RuntimeError("Database not connected")
        
        # Get column information
        self.cursor.execute(f"""
            SELECT 
                column_name,
                data_type,
                data_length,
                nullable
            FROM user_tab_columns
            WHERE table_name = :table_name
            ORDER BY column_id
        """, {"table_name": table_name.upper()})
        
        columns = []
        for row in self.cursor.fetchall():
            columns.append(ColumnInfo(
                name=row[0],
                type=row[1],
                length=row[2] if row[2] else None,
                nullable=(row[3] == 'Y')
            ))
        
        # Get primary key information
        self.cursor.execute(f"""
            SELECT column_name
            FROM user_cons_columns
            WHERE constraint_name = (
                SELECT constraint_name
                FROM user_constraints
                WHERE table_name = :table_name
                AND constraint_type = 'P'
            )
        """, {"table_name": table_name.upper()})
        
        pk_columns = {row[0] for row in self.cursor.fetchall()}
        
        # Mark primary key columns
        for col in columns:
            if col.name in pk_columns:
                col.primary_key = True
        
        return TableInfo(name=table_name, columns=columns)
    
    def execute_query(self, sql: str, params: Optional[Dict] = None) -> List[tuple]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query string
            params: Optional parameters for parameterized queries
            
        Returns:
            List of result tuples
        """
        if not self.cursor:
            raise RuntimeError("Database not connected")
        
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            
            # Fetch all results
            return self.cursor.fetchall()
            
        except Exception as e:
            print(f"Query execution error: {e}")
            print(f"SQL: {sql}")
            raise
    
    def get_sample_values(self, table_name: str, column_name: str, limit: int = 10) -> List[Any]:
        """
        Get sample values from a column.
        
        Args:
            table_name: Table name
            column_name: Column name
            limit: Number of samples to retrieve
            
        Returns:
            List of sample values
        """
        sql = f"""
            SELECT DISTINCT {column_name}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            AND ROWNUM <= :limit
        """
        
        results = self.execute_query(sql, {"limit": limit})
        return [row[0] for row in results]
    
    def get_row_count(self, table_name: str) -> int:
        """
        Get total number of rows in a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Row count
        """
        sql = f"SELECT COUNT(*) FROM {table_name}"
        result = self.execute_query(sql)
        return result[0][0] if result else 0
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        
        if self.connection:
            self.connection.close()
            self.connection = None
            print(" Oracle connection closed")

# Convenience function matching existing pattern
def connect_database(dsn: str, user: str, password: str):
    """
    Create Oracle database connection.
    
    This matches existing connect_database() function signature
    for easy integration.
    
    Args:
        dsn: Data Source Name or connection string
        user: Oracle username
        password: Oracle password
        
    Returns:
        OracleDatabase instance (use with context manager)
    """
    return OracleDatabase(user, password, dsn)


if __name__ == "__main__":
    """Quick test of Oracle adapter"""
    
    print("\n" + "="*60)
    print("ORACLE ADAPTER TEST")
    print("="*60 + "\n")
    
    try:
        with create_oracle_connection() as db:
            # Test 1: Get tables
            print(" Available tables:")
            tables = db.get_tables()
            for table in tables[:10]:  # Show first 10
                print(f"   - {table}")
            print(f"   ... ({len(tables)} total tables)")
            
            # Test 2: Get table info
            test_table = "CIM_EQP_MST"
            if test_table in tables:
                print(f"\n Table structure: {test_table}")
                info = db.get_table_info(test_table)
                for col in info.columns[:5]:  # Show first 5 columns
                    pk_marker = " [PK]" if col.primary_key else ""
                    print(f"   - {col.name} ({col.type}){pk_marker}")
                print(f"   ... ({len(info.columns)} total columns)")
                
                # Test 3: Row count
                count = db.get_row_count(test_table)
                print(f"   Row count: {count:,}")
                
                # Test 4: Sample query
                print(f"\n Sample query:")
                results = db.execute_query(f"SELECT * FROM {test_table} WHERE ROWNUM <= 3")
                print(f"   Retrieved {len(results)} rows")
                if results:
                    print(f"   First row: {results[0][:5]}...")
            
            print("\n All tests passed!")
            
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()