"""
Parse SME (Subject Matter Expert) descriptions from BIRD dataset.
Sources: database_description/*.csv and dev_tables.json
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import csv
import json
from dataclasses import dataclass


@dataclass
class SMEFieldDescription:
    """SME description for a database field."""
    table: str
    column: str
    description: str
    data_format: Optional[str] = None
    value_description: Optional[str] = None
    source: str = "sme"  # "csv" or "json"
    
    def __repr__(self):
        return f"SMEFieldDescription({self.table}.{self.column}, source={self.source})"


class SMEParser:
    """Parse SME descriptions from BIRD dataset."""
    
    def __init__(self, bird_root_path: Path):
        """
        Initialize SME parser.
        
        Args:
            bird_root_path: Path to BIRD dataset root 
                           (e.g., /path/to/dev_20240627)
                           Should contain dev_tables.json and dev_databases/
        """
        self.bird_root_path = Path(bird_root_path)
        self.dev_tables_json_path = self.bird_root_path / "dev_tables.json"
        self.dev_tables_cache = None
    
    def load_database_descriptions(
        self, 
        db_name: str
    ) -> Dict[Tuple[str, str], SMEFieldDescription]:
        """
        Load all SME descriptions for a database.
        
        Priority:
        1. database_description/*.csv files (most detailed)
        2. dev_tables.json (fallback)
        
        Args:
            db_name: Database name (e.g., "superhero")
            
        Returns:
            Dict mapping (table, column) to SMEFieldDescription
        """
        descriptions = {}
        
        # Try CSV files first (most detailed, has value_description)
        csv_dir = self.bird_root_path / "dev_databases" / db_name / "database_description"
        if csv_dir.exists():
            csv_descriptions = self._load_from_csv(csv_dir)
            descriptions.update(csv_descriptions)
        
        # Add from dev_tables.json for any missing fields
        json_descriptions = self._load_from_json(db_name)
        for key, desc in json_descriptions.items():
            if key not in descriptions:
                descriptions[key] = desc
        
        return descriptions
    
    def _load_from_csv(
        self, 
        csv_dir: Path
    ) -> Dict[Tuple[str, str], SMEFieldDescription]:
        """
        Load descriptions from database_description/*.csv files.
        
        CSV format:
        - original_column_name
        - column_name
        - column_description
        - data_format
        - value_description (detailed explanation with examples)
        """
        descriptions = {}
        
        for csv_file in csv_dir.glob("*.csv"):
            table_name = csv_file.stem  # filename without .csv
            
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # Get column name
                        column_name = row.get('column_name', '').strip()
                        if not column_name:
                            column_name = row.get('original_column_name', '').strip()
                        
                        if not column_name:
                            continue
                        
                        # Combine column_description and value_description
                        col_desc = row.get('column_description', '').strip()
                        val_desc = row.get('value_description', '').strip()
                        
                        # Create full description
                        # value_description is GOLD - has examples and commonsense reasoning
                        full_desc = col_desc
                        if val_desc:
                            full_desc = f"{col_desc}. {val_desc}" if col_desc else val_desc
                        
                        if full_desc:
                            descriptions[(table_name, column_name)] = SMEFieldDescription(
                                table=table_name,
                                column=column_name,
                                description=full_desc,
                                data_format=row.get('data_format'),
                                value_description=val_desc,
                                source="csv"
                            )
            except Exception as e:
                print(f"Warning: Failed to parse {csv_file}: {e}")
        
        return descriptions
    
    def _load_from_json(
        self, 
        db_name: str
    ) -> Dict[Tuple[str, str], SMEFieldDescription]:
        """
        Load descriptions from dev_tables.json.
        
        This is a fallback for fields not covered in CSV files.
        """
        if not self.dev_tables_json_path.exists():
            return {}
        
        # Load and cache
        if self.dev_tables_cache is None:
            try:
                with open(self.dev_tables_json_path, 'r', encoding='utf-8') as f:
                    self.dev_tables_cache = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load dev_tables.json: {e}")
                return {}
        
        descriptions = {}
        
        # Find database in JSON
        for db in self.dev_tables_cache:
            if db.get('db_id') == db_name:
                table_names = db.get('table_names_original', [])
                column_names = db.get('column_names_original', [])
                
                # Get column descriptions if available
                column_descriptions = db.get('column_descriptions', {})
                
                for col_idx, col_info in enumerate(column_names):
                    if col_info[0] == -1:  # Skip * column
                        continue
                    
                    table_idx = col_info[0]
                    column_name = col_info[1]
                    
                    if table_idx >= len(table_names):
                        continue
                    
                    table_name = table_names[table_idx]
                    
                    # Get description if available
                    col_desc = column_descriptions.get(str(col_idx), "")
                    
                    if col_desc:
                        descriptions[(table_name, column_name)] = SMEFieldDescription(
                            table=table_name,
                            column=column_name,
                            description=col_desc,
                            source="json"
                        )
                
                break
        
        return descriptions
    
    def get_field_description(
        self, 
        db_name: str, 
        table: str, 
        column: str
    ) -> Optional[SMEFieldDescription]:
        """
        Get SME description for a specific field.
        
        Args:
            db_name: Database name
            table: Table name
            column: Column name
            
        Returns:
            SMEFieldDescription or None if not found
        """
        descriptions = self.load_database_descriptions(db_name)
        return descriptions.get((table, column))