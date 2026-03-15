import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Comprehensive pytest test suite for SME Parser.
Tests CSV parsing, JSON fallback, error handling, and edge cases.

Run with: pytest test_comprehensive_sme.py -v
"""

import pytest
from pathlib import Path
from schema_linking.sme_parser import SMEParser, SMEFieldDescription
from config import settings


class TestSMEParserPaths:
    """Test that paths are configured correctly."""
    
    def test_bird_root_path_exists(self):
        """BIRD_ROOT_PATH must exist."""
        bird_path = Path(settings.bird_root_path)
        assert bird_path.exists(), f"BIRD_ROOT_PATH does not exist: {bird_path}"
    
    def test_dev_tables_json_exists(self):
        """dev_tables.json must exist in BIRD_ROOT_PATH."""
        bird_path = Path(settings.bird_root_path)
        dev_tables = bird_path / "dev_tables.json"
        assert dev_tables.exists(), f"dev_tables.json not found at {dev_tables}"
    
    def test_dev_databases_folder_exists(self):
        """dev_databases/ folder must exist."""
        bird_path = Path(settings.bird_root_path)
        dev_databases = bird_path / "dev_databases"
        assert dev_databases.exists(), f"dev_databases/ not found at {dev_databases}"
    
    def test_superhero_database_exists(self):
        """Test database (superhero) must exist."""
        bird_path = Path(settings.bird_root_path)
        superhero = bird_path / "dev_databases" / "superhero"
        assert superhero.exists(), f"superhero database not found at {superhero}"


class TestSMEParserCSVLoading:
    """Test CSV description loading."""
    
    @pytest.fixture
    def parser(self):
        """Create SME parser instance."""
        return SMEParser(settings.bird_root_path)
    
    def test_load_superhero_descriptions(self, parser):
        """Load superhero descriptions from CSV files."""
        descriptions = parser.load_database_descriptions("superhero")
        
        # Should load multiple descriptions
        assert len(descriptions) > 0, "No descriptions loaded"
        
        # All keys should be (table, column) tuples
        for key in descriptions.keys():
            assert isinstance(key, tuple), f"Key should be tuple, got {type(key)}"
            assert len(key) == 2, f"Key should have 2 elements, got {len(key)}"
            table, column = key
            assert isinstance(table, str) and table, "Table name must be non-empty string"
            assert isinstance(column, str) and column, "Column name must be non-empty string"
    
    def test_csv_descriptions_have_source(self, parser):
        """CSV descriptions should have source='csv'."""
        descriptions = parser.load_database_descriptions("superhero")
        
        # At least some should be from CSV (superhero has database_description/)
        csv_sources = [desc for desc in descriptions.values() if desc.source == "csv"]
        assert len(csv_sources) > 0, "No CSV descriptions found"
    
    def test_known_fields_exist(self, parser):
        """Test that known superhero fields exist."""
        descriptions = parser.load_database_descriptions("superhero")
        
        # Known fields from superhero database
        known_fields = [
            ("alignment", "id"),
            ("alignment", "alignment"),
            ("gender", "id"),
            ("gender", "gender"),
        ]
        
        for table, column in known_fields:
            assert (table, column) in descriptions, \
                f"Known field {table}.{column} not found in descriptions"
    
    def test_descriptions_are_non_empty(self, parser):
        """All descriptions should have non-empty text."""
        descriptions = parser.load_database_descriptions("superhero")
        
        for key, desc in descriptions.items():
            assert desc.description, \
                f"Description for {key} is empty"
            assert len(desc.description) > 0, \
                f"Description for {key} has zero length"
    
    def test_csv_combines_column_and_value_descriptions(self, parser):
        """CSV parser should combine column_description and value_description."""
        descriptions = parser.load_database_descriptions("superhero")
        
        # alignment.alignment has rich value_description
        alignment_desc = descriptions.get(("alignment", "alignment"))
        if alignment_desc and alignment_desc.source == "csv":
            # Should contain both short description and detailed explanation
            desc_text = alignment_desc.description.lower()
            assert "alignment" in desc_text, "Description missing key term"
            # Value description has detailed examples
            assert len(alignment_desc.description) > 50, \
                "Description too short, likely missing value_description"


class TestSMEParserJSONFallback:
    """Test JSON fallback when CSV is not available."""
    
    @pytest.fixture
    def parser(self):
        """Create SME parser instance."""
        return SMEParser(settings.bird_root_path)
    
    def test_nonexistent_database_returns_empty(self, parser):
        """Non-existent database should return empty dict, not crash."""
        descriptions = parser.load_database_descriptions("nonexistent_db_12345")
        assert descriptions == {}, "Should return empty dict for non-existent database"
    
    def test_json_fallback_works(self, parser):
        """Test that JSON loading works (may supplement CSV)."""
        # Load any database
        descriptions = parser.load_database_descriptions("superhero")
        
        # Check if any came from JSON (might be supplemental to CSV)
        json_sources = [desc for desc in descriptions.values() if desc.source == "json"]
        
        # Either all CSV, or some JSON (depends on data completeness)
        # Just verify JSON source handling works
        for desc in json_sources:
            assert desc.description, "JSON description should not be empty"
            assert desc.table, "JSON description should have table"
            assert desc.column, "JSON description should have column"


class TestSMEParserGetField:
    """Test get_field_description method."""
    
    @pytest.fixture
    def parser(self):
        """Create SME parser instance."""
        return SMEParser(settings.bird_root_path)
    
    def test_get_existing_field(self, parser):
        """Get description for existing field."""
        desc = parser.get_field_description("superhero", "alignment", "alignment")
        
        assert desc is not None, "Should find alignment.alignment"
        assert desc.table == "alignment"
        assert desc.column == "alignment"
        assert desc.description
    
    def test_get_nonexistent_field_returns_none(self, parser):
        """Get description for non-existent field returns None."""
        desc = parser.get_field_description("superhero", "fake_table", "fake_column")
        assert desc is None, "Should return None for non-existent field"
    
    def test_get_field_from_nonexistent_db_returns_none(self, parser):
        """Get field from non-existent database returns None."""
        desc = parser.get_field_description("fake_db", "table", "column")
        assert desc is None, "Should return None for non-existent database"


class TestSMEFieldDescription:
    """Test SMEFieldDescription dataclass."""
    
    def test_field_description_creation(self):
        """Test creating SMEFieldDescription."""
        desc = SMEFieldDescription(
            table="users",
            column="age",
            description="The user's age in years",
            data_format="integer",
            value_description="Age ranges from 0 to 120",
            source="csv"
        )
        
        assert desc.table == "users"
        assert desc.column == "age"
        assert desc.description == "The user's age in years"
        assert desc.source == "csv"
    
    def test_field_description_repr(self):
        """Test __repr__ method."""
        desc = SMEFieldDescription(
            table="users",
            column="age",
            description="Age",
            source="csv"
        )
        
        repr_str = repr(desc)
        assert "users" in repr_str
        assert "age" in repr_str
        assert "csv" in repr_str


class TestSMEParserRobustness:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def parser(self):
        """Create SME parser instance."""
        return SMEParser(settings.bird_root_path)
    
    def test_handles_missing_csv_gracefully(self, parser, tmp_path):
        """Parser should handle missing CSV files gracefully."""
        # Use a temp path with no CSV files
        fake_bird_path = tmp_path / "fake_bird"
        fake_bird_path.mkdir()
        (fake_bird_path / "dev_databases").mkdir()
        (fake_bird_path / "dev_databases" / "test_db").mkdir()
        
        # Create parser with fake path
        temp_parser = SMEParser(fake_bird_path)
        
        # Should not crash, just return empty
        descriptions = temp_parser.load_database_descriptions("test_db")
        assert descriptions == {}
    
    def test_handles_malformed_json_gracefully(self, parser, tmp_path):
        """Parser should handle malformed JSON gracefully."""
        # Create temp path with malformed JSON
        fake_bird_path = tmp_path / "fake_bird"
        fake_bird_path.mkdir()
        
        # Write malformed JSON
        json_path = fake_bird_path / "dev_tables.json"
        json_path.write_text("{malformed json")
        
        # Should not crash
        temp_parser = SMEParser(fake_bird_path)
        descriptions = temp_parser.load_database_descriptions("any_db")
        assert descriptions == {}
    
    def test_caches_json_loading(self, parser):
        """JSON should be loaded and cached once."""
        # First load
        desc1 = parser.load_database_descriptions("superhero")
        cache1 = parser.dev_tables_cache
        
        # Second load
        desc2 = parser.load_database_descriptions("superhero")
        cache2 = parser.dev_tables_cache
        
        # Should use same cache object
        assert cache1 is cache2, "Should reuse cached JSON"


# Integration test
def test_sme_parser_integration():
    """End-to-end integration test."""
    parser = SMEParser(settings.bird_root_path)
    
    # Load superhero database
    descriptions = parser.load_database_descriptions("superhero")
    
    # Should have descriptions
    assert len(descriptions) > 0
    
    # All should be valid
    for (table, column), desc in descriptions.items():
        assert table
        assert column
        assert desc.description
        assert desc.source in ("csv", "json")
    
    # Get specific field
    alignment_desc = parser.get_field_description("superhero", "alignment", "alignment")
    assert alignment_desc is not None
    assert "alignment" in alignment_desc.description.lower()


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])