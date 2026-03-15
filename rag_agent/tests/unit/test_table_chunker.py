import pytest
from src.config.models import ChunkingConfig
from src.ingestion.chunkers.table_chunker import TableChunker

# --- Fixture: Sets up a fresh config for every test function ---
@pytest.fixture
def base_config():
    """Creates a default config object for testing"""
    config = ChunkingConfig()
    # Use gpt-4o or cl100k_base to ensure accurate token counting in tests
    config.tokenizer_model = "gpt-4o"
    return config

def test_large_table_split(base_config):
    """Large table should split into multiple parents"""
    # 1. Setup Table Data
    header = "| Insert | Material | Speed | Feed | Coating |"
    sep = "|--------|----------|-------|------|---------|"
    # Create 100 rows to ensure it's "large"
    rows = [f"| A{i:03d} | SCM440 | 200 | 0.25 | TiAlN |" for i in range(100)]
    table_md = "\n".join([header, sep] + rows)
    
    # 2. Configure Limits (Modify the config object)
    # We set max tokens to 500 so the ~1500 token table MUST split
    base_config.table.parent_max_tokens = 500
    
    # 3. Initialize Chunker with the Config
    chunker = TableChunker(config=base_config)
    parents, children = chunker.chunk_table(table_md, "test_doc")
    
    # 4. Assertions
    assert len(parents) > 1, f"Should split with limit 500. Got {len(parents)}"
    assert parents[0].is_split is True
    assert parents[0].chunk_id == "test_doc_table_part1"
    
    # Verify row coverage
    total_rows_covered = 0
    for p in parents:
        # Calculate how many rows this parent holds
        total_rows_covered += (p.end_row_idx - p.start_row_idx)
    assert total_rows_covered == 100

def test_missing_separator_recovery(base_config):
    """Test that chunker fixes tables where Upstage dropped the separator"""
    malformed_table = """
| Col A | Col B |
| Val 1 | Val 2 |
| Val 3 | Val 4 |
"""
    # Use default config settings
    chunker = TableChunker(config=base_config)
    parents, children = chunker.chunk_table(malformed_table, "doc_missing_sep")
    
    assert len(children) > 0
    # The chunker should have invented a separator line
    assert "|---" in children[0].text
    # "Val 1" should be treated as data, not lost
    assert "Val 1" in children[0].text

def test_single_row_table(base_config):
    """Test table with fewer rows than min_rows_per_child"""
    tiny_table = """
| Header |
|--------|
| Lone Row |
"""
    # Set min rows to 3, but provide only 1 row
    base_config.table.min_rows_per_child = 3
    
    chunker = TableChunker(config=base_config)
    parents, children = chunker.chunk_table(tiny_table, "tiny_doc")
    
    assert len(children) == 1
    assert "Lone Row" in children[0].text