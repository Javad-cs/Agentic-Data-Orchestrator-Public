import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from final_sql_w_cand_voting.query_masker import QueryMasker

@pytest.fixture
def masker():
    return QueryMasker()

def test_basic_masking(masker):
    """Test basic masking of strings, numbers, and dates."""
    question = "Who played 'Batman' in 2008?"
    result = masker.mask(question)
    
    # Check structure
    assert "<STR_" in result.masked_text
    assert "<NUM_" in result.masked_text
    assert "Batman" not in result.masked_text
    assert "2008" not in result.masked_text
    
    # Check unmasking
    assert result.unmask(result.masked_text) == question

def test_proper_noun_spans(masker):
    """Test contiguous proper noun spans (Bruce Wayne)."""
    question = "Is Bruce Wayne richer than Tony Stark?"
    result = masker.mask(question)
    
    # Should be SINGLE placeholders for full names
    assert result.masked_text.count("<NAME_") == 2
    assert "Bruce" not in result.masked_text
    assert "Wayne" not in result.masked_text
    
    # Verify mapping
    values = list(result.entity_map.values())
    assert "Bruce Wayne" in values
    assert "Tony Stark" in values

def test_connector_spans(masker):
    """Test connectors inside proper nouns (University of Berlin)."""
    question = "Show students from the University of Berlin."
    result = masker.mask(question)
    
    # "University of Berlin" should be ONE token
    assert result.masked_text.count("<NAME_") == 1
    assert "University of Berlin" in result.entity_map.values()
    
    # "United States of America" check
    q2 = "Who rules the United States of America?"
    r2 = masker.mask(q2)
    assert "United States of America" in r2.entity_map.values()

def test_connector_backtracking(masker):
    """Test that connectors at the end of a span are dropped."""
    # "Age of" -> "Age" (masked) + "of" (unmasked)
    question = "What is the Age of?" 
    result = masker.mask(question)
    
    # Should NOT capture "of" because it's not followed by a capitalized word
    assert "Age of" not in result.entity_map.values()
    assert "Age" in result.entity_map.values()

def test_punctuation_boundaries(masker):
    """Test punctuation sticking to words (New York?)."""
    question = "Do you live in New York?"
    result = masker.mask(question)
    
    # "New York" should be masked despite the "?"
    assert "<NAME_" in result.masked_text
    assert "?" in result.masked_text
    assert "New York" in result.entity_map.values()
    
    # Round trip check
    assert result.unmask(result.masked_text) == question

def test_decimal_safety(masker):
    """Test float vs integer masking (10.5 vs 10)."""
    question = "Is the rating 10.5 or 10?"
    result = masker.mask(question)
    
    # Should catch 10.5 as one token
    assert "10.5" in result.entity_map.values()
    assert "10" in result.entity_map.values()
    
    # Ensure 10.5 didn't get split
    assert result.masked_text.count("<NUM_") == 2

def test_parentheses_handling(masker):
    """Test entities inside parentheses (California)."""
    question = "I visited (California) last year."
    result = masker.mask(question)
    
    assert "California" in result.entity_map.values()
    # The parens should remain in the text
    assert "(" in result.masked_text
    assert ")" in result.masked_text

def test_smart_quotes(masker):
    """Test smart quotes and exotic punctuation."""
    question = "He said ‘Hello’ to “Batman”."
    result = masker.mask(question)
    
    # Should catch 'Hello' and "Batman"
    assert len(result.entity_map) >= 2
    assert result.unmask(result.masked_text) == question

def test_unicode_names(masker):
    """Test non-ASCII names (Émile)."""
    question = "Who is Émile Zola?"
    result = masker.mask(question)
    
    # Should capture "Émile Zola" as one name
    assert "Émile Zola" in result.entity_map.values()

def test_unmask_overlap(masker):
    """Test protection against placeholder overlap (NUM_1 vs NUM_10)."""
    # Create enough numbers to trigger double digits
    question = "Numbers 1 2 3 4 5 6 7 8 9 10 11."
    result = masker.mask(question)
    
    # Verify NUM_10+ exists
    assert "<NUM_10>" in result.masked_text
    
    # Verify clean round-trip
    # Logic failure here would result in "NUM_10" becoming "10" then "1" getting unmasked inside it
    assert result.unmask(result.masked_text) == question
    
def test_span_suffix_preserved(masker):
    q = "Is Bruce Wayne richer than Tony Stark?"
    r = masker.mask(q)
    assert "?" in r.masked_text
    assert "Tony Stark" in r.entity_map.values()
    assert r.unmask(r.masked_text) == q