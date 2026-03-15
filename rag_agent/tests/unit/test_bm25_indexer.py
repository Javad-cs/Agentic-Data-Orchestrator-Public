# tests/unit/test_bm25_indexer.py

import pytest
import asyncpg
from src.ingestion.indexers.bm25_tokenizer import BM25Tokenizer
from src.ingestion.indexers.bm25_indexer import BM25Indexer


# Tokenizer tests
def test_tokenizer_basic():
    """Test basic tokenization"""
    tokenizer = BM25Tokenizer()
    
    text = "This is a test"
    tokens = tokenizer.tokenize(text)
    
    # Stopwords 'is', 'a' should be removed
    assert 'this' in tokens or 'test' in tokens
    assert 'is' not in tokens
    assert 'a' not in tokens


def test_tokenizer_korean():
    """Test Korean tokenization"""
    tokenizer = BM25Tokenizer()
    
    text = "한국어 텍스트입니다"
    tokens = tokenizer.tokenize(text)
    
    assert len(tokens) > 0
    assert '한국어' in tokens
    assert '텍스트입니다' in tokens


def test_tokenizer_mixed():
    """Test mixed Korean-English"""
    tokenizer = BM25Tokenizer()
    
    text = "Upstage API는 excellent합니다"
    tokens = tokenizer.tokenize(text)
    
    assert 'upstage' in tokens
    assert 'api' in tokens
    assert 'excellent' in tokens
    assert '합니다' in tokens


def test_tokenizer_frequencies():
    """Test term frequency counting"""
    tokenizer = BM25Tokenizer()
    
    text = "test test test word word"
    freqs = tokenizer.tokenize_with_frequencies(text)
    
    assert freqs['test'] == 3
    assert freqs['word'] == 2


def test_tokenizer_min_length():
    """Test minimum token length filter"""
    tokenizer = BM25Tokenizer(min_token_length=3)
    
    text = "a ab abc abcd"
    tokens = tokenizer.tokenize(text)
    
    assert 'a' not in tokens
    assert 'ab' not in tokens
    assert 'abc' in tokens
    assert 'abcd' in tokens


def test_tokenizer_stopwords():
    """Test stopword removal"""
    tokenizer = BM25Tokenizer(remove_stopwords=True)
    
    text = "the quick brown fox"
    tokens = tokenizer.tokenize(text)
    
    assert 'the' not in tokens  # Stopword removed
    assert 'quick' in tokens
    assert 'brown' in tokens
    assert 'fox' in tokens


def test_tokenizer_no_stopwords():
    """Test keeping stopwords"""
    tokenizer = BM25Tokenizer(remove_stopwords=False)
    
    text = "the quick brown fox"
    tokens = tokenizer.tokenize(text)
    
    assert 'the' in tokens  # Stopword kept
    assert 'quick' in tokens


def test_tokenizer_numbers():
    """Test number tokenization"""
    tokenizer = BM25Tokenizer()
    
    text = "SCM440 and SUS304 with speed 200"
    tokens = tokenizer.tokenize(text)
    
    assert 'scm' in tokens
    assert '440' in tokens
    assert 'sus' in tokens
    assert '304' in tokens
    assert '200' in tokens


def test_tokenizer_empty():
    """Test empty text"""
    tokenizer = BM25Tokenizer()
    
    tokens = tokenizer.tokenize("")
    assert tokens == []
    
    tokens = tokenizer.tokenize("   ")
    assert tokens == []


def test_tokenizer_unique_terms():
    """Test unique terms extraction"""
    tokenizer = BM25Tokenizer()
    
    text = "test test word word"
    unique = tokenizer.get_unique_terms(text)
    
    assert len(unique) == 2
    assert 'test' in unique
    assert 'word' in unique


# Note: BM25Indexer tests require a real PostgreSQL connection
# These are integration tests, not unit tests
# They should be in tests/integration/ instead

@pytest.mark.integration
@pytest.mark.asyncio
async def test_bm25_indexer_basic():
    """
    Integration test for BM25 indexer.
    
    Requires PostgreSQL to be running with schema loaded.
    """
    # This would need actual DB connection
    # Skipping detailed implementation here
    pass