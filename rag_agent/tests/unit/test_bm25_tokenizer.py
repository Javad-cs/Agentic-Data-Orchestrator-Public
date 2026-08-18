# tests/unit/test_bm25_tokenizer.py

import pytest
from src.ingestion.indexers.bm25_tokenizer import BM25Tokenizer


def test_tokenizer_basic():
    """Test basic tokenization with stopword removal"""
    tokenizer = BM25Tokenizer()
    
    text = "Hello is a test"  #  Changed "This" to "Hello"
    tokens = tokenizer.tokenize(text)
    
    # Both content words should remain
    assert 'hello' in tokens
    assert 'test' in tokens
    
    # Stopwords should be removed
    assert 'is' not in tokens
    assert 'a' not in tokens

def test_tokenizer_korean():
    """Test Korean tokenization"""
    tokenizer = BM25Tokenizer()
    
    text = "한국어 텍스트입니다"
    tokens = tokenizer.tokenize(text)
    
    assert len(tokens) > 0
    assert '한국어' in tokens
    assert '텍스트입니다' in tokens  # Note: Regex treats as single token


def test_tokenizer_mixed():
    """Test mixed Korean-English tokenization"""
    tokenizer = BM25Tokenizer()
    
    text = "Upstage API는 excellent합니다"
    tokens = tokenizer.tokenize(text)
    
    # English words (lowercased)
    assert 'upstage' in tokens
    assert 'api' in tokens
    assert 'excellent' in tokens
    
    # Korean content word
    assert '합니다' in tokens
    
    # Korean stopword '는' should be removed
    assert '는' not in tokens  #Verify stopword removal


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


def test_tokenizer_case_sensitivity():
    """Test lowercase conversion"""
    tokenizer = BM25Tokenizer(lowercase=True)
    
    text = "HELLO World TeSt"
    tokens = tokenizer.tokenize(text)
    
    assert 'hello' in tokens
    assert 'world' in tokens
    assert 'test' in tokens
    assert 'HELLO' not in tokens


def test_tokenizer_no_lowercase():
    """Test preserving case when lowercase=False"""
    tokenizer = BM25Tokenizer(lowercase=False)
    
    text = "HELLO World TeSt"
    tokens = tokenizer.tokenize(text)
    
    # Properly extracts both uppercase and mixed case
    assert 'HELLO' in tokens
    assert 'World' in tokens
    assert 'TeSt' in tokens
    
    # Lowercase versions should NOT be present
    assert 'hello' not in tokens
    assert 'world' not in tokens


def test_tokenizer_max_length():
    """Test maximum token length filter"""
    tokenizer = BM25Tokenizer(max_token_length=5)
    
    text = "short verylongword"
    tokens = tokenizer.tokenize(text)
    
    assert 'short' in tokens
    assert 'verylongword' not in tokens  # Exceeds max_length


def test_tokenizer_korean_stopwords():
    """Test Korean stopword removal"""
    tokenizer = BM25Tokenizer(remove_stopwords=True)
    
    text = "이것은 테스트입니다"  # "This is a test"
    tokens = tokenizer.tokenize(text)
    
    # '은' (topic marker) should be removed
    assert '은' not in tokens
    # Content words should remain
    assert '이것' in tokens or '테스트입니다' in tokens
    
def test_tokenizer_case_sensitive_stopwords():
    """Test that stopwords are removed case-insensitively even with lowercase=False"""
    tokenizer = BM25Tokenizer(lowercase=False, remove_stopwords=True)
    
    text = "The QUICK Brown FOX"
    tokens = tokenizer.tokenize(text)
    
    # "The" should be removed (stopword, even when capitalized)
    assert 'The' not in tokens
    assert 'THE' not in tokens
    
    # Content words should remain with original case
    assert 'QUICK' in tokens
    assert 'Brown' in tokens
    assert 'FOX' in tokens