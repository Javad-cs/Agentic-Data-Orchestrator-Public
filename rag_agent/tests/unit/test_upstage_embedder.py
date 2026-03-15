import pytest
import requests
import numpy as np
from unittest.mock import Mock, patch
from src.ingestion.embedders.upstage import UpstageEmbedder
from src.ingestion.embedders.base import EmbeddingResult, BatchEmbeddingResult
from src.config.models import UpstageConfig  # ← ADD THIS IMPORT


@pytest.fixture
def embedder():
    """Create embedder with test API key"""
    return UpstageEmbedder(api_key="test_key_123")


@pytest.fixture
def mock_api_response():
    """Mock successful API response"""
    return {
        "data": [
            {"embedding": [0.1] * 4096, "index": 0},
            {"embedding": [0.2] * 4096, "index": 1}
        ],
        "usage": {"total_tokens": 50}
    }


def test_embedder_initialization_with_config():
    """Test embedder initializes with config object"""
    config = UpstageConfig(
        api_key="test_key_123",
        embedding_model_passage="custom-passage-model",  # Correct field
        embedding_model_query="custom-query-model",      # Correct field
        embedding_dimension=4096
    )
    
    embedder = UpstageEmbedder(config=config)
    
    assert embedder.api_key == "test_key_123"
    assert embedder.model == "custom-passage-model"  # Default mode is "passage"
    assert embedder.get_dimension() == 4096


def test_embedder_initialization_without_config():
    """Test embedder initializes with individual parameters"""
    embedder = UpstageEmbedder(
        api_key="test_key",
        model="solar-embedding-1-large-passage",
        dimension=4096
    )
    
    assert embedder.api_key == "test_key"
    assert embedder.model == "solar-embedding-1-large-passage"
    assert embedder.get_dimension() == 4096


def test_embedder_requires_api_key_or_config():
    """Test embedder raises error without API key or config"""
    with pytest.raises(ValueError, match="Either config or api_key must be provided"):
        UpstageEmbedder()


def test_text_validation(embedder):
    """Test text validation"""
    # Empty text
    with pytest.raises(ValueError, match="cannot be empty"):
        embedder.validate_text("")
    
    # Whitespace only
    with pytest.raises(ValueError, match="cannot be empty"):
        embedder.validate_text("   ")
    
    # Valid text
    validated = embedder.validate_text("  Hello world  ")
    assert validated == "Hello world"
    
    # Truncation
    long_text = "a" * 10000
    truncated = embedder.validate_text(long_text, max_length=100)
    assert len(truncated) == 100


@patch('src.ingestion.embedders.upstage.requests.post')
def test_embed_single_text(mock_post, embedder, mock_api_response):
    """Test embedding single text"""
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {
            "data": [{"embedding": [0.1] * 4096, "index": 0}],
            "usage": {"total_tokens": 10}
        }
    )
    
    result = embedder.embed("Test text")
    
    assert isinstance(result, EmbeddingResult)
    assert result.text == "Test text"
    assert isinstance(result.embedding, np.ndarray)
    assert len(result.embedding) == 4096
    assert result.metadata['model'] == "solar-embedding-1-large-passage"


@patch('src.ingestion.embedders.upstage.requests.post')
def test_embed_batch(mock_post, embedder, mock_api_response):
    """Test batch embedding"""
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: mock_api_response
    )
    
    texts = ["Text 1", "Text 2"]
    result = embedder.embed_batch(texts)
    
    assert isinstance(result, BatchEmbeddingResult)
    assert len(result.results) == 2
    assert result.total_tokens == 50
    assert all(isinstance(r.embedding, np.ndarray) for r in result.results)
    assert all(len(r.embedding) == 4096 for r in result.results)


@patch('src.ingestion.embedders.upstage.requests.post')
def test_large_batch_splitting(mock_post, embedder):
    """Test that large batches are split correctly"""
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {
            "data": [{"embedding": [0.1] * 4096, "index": i} for i in range(100)],
            "usage": {"total_tokens": 1000}
        }
    )
    
    # Create 250 texts (should split into 3 batches)
    texts = [f"Text {i}" for i in range(250)]
    result = embedder.embed_batch(texts)
    
    assert len(result.results) == 250
    assert mock_post.call_count == 3  # 100 + 100 + 50


@patch('src.ingestion.embedders.upstage.requests.post')
def test_api_error_handling(mock_post, embedder):
    """Test API error handling"""
    mock_response = Mock(
        status_code=401,
        json=lambda: {"error": "Invalid API key"},
        text="Invalid API key"
    )
    
    http_error = requests.exceptions.HTTPError("401 Error")
    http_error.response = mock_response  # ← Attach response
    mock_response.raise_for_status = Mock(side_effect=http_error)
    
    mock_post.return_value = mock_response
    
    with pytest.raises(requests.exceptions.HTTPError):
        embedder.embed("Test")


@patch('src.ingestion.embedders.upstage.requests.post')
@patch('src.ingestion.embedders.upstage.time.sleep')
def test_retry_on_rate_limit(mock_sleep, mock_post, embedder):
    """Test retry logic on rate limit (429)"""
    # First call: 429 with response attached
    mock_response_429 = Mock(status_code=429)
    http_error_429 = requests.exceptions.HTTPError("429")
    http_error_429.response = mock_response_429
    mock_response_429.raise_for_status = Mock(side_effect=http_error_429)
    
    # Second call: success
    mock_response_200 = Mock(
        status_code=200,
        json=lambda: {
            "data": [{"embedding": [0.1] * 4096, "index": 0}],
            "usage": {"total_tokens": 10}
        }
    )
    
    # Set side_effect with both responses
    mock_post.side_effect = [mock_response_429, mock_response_200]
    
    result = embedder.embed("Test")
    
    assert isinstance(result, EmbeddingResult)
    assert mock_post.call_count == 2  # First failed, second succeeded
    assert mock_sleep.call_count == 1  # Slept once between retries


def test_dimension_validation(embedder):
    """Test embedding dimension validation"""
    with patch('src.ingestion.embedders.upstage.requests.post') as mock_post:
        # Return wrong dimension
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "data": [{"embedding": [0.1] * 100, "index": 0}],  # Wrong dimension!
                "usage": {"total_tokens": 10}
            }
        )
        
        with pytest.raises(ValueError, match="Unexpected embedding dimension"):
            embedder.embed("Test")


def test_statistics_tracking(embedder):
    """Test usage statistics tracking"""
    with patch('src.ingestion.embedders.upstage.requests.post') as mock_post:
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "data": [{"embedding": [0.1] * 4096, "index": 0}],
                "usage": {"total_tokens": 10}
            }
        )
        
        # Make 3 requests
        embedder.embed("Test 1")
        embedder.embed("Test 2")
        embedder.embed("Test 3")
        
        stats = embedder.get_stats()
        assert stats['total_requests'] == 3
        assert stats['total_tokens_used'] == 30
        
        # Reset
        embedder.reset_stats()
        stats = embedder.get_stats()
        assert stats['total_requests'] == 0
        assert stats['total_tokens_used'] == 0


def test_empty_batch_error(embedder):
    """Test that empty batch raises error"""
    with pytest.raises(ValueError, match="cannot be empty"):
        embedder.embed_batch([])


def test_embedder_passage_mode():
    """Test embedder in passage mode"""
    embedder = UpstageEmbedder(
        api_key="test_key",
        mode="passage"
    )
    assert embedder.model == "solar-embedding-1-large-passage"
    assert embedder.mode == "passage"


def test_embedder_query_mode():
    """Test embedder in query mode"""
    embedder = UpstageEmbedder(
        api_key="test_key",
        mode="query"
    )
    assert embedder.model == "solar-embedding-1-large-query"
    assert embedder.mode == "query"


def test_embedder_invalid_mode():
    """Test embedder rejects invalid mode"""
    with pytest.raises(ValueError, match="Invalid mode"):
        UpstageEmbedder(api_key="test_key", mode="invalid")


def test_embedder_mode_with_config():
    """Test embedder uses correct model from config based on mode"""
    config = UpstageConfig(
        api_key="test_key",
        embedding_model_passage="passage-model",
        embedding_model_query="query-model"
    )
    
    # Passage mode
    embedder_passage = UpstageEmbedder(config=config, mode="passage")
    assert embedder_passage.model == "passage-model"
    
    # Query mode
    embedder_query = UpstageEmbedder(config=config, mode="query")
    assert embedder_query.model == "query-model"