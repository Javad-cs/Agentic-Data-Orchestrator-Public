import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.retrieval.rerankers.upstage import UpstageReranker
from src.retrieval.rerankers.base import RerankResponse, CandidateDocument
from src.config.models import UpstageConfig


@pytest.fixture
def reranker():
    """Create reranker with test API key"""
    return UpstageReranker(api_key="test_key_123")


@pytest.fixture
def sample_documents():
    """Create sample documents for reranking"""
    return [
        CandidateDocument(
            id='doc1',
            text='Machine learning is a subset of AI',
            score=0.8,
            source='dense'
        ),
        CandidateDocument(
            id='doc2',
            text='Python is a programming language',
            score=0.7,
            source='sparse'
        ),
        CandidateDocument(
            id='doc3',
            text='Deep learning uses neural networks',
            score=0.9,
            source='dense'
        )
    ]


@pytest.fixture
def mock_api_response():
    """Mock successful API response"""
    return {
        'results': [
            {'index': 2, 'relevance_score': 0.95},  # doc3 most relevant
            {'index': 0, 'relevance_score': 0.85},  # doc1 second
            {'index': 1, 'relevance_score': 0.60}   # doc2 least
        ]
    }


@pytest.mark.asyncio
async def test_rerank_success(reranker, sample_documents, mock_api_response):
    """Test successful async reranking"""
    with patch.object(reranker, '_call_api', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_api_response['results']
        
        response = await reranker.rerank(  # ← AWAIT!
            query="What is deep learning?",
            documents=sample_documents
        )
        
        assert isinstance(response, RerankResponse)
        assert len(response.results) == 3
        
        # Check reranking worked
        assert response.results[0].document_id == 'doc3'
        assert response.results[0].score == 0.95


@pytest.mark.asyncio
async def test_rerank_with_top_k(reranker, sample_documents, mock_api_response):
    """Test reranking with top_k limit"""
    with patch.object(reranker, '_call_api', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_api_response['results']
        
        response = await reranker.rerank(
            query="What is deep learning?",
            documents=sample_documents,
            top_k=2
        )
        
        assert len(response.results) == 2
        assert response.total_reranked == 2


@pytest.mark.asyncio
async def test_rerank_empty_documents(reranker):
    """Test reranking with empty document list"""
    response = await reranker.rerank(
        query="test query",
        documents=[]
    )
    
    assert len(response.results) == 0


@pytest.mark.asyncio
async def test_candidate_document_validation():
    """Test CandidateDocument validates required fields"""
    # Valid document
    doc = CandidateDocument(id="doc1", text="Content")
    assert doc.id == "doc1"
    
    # Empty ID
    with pytest.raises(ValueError, match="id cannot be empty"):
        CandidateDocument(id="", text="Content")
    
    # Empty text
    with pytest.raises(ValueError, match="text cannot be empty"):
        CandidateDocument(id="doc1", text="")


@pytest.mark.asyncio
async def test_rerank_disabled(sample_documents):
    """Test reranking when disabled"""
    config = UpstageConfig(
        api_key="test_key",
        reranking_enabled=False
    )
    reranker = UpstageReranker(config=config)
    
    response = await reranker.rerank(
        query="test query",
        documents=sample_documents
    )
    
    # Should use passthrough
    assert response.metadata['enabled'] == False
    assert len(response.results) == 3


@pytest.mark.asyncio
async def test_close_cleanup(reranker):
    """Test HTTP client cleanup"""
    await reranker.close()
    # Should close without errors