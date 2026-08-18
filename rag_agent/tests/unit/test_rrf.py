import pytest
from src.retrieval.merge.rrf import reciprocal_rank_fusion


def test_rrf_partial_weights():
    """Test that partial weights don't crash"""
    dense = [{'child_id': 'doc1', 'text': 'text1', 'score': 0.9}]
    sparse = [{'child_id': 'doc1', 'text': 'text1', 'score': 5.0}]
    
    # Should not crash
    result = reciprocal_rank_fusion(
        dense, sparse,
        weights={'dense': 0.9}  # Missing 'sparse' key
    )
    
    assert len(result) == 1
    assert result[0].rrf_score > 0


def test_rrf_no_weights():
    """Test that None weights work"""
    dense = [{'child_id': 'doc1', 'text': 'text1', 'score': 0.9}]
    sparse = [{'child_id': 'doc1', 'text': 'text1', 'score': 5.0}]
    
    result = reciprocal_rank_fusion(dense, sparse, weights=None)
    
    assert len(result) == 1


def test_rrf_empty_text_from_dense():
    """Test that empty text from dense gets replaced by sparse"""
    dense = [{'child_id': 'doc1', 'text': '', 'score': 0.9}]  # Empty!
    sparse = [{'child_id': 'doc1', 'text': 'correct text', 'score': 5.0}]
    
    result = reciprocal_rank_fusion(dense, sparse)
    
    assert result[0].text == 'correct text'  # Updated from sparse


def test_rrf_empty_text_from_both():
    """Test that empty text from both sources gets logged"""
    dense = [{'child_id': 'doc1', 'text': '', 'score': 0.9}]
    sparse = [{'child_id': 'doc1', 'text': '', 'score': 5.0}]
    
    result = reciprocal_rank_fusion(dense, sparse)
    
    # Should still work but text is empty
    assert len(result) == 0


def test_rrf_weight_normalization():
    """Test that weights are normalized to sum to 1"""
    dense = [{'child_id': 'doc1', 'text': 'text1', 'score': 0.9}]
    sparse = [{'child_id': 'doc2', 'text': 'text2', 'score': 5.0}]
    
    # Unnormalized weights
    result = reciprocal_rank_fusion(
        dense, sparse,
        weights={'dense': 70, 'sparse': 30}  # Sum to 100, not 1
    )
    
    # Should still work (normalized internally)
    assert len(result) == 2


def test_rrf_negative_weights_error():
    """Test that negative weights raise error"""
    dense = [{'child_id': 'doc1', 'text': 'text1', 'score': 0.9}]
    sparse = [{'child_id': 'doc1', 'text': 'text1', 'score': 5.0}]
    
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion(
            dense, sparse,
            weights={'dense': -0.5, 'sparse': 0.5}
        )


def test_rrf_zero_weights_error():
    """Test that all-zero weights raise error"""
    dense = [{'child_id': 'doc1', 'text': 'text1', 'score': 0.9}]
    sparse = [{'child_id': 'doc1', 'text': 'text1', 'score': 5.0}]
    
    with pytest.raises(ValueError, match="At least one weight"):
        reciprocal_rank_fusion(
            dense, sparse,
            weights={'dense': 0, 'sparse': 0}
        )
        
def test_rrf_preserves_rank_despite_empty_text():
    """
    Test that RRF preserves rank from sparse even if text is empty,
    as long as dense provided valid text.
    """
    dense = [
        {'child_id': 'doc1', 'text': 'valid text from dense', 'score': 0.9}
    ]
    sparse = [
        {'child_id': 'doc1', 'text': '', 'score': 5.0}  # Empty text but rank 1!
    ]
    
    result = reciprocal_rank_fusion(dense, sparse, k=60)
    
    # Should have 1 result (not filtered out)
    assert len(result) == 1
    assert result[0].text == 'valid text from dense'
    
    # CRITICAL: Should have BOTH ranks contributing to score
    # RRF = 0.5/(60+1) + 0.5/(60+1) = 0.0164
    expected_score = 0.5/61 + 0.5/61
    assert abs(result[0].rrf_score - expected_score) < 0.0001
    
    # Verify both ranks were recorded
    assert result[0].dense_rank == 1
    assert result[0].sparse_rank == 1


def test_rrf_skips_sparse_only_empty_text():
    """
    Test that documents only in sparse with empty text ARE skipped.
    """
    dense = [
        {'child_id': 'doc1', 'text': 'valid text', 'score': 0.9}
    ]
    sparse = [
        {'child_id': 'doc2', 'text': '', 'score': 5.0}  # Sparse-only with empty text
    ]
    
    result = reciprocal_rank_fusion(dense, sparse)
    
    # Should only have doc1 (doc2 skipped)
    assert len(result) == 1
    assert result[0].document_id == 'doc1'