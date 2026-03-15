import pytest
from src.generation.safety_check import SafetyChecker, SafetyCheckResult


class TestSafetyChecker:
    """Test suite for SafetyChecker"""
    
    @pytest.fixture
    def checker(self):
        """Create a SafetyChecker instance for testing"""
        return SafetyChecker(
            check_no_answer=True,
            check_citation=True,
            min_answer_length=10,
            use_nli=False
        )
    
    def test_good_answer_with_citations(self, checker):
        """Test that good answer with citations passes"""
        answer = "스테인레스강 가공에는 PVD 코팅이 적합합니다 [1]. 특히 AlTiN 코팅이 효과적입니다 [2]."
        context = "PVD 코팅은 고속 가공에 적합합니다."
        
        result = checker.check(answer, context)
        
        assert result.passed is True
        assert result.confidence > 0.8
        assert result.details['citation_count'] == 2
        assert 'no_answer_check' in result.details
    
    def test_answer_without_citations(self, checker):
        """Test that answer without citations fails citation check"""
        answer = "스테인레스강 가공에는 PVD 코팅이 적합합니다. 매우 효과적입니다."
        context = "PVD 코팅은 고속 가공에 적합합니다."
        
        result = checker.check(answer, context)
        
        assert result.passed is False
        assert result.confidence == 0.3
        assert "citation" in result.reason.lower()
        assert result.details['citation_count'] == 0
    
    def test_refusal_answer_english(self, checker):
        """Test detection of English refusal phrases"""
        answer = "I cannot answer this question based on the available information."
        context = "Some context here."
        
        result = checker.check(answer, context)
        
        assert result.passed is False
        assert result.confidence == 0.0
        assert "refusal" in result.reason.lower() or "phrase" in result.reason.lower()
    
    def test_refusal_answer_korean(self, checker):
        """Test detection of Korean refusal phrases"""
        answer = "죄송합니다. 해당 정보를 찾을 수 없습니다."
        context = "Some context here."
        
        result = checker.check(answer, context)
        
        assert result.passed is False
        assert result.confidence == 0.0
    
    def test_answer_too_short(self, checker):
        """Test that very short answers fail"""
        answer = "PVD"
        context = "Some context."
        
        result = checker.check(answer, context)
        
        assert result.passed is False
        assert "short" in result.reason.lower()
        assert result.details['answer_length'] < 10
    
    def test_answer_at_min_length(self, checker):
        """Test answer exactly at minimum length"""
        answer = "PVD 코팅 [1]"  # Exactly 10 chars
        context = "Context"
        
        result = checker.check(answer, context)
        
        # Should pass length check, but might fail citation or other checks
        assert result.details['answer_length'] >= 10
    
    def test_multiple_citations(self, checker):
        """Test detection of multiple citations"""
        answer = "PVD 코팅 [1] 과 AlTiN [2] 그리고 CrN [3] 이 좋습니다."
        context = "Context"
        
        result = checker.check(answer, context)
        
        assert result.details['citation_count'] == 3
    
    def test_checker_without_citation_requirement(self):
        """Test checker when citation check is disabled"""
        checker = SafetyChecker(
            check_no_answer=True,
            check_citation=False,  # Disabled
            min_answer_length=10
        )
        
        answer = "스테인레스강 가공에는 PVD 코팅이 적합합니다."  # No citations
        context = "Context"
        
        result = checker.check(answer, context)
        
        # Should pass since citation check is disabled
        assert result.passed is True
    
    def test_checker_without_no_answer_check(self):
        """Test checker when no-answer check is disabled"""
        checker = SafetyChecker(
            check_no_answer=False,  # Disabled
            check_citation=True,
            min_answer_length=10
        )
        
        answer = "I don't know the answer [1]."
        context = "Context"
        
        result = checker.check(answer, context)
        
        # Should pass since we're not checking for refusal phrases
        assert result.passed is True
    
    def test_empty_answer(self, checker):
        """Test empty or whitespace-only answer"""
        answer = "   "
        context = "Context"
        
        result = checker.check(answer, context)
        
        assert result.passed is False
        assert "short" in result.reason.lower()
    
    def test_case_insensitive_refusal_detection(self, checker):
        """Test that refusal detection is case-insensitive"""
        answer = "I DON'T KNOW the answer to this question."
        context = "Context"
        
        result = checker.check(answer, context)
        
        assert result.passed is False


class TestSafetyCheckResult:
    """Test SafetyCheckResult dataclass"""
    
    def test_result_creation(self):
        """Test creating a SafetyCheckResult"""
        result = SafetyCheckResult(
            passed=True,
            confidence=0.95,
            reason="All checks passed",
            details={'test': 'value'}
        )
        
        assert result.passed is True
        assert result.confidence == 0.95
        assert result.reason == "All checks passed"
        assert result.details['test'] == 'value'