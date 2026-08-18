import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result from safety check"""
    passed: bool
    confidence: float  # 0-1
    issues: List[str]  # List of issues found
    nli_scores: Optional[List[float]] = None  # Per-sentence entailment scores
    
    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0


class NLISafetyChecker:
    """
    NLI-based safety checker for generated answers.
    
    Validates that generated answers are:
    1. Grounded in retrieved context (via NLI entailment)
    2. Have proper citations
    3. Meet minimum quality standards
    
    Uses a cross-encoder NLI model from HuggingFace to check
    if each sentence in the answer is entailed by the context.
    
    NLI Model Label Mapping (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7):
    - Index 0: Entailment (answer is supported by context) ← We use this
    - Index 1: Neutral (context is irrelevant to answer)
    - Index 2: Contradiction (answer contradicts context)
    
    Design:
    - Fast heuristic checks first (citations, length)
    - Optional deep NLI check (can be disabled for speed)
    - Batched predictions for performance
    - Graceful degradation on failures
    
    NLI prediction is CPU-intensive and blocks the event loop.
    This implementation uses `run_in_executor` to offload predictions to
    a thread pool, preventing FastAPI from freezing during checks.
    """
    
    def __init__(
        self,
        nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_threshold: float = 0.5,
        use_nli: bool = True,
        check_citations: bool = True,
        min_answer_length: int = 10
    ):
        """
        Initialize NLI safety checker.
        
        Args:
            nli_model_name: HuggingFace model for NLI
            nli_threshold: Minimum entailment score (0-1)
            use_nli: Enable deep NLI checking
            check_citations: Require citations in answer
            min_answer_length: Minimum answer length in characters
        """
        self.nli_model_name = nli_model_name
        self.nli_threshold = nli_threshold
        self.use_nli = use_nli
        self.check_citations = check_citations
        self.min_answer_length = min_answer_length
        
        # Lazy load NLI model (only if enabled)
        self._nli_model = None
        
        logger.info(
            f"NLISafetyChecker initialized "
            f"(nli={use_nli}, citations={check_citations}, min_len={min_answer_length})"
        )
    
    @property
    def nli_model(self):
        """Lazy load NLI model on first use"""
        if self._nli_model is None and self.use_nli:
            logger.info(f"Loading NLI model: {self.nli_model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._nli_model = CrossEncoder(self.nli_model_name)
                logger.info(" NLI model loaded")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                logger.warning("NLI checks will be disabled")
                self.use_nli = False
        
        return self._nli_model
    
    async def check(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> SafetyCheckResult:
        """
        Check if answer is safe and grounded.
        
        Args:
            answer: Generated answer text
            context_chunks: Retrieved context chunks
            citations: Citation metadata (optional)
            
        Returns:
            SafetyCheckResult with pass/fail and issues
        """
        issues = []
        nli_scores = None
        
        # Check 1: Minimum length
        if len(answer.strip()) < self.min_answer_length:
            issues.append(f"Answer too short ({len(answer)} < {self.min_answer_length} chars)")
        
        # Check 2: Citations presence
        if self.check_citations:
            has_citations = self._check_citations(answer, citations)
            if not has_citations:
                issues.append("No citations found in answer")
        
        # Check 3: No answer phrases
        if self._has_no_answer_phrases(answer):
            issues.append("Answer contains 'I don't know' type phrases")
        
        # Check 4: NLI entailment (optional, deep check)
        if self.use_nli and not issues:  # Only if basic checks pass
            entailment_passed, nli_scores = await self._check_entailment(
                answer, context_chunks
            )
            if not entailment_passed:
                avg_score = sum(nli_scores) / len(nli_scores) if nli_scores else 0
                issues.append(
                    f"Low entailment score ({avg_score:.2f} < {self.nli_threshold})"
                )
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(issues, nli_scores)
        
        # Pass if no issues
        passed = len(issues) == 0
        
        if not passed:
            logger.warning(f"Safety check failed: {', '.join(issues)}")
        else:
            logger.info(f"Safety check passed (confidence={confidence:.2f})")
        
        return SafetyCheckResult(
            passed=passed,
            confidence=confidence,
            issues=issues,
            nli_scores=nli_scores
        )
    
    def _check_citations(
        self,
        answer: str,
        citations: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """Check if answer has citations"""
        # Check for citation markers like [1], [2], etc.
        import re
        citation_pattern = r'\[\d+\]'
        has_markers = bool(re.search(citation_pattern, answer))
        
        # Also check if citations list is provided and non-empty
        has_citation_data = citations and len(citations) > 0
        
        return has_markers or has_citation_data
    
    def _has_no_answer_phrases(self, answer: str) -> bool:
        """Check for 'I don't know' type phrases"""
        answer_lower = answer.lower()
        
        no_answer_phrases = [
            "i don't know",
            "i do not know",
            "i'm not sure",
            "i am not sure",
            "cannot answer",
            "can't answer",
            "no information",
            "모르겠습니다",  # Korean: I don't know
            "알 수 없습니다",  # Korean: Cannot know
            "정보가 없습니다",  # Korean: No information
        ]
        
        return any(phrase in answer_lower for phrase in no_answer_phrases)
    
    async def _check_entailment(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> tuple[bool, List[float]]:
        """
        Check if answer is entailed by context using NLI.
         
        1. Uses correct index for entailment
        2. Batches all sentences in one prediction call for performance
        3. Uses run_in_executor to avoid blocking event loop
        
        Returns:
            (passed, nli_scores)
        """
        if not self.nli_model:
            logger.warning("NLI model not available, skipping entailment check")
            return True, []
        
        # Split answer into sentences
        sentences = self._split_sentences(answer)
        
        if not sentences:
            return True, []
        
        # Combine context chunks
        context = self._combine_context(context_chunks)
        
        if not context:
            logger.warning("No context available for entailment check")
            return True, []
        
        # Batch all sentences into one prediction call
        # Create all premise-hypothesis pairs at once
        pairs = [[context, sentence] for sentence in sentences]
        
        try:
            import asyncio
            import numpy as np
            
            loop = asyncio.get_event_loop()
            
            # Single prediction call for all sentences
            # This is much faster than calling predict() in a loop
            all_logits = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.nli_model.predict,
                pairs
            )
            
            # Process results
            scores = []
            for i, logits in enumerate(all_logits):
                # For MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7:
                # Index 0: Entailment
                # Index 1: Neutral
                # Index 2: Contradiction
                probs = self._softmax(logits)
                entailment_prob = probs[0]
                
                scores.append(entailment_prob)
                
                logger.debug(
                    f"Sentence {i+1} entailment: {entailment_prob:.3f} - {sentences[i][:50]}..."
                )
        
        except Exception as e:
            logger.error(f"NLI batch prediction error: {e}")
            # On error, return neutral scores for all sentences
            scores = [0.5] * len(sentences)
        
        # Check if average score meets threshold
        avg_score = sum(scores) / len(scores) if scores else 0
        passed = avg_score >= self.nli_threshold
        
        logger.info(
            f"Entailment check: {len(sentences)} sentences, "
            f"avg_score={avg_score:.3f}, threshold={self.nli_threshold}, passed={passed}"
        )
        
        return passed, scores
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with improved handling of abbreviations.
        
        Better handling of "Dr.", "Mr.", etc.
        """
        import re
        
        # Replace common abbreviations temporarily to avoid false splits
        text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
        text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<DOT>', text)
        text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
        text = re.sub(r'\bProf\.', 'Prof<DOT>', text)
        
        # Split on sentence terminators followed by space/uppercase/end
        # More conservative: requires space + capital letter or end of string
        sentences = re.split(r'[.!?](?:\s+(?=[A-Z가-힣])|$)', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter out empty strings and very short fragments
        sentences = [s for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _combine_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Combine context chunks into single text.
        
        IMPROVED: Increased limit to reduce false positives.
        DeBERTa v3 handles ~512 tokens (roughly 2000-3000 chars).
        """
        texts = []
        
        for chunk in context_chunks:
            # Try parent_text first, fall back to child_text
            text = chunk.get('parent_text') or chunk.get('child_text', '')
            if text:
                texts.append(text)
        
        # Combine with newlines
        combined = '\n\n'.join(texts)
        
        # Truncate if too long
        # DeBERTa v3 small: 512 tokens ≈ 2000-3000 chars
        max_chars = 3000
        if len(combined) > max_chars:
            combined = combined[:max_chars] + '...'
            logger.debug(f"Context truncated to {max_chars} chars for NLI")
        
        return combined
    
    def _softmax(self, logits) -> list:
        """Apply softmax to logits"""
        import numpy as np
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for stability
        return exp_logits / exp_logits.sum()
    
    def _calculate_confidence(
        self,
        issues: List[str],
        nli_scores: Optional[List[float]]
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            issues: List of issues found
            nli_scores: NLI entailment scores
            
        Returns:
            Confidence score 0-1
        """
        # Start with 1.0 if no issues
        if not issues:
            confidence = 1.0
        else:
            # Decrease confidence based on number of issues
            confidence = max(0.0, 1.0 - (len(issues) * 0.3))
        
        # Adjust based on NLI scores if available
        if nli_scores:
            avg_nli = sum(nli_scores) / len(nli_scores)
            confidence = (confidence + avg_nli) / 2  # Average of both
        
        return confidence


def create_nli_safety_checker(config) -> Optional[NLISafetyChecker]:
    """
    Create NLI safety checker from config.
    
    Args:
        config: SafetyCheckConfig
        
    Returns:
        NLISafetyChecker or None if disabled
    """
    if not config.enabled:
        logger.info("Safety check disabled")
        return None
    
    return NLISafetyChecker(
        nli_model_name=config.nli_model,
        nli_threshold=config.nli_threshold,
        use_nli=config.use_nli,
        check_citations=config.check_citation_presence,
        min_answer_length=config.min_answer_length
    )