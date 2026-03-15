import logging
import asyncio
import time
from typing import List, Optional
from dataclasses import dataclass

from src.generation.base import BaseGenerator

logger = logging.getLogger(__name__)


@dataclass
class QueryExpansionResult:
    """Result from query expansion"""
    original_query: str
    expanded_queries: List[str]
    success: bool
    latency_ms: int
    metadata: dict
    
    @property
    def all_queries(self) -> List[str]:
        """Get all queries (original + expanded)"""
        return [self.original_query] + self.expanded_queries
    
    @property
    def total_queries(self) -> int:
        """Total number of queries"""
        return len(self.all_queries)


class QueryExpander:
    """
    Query expansion using LLM.
    
    Generates semantically similar query variants to improve retrieval recall.
    
    Example:
        Original: "스테인레스강 가공"
        Expanded:
        - "스테인레스 철강 절삭 가공"
        - "SUS 재료 CNC 가공"
        - "오스테나이트강 밀링 가공"
    
    Features:
    - Configurable number of variants
    - Parallel or sequential generation
    - Fallback to original query on failure
    - Detailed metrics and logging
    """
    
    def __init__(
        self,
        llm_client: BaseGenerator,
        num_variants: int = 3,
        temperature: float = 0.7,
        parallel: bool = True,
        timeout_seconds: int = 10
    ):
        """
        Initialize query expander.
        
        Args:
            llm_client: LLM client for generation
            num_variants: Number of query variants to generate
            temperature: LLM temperature (higher = more diverse)
            parallel: Generate variants in parallel (faster) vs sequential
            timeout_seconds: Timeout for expansion operation
        """
        self.llm_client = llm_client
        self.num_variants = num_variants
        self.temperature = temperature
        self.parallel = parallel
        self.timeout_seconds = timeout_seconds
        
        logger.info(
            f"QueryExpander initialized "
            f"(variants={num_variants}, temp={temperature}, parallel={parallel})"
        )
    
    async def expand(
        self,
        query: str,
        language: str = "ko"
    ) -> QueryExpansionResult:
        """
        Expand query into multiple variants.
        
        Args:
            query: Original query
            language: Query language ("ko" or "en")
            
        Returns:
            QueryExpansionResult with expanded queries
        """
        start_time = time.time()
        
        logger.info(f"Expanding query: '{query}' (language={language})")
        
        try:
            # Generate variants
            if self.parallel:
                expanded = await self._expand_parallel(query, language)
            else:
                expanded = await self._expand_sequential(query, language)
            
            # Filter out empty/invalid variants
            expanded = [q.strip() for q in expanded if q and q.strip()]
            
            # Remove duplicates (case-insensitive)
            seen = set()
            unique_expanded = []
            for q in expanded:
                q_lower = q.lower()
                if q_lower not in seen and q_lower != query.lower():
                    seen.add(q_lower)
                    unique_expanded.append(q)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Query expansion complete: {len(unique_expanded)} variants "
                f"in {latency_ms}ms"
            )
            
            for i, variant in enumerate(unique_expanded, 1):
                logger.debug(f"  Variant {i}: {variant}")
            
            return QueryExpansionResult(
                original_query=query,
                expanded_queries=unique_expanded,
                success=True,
                latency_ms=latency_ms,
                metadata={
                    "language": language,
                    "parallel": self.parallel,
                    "temperature": self.temperature,
                    "requested_variants": self.num_variants,
                    "generated_variants": len(unique_expanded)
                }
            )
        
        except asyncio.TimeoutError:
            logger.error(f"Query expansion timeout after {self.timeout_seconds}s")
            return self._fallback_result(query, "timeout")
        
        except Exception as e:
            logger.error(f"Query expansion error: {e}", exc_info=True)
            return self._fallback_result(query, str(e))
    
    async def _expand_parallel(
        self,
        query: str,
        language: str
    ) -> List[str]:
        """Generate variants in parallel (faster)"""
        logger.debug(f"Generating {self.num_variants} variants in parallel")
        
        # Create multiple concurrent generation tasks
        tasks = [
            self._generate_single_variant(query, language, i)
            for i in range(self.num_variants)
        ]
        
        # Run with timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=self.timeout_seconds
        )
        
        # Filter out exceptions
        variants = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Variant {i+1} generation failed: {result}")
            elif result:
                variants.append(result)
        
        return variants
    
    async def _expand_sequential(
        self,
        query: str,
        language: str
    ) -> List[str]:
        """Generate variants sequentially (more controlled)"""
        logger.debug(f"Generating {self.num_variants} variants sequentially")
        
        variants = []
        for i in range(self.num_variants):
            try:
                variant = await asyncio.wait_for(
                    self._generate_single_variant(query, language, i),
                    timeout=self.timeout_seconds / self.num_variants
                )
                if variant:
                    variants.append(variant)
            except Exception as e:
                logger.warning(f"Variant {i+1} generation failed: {e}")
                continue
        
        return variants
    
    async def _generate_single_variant(
        self,
        query: str,
        language: str,
        variant_index: int
    ) -> str:
        """Generate a single query variant"""
        system_prompt = self._build_system_prompt(language)
        user_prompt = self._build_user_prompt(query, variant_index, language)
        
        logger.debug(f"Generating variant {variant_index + 1} for: {query}")
        
        # Generate with LLM
        result = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=100
        )
        
        # Log raw response for debugging
        logger.debug(f"Raw LLM response for variant {variant_index + 1}: {repr(result.content)}")
        
        # Extract query from response
        variant = result.content.strip()
        
        # Remove common prefixes/formatting
        # Sometimes LLM adds "Variant query 1:" or similar
        prefixes_to_remove = [
            f"Variant query {variant_index + 1}:",
            f"변형 쿼리 {variant_index + 1}:",
            "Query:",
            "쿼리:",
        ]
        
        for prefix in prefixes_to_remove:
            if variant.lower().startswith(prefix.lower()):
                variant = variant[len(prefix):].strip()
        
        # Remove quotes if present
        if variant.startswith('"') and variant.endswith('"'):
            variant = variant[1:-1]
        if variant.startswith("'") and variant.endswith("'"):
            variant = variant[1:-1]
        
        # Take only first line if multiline
        if '\n' in variant:
            variant = variant.split('\n')[0].strip()
        
        logger.debug(f"Cleaned variant {variant_index + 1}: {variant}")
        
        return variant
    
    def _build_system_prompt(self, language: str) -> str:
        """Build system prompt for query expansion"""
        if language == "ko":
            return """당신은 검색 쿼리 확장 전문가입니다.

주어진 검색 쿼리를 의미적으로 유사하지만 다르게 표현된 변형 쿼리로 변환하세요.

규칙:
1. 원래 의도를 유지하되 다른 단어/표현을 사용
2. 동의어, 관련 용어, 기술 용어 활용
3. 하나의 변형 쿼리만 생성 (따옴표 없이)
4. 간결하게 유지 (10단어 이하)"""
        else:
            return """You are a search query expansion expert.

Transform the given search query into a semantically similar but differently expressed variant.

Rules:
1. Maintain original intent but use different words/expressions
2. Use synonyms, related terms, technical terminology
3. Generate only ONE variant query (no quotes)
4. Keep it concise (under 10 words)"""
    
    def _build_user_prompt(
        self,
        query: str,
        variant_index: int,
        language: str
    ) -> str:
        """Build user prompt for query expansion"""
        if language == "ko":
            prompt = f"""원본 쿼리: {query}

변형 쿼리 {variant_index + 1}:"""
        else:
            prompt = f"""Original query: {query}

Variant query {variant_index + 1}:"""
        
        return prompt
    
    def _fallback_result(
        self,
        query: str,
        error: str
    ) -> QueryExpansionResult:
        """Create fallback result on error"""
        logger.warning(f"Using fallback (original query only): {error}")
        
        return QueryExpansionResult(
            original_query=query,
            expanded_queries=[],
            success=False,
            latency_ms=0,
            metadata={
                "fallback": True,
                "error": error
            }
        )


def create_query_expander(
    llm_client: BaseGenerator,
    config
) -> Optional[QueryExpander]:
    """
    Create query expander from config.
    
    Args:
        llm_client: LLM client
        config: QueryExpansionConfig
        
    Returns:
        QueryExpander instance or None if disabled
    """
    if not config.enabled:
        logger.info("Query expansion disabled")
        return None
    
    return QueryExpander(
        llm_client=llm_client,
        num_variants=config.num_variants,
        temperature=config.temperature,
        parallel=config.parallel
    )