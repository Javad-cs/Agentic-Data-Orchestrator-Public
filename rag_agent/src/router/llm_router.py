import logging
import re
from typing import Literal, Optional
from dataclasses import dataclass

from src.generation.base import BaseGenerator

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Router decision result"""
    lane: Literal["fast", "slow"]
    confidence: float  # 0-1
    reasoning: str
    
    @property
    def is_fast_lane(self) -> bool:
        return self.lane == "fast"
    
    @property
    def is_slow_lane(self) -> bool:
        return self.lane == "slow"


class LLMRouter:
    """
    LLM-based router for Fast vs Slow lane decision.
    
    Uses LLM to analyze query complexity and route appropriately:
    - Fast Lane: Simple factual queries, single-step answers
    - Slow Lane: Complex reasoning, multi-step analysis, comparisons
    
    Design:
    - Single LLM call for decision (~200-500ms)
    - Structured output parsing (regex-based, robust)
    - Confidence scoring
    - Fallback to Fast Lane on errors
    
    TODO: Future improvement with JSON mode
    If BaseGenerator supports JSON mode (response_format="json_object"),
    upgrade to structured JSON output for 99% parsing reliability:
        {
          "lane": "fast",
          "confidence": 0.9,
          "reasoning": "Simple factual query"
        }
    """
    
    def __init__(
        self,
        llm_client: BaseGenerator,
        default_lane: Literal["fast", "slow"] = "fast",
        temperature: float = 0.3
    ):
        """
        Initialize LLM router.
        
        Args:
            llm_client: LLM client for routing decisions
            default_lane: Fallback lane on errors
            temperature: LLM temperature (lower = more consistent)
        """
        self.llm_client = llm_client
        self.default_lane = default_lane
        self.temperature = temperature
        
        logger.info(f"LLMRouter initialized (default={default_lane}, temp={temperature})")
    
    async def route(
        self,
        query: str,
        language: str = "ko"
    ) -> RoutingDecision:
        """
        Route query to Fast or Slow lane.
        
        Args:
            query: User query
            language: Query language
            
        Returns:
            RoutingDecision with lane, confidence, and reasoning
        """
        logger.debug(f"Routing query: {query[:50]}...")
        
        try:
            # Get routing decision from LLM
            system_prompt = self._build_system_prompt(language)
            user_prompt = self._build_user_prompt(query, language)
            
            result = await self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=200
            )
            
            # Parse decision
            decision = self._parse_decision(result.content)
            
            logger.info(
                f"Routed to {decision.lane.upper()} lane "
                f"(confidence={decision.confidence:.2f}): {decision.reasoning}"
            )
            
            return decision
        
        except Exception as e:
            logger.error(f"Routing error: {e}, using default lane={self.default_lane}")
            return RoutingDecision(
                lane=self.default_lane,
                confidence=0.5,
                reasoning=f"Fallback due to error: {str(e)}"
            )
    
    def _build_system_prompt(self, language: str) -> str:
        """Build system prompt for routing"""
        if language == "ko":
            return """당신은 쿼리 라우팅 전문가입니다.

사용자 질문을 분석하여 Fast Lane 또는 Slow Lane으로 라우팅하세요.

**Fast Lane** - 간단한 질문 (목표: <4초):
- 단순 사실 확인
- 정의 또는 설명 요청
- 단일 개념 질문
- 명확한 답이 있는 질문
예시: "PVD 코팅이란?", "스테인레스강 가공 조건은?", "AlTiN 코팅 특징은?"

**Slow Lane** - 복잡한 질문 (목표: 정확성):
- 다단계 추론 필요
- 비교/분석 요청
- 여러 개념 통합
- "왜", "어떻게" 등 심층 분석
- 의사결정 지원
예시: "X와 Y 중 어떤 것이 더 나은가?", "어떻게 최적화할 수 있는가?", "왜 이런 결과가 나오는가?"

응답 형식 (반드시 준수):
LANE: fast 또는 slow
CONFIDENCE: 0.0-1.0 (소수점)
REASONING: 한 문장으로 이유 설명

중요: 마크다운, 코드 블록, 또는 추가 설명 없이 위 형식만 사용하세요."""
        else:
            return """You are a query routing expert.

Analyze user queries and route to Fast Lane or Slow Lane.

**Fast Lane** - Simple queries (target: <4s):
- Simple fact-checking
- Definitions or explanations
- Single concept questions
- Questions with clear answers
Examples: "What is PVD coating?", "Stainless steel machining conditions?", "AlTiN coating features?"

**Slow Lane** - Complex queries (target: accuracy):
- Multi-step reasoning required
- Comparison/analysis requests
- Integration of multiple concepts
- "Why", "how" questions requiring deep analysis
- Decision support
Examples: "Which is better X or Y?", "How to optimize?", "Why does this result occur?"

Response format (must follow):
LANE: fast or slow
CONFIDENCE: 0.0-1.0 (decimal)
REASONING: One sentence explanation

CRITICAL: Use ONLY the format above. No markdown, code blocks, or extra text."""
    
    def _build_user_prompt(self, query: str, language: str) -> str:
        """Build user prompt for routing"""
        if language == "ko":
            return f"""쿼리: {query}

위 쿼리를 분석하여 Fast Lane 또는 Slow Lane으로 라우팅하세요."""
        else:
            return f"""Query: {query}

Analyze the query above and route to Fast Lane or Slow Lane."""
    
    def _parse_decision(self, response: str) -> RoutingDecision:
        """
        Parse LLM response into RoutingDecision using regex.
        
        Expected format:
        LANE: fast
        CONFIDENCE: 0.9
        REASONING: Simple factual query
        
        Uses regex to handle:
        - Case insensitivity
        - Extra whitespace
        - Markdown formatting
        - Line breaks and noise
        """
        # Default values
        lane = self.default_lane
        confidence = 0.5
        reasoning = "Unable to parse LLM response"
        
        # Clean response (remove markdown code blocks)
        clean_response = re.sub(r'```[a-z]*\s*|\s*```', '', response, flags=re.IGNORECASE)
        
        # Extract LANE (case-insensitive, flexible spacing)
        lane_match = re.search(
            r'LANE\s*[:：]\s*(fast|slow)',
            clean_response,
            re.IGNORECASE
        )
        if lane_match:
            lane = lane_match.group(1).lower()  # type: ignore
        else:
            logger.warning(f"Could not parse LANE from: {response[:100]}")
        
        # Extract CONFIDENCE (handles various number formats)
        conf_match = re.search(
            r'CONFIDENCE\s*[:：]\s*([0-9]*\.?[0-9]+)',
            clean_response,
            re.IGNORECASE
        )
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                logger.warning(f"Could not parse CONFIDENCE: {conf_match.group(1)}")
        
        # Extract REASONING (everything after "REASONING:" until end)
        reasoning_match = re.search(
            r'REASONING\s*[:：]\s*(.+?)$',  # Capture to end of string
            clean_response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Clean up: collapse multiple newlines into single space
            reasoning = re.sub(r'[\n\r]+', ' ', reasoning).strip()
            # Remove trailing "Hope this helps!" type noise
            reasoning = re.sub(r'\s*(hope this helps|let me know).*$', '', reasoning, flags=re.IGNORECASE).strip()
        
        return RoutingDecision(
            lane=lane,
            confidence=confidence,
            reasoning=reasoning
        )


def create_router(
    llm_client: BaseGenerator,
    default_lane: Literal["fast", "slow"] = "fast",
    temperature: float = 0.3
) -> LLMRouter:
    """
    Create LLM router.
    
    Args:
        llm_client: LLM client
        default_lane: Fallback lane
        temperature: LLM temperature
        
    Returns:
        LLMRouter instance
    """
    return LLMRouter(
        llm_client=llm_client,
        default_lane=default_lane,
        temperature=temperature
    )