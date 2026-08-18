import logging
import asyncio
from typing import AsyncIterator, List, Dict, Any, Optional
from enum import Enum

from .base import BaseGenerator
from .citation_formatter import CitationFormatter
from .safety_check import NLISafetyChecker as SafetyChecker

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """SSE event types"""
    STATUS = "status"
    CITATION = "citation"
    CHUNK = "chunk"
    CITATION_MARKER = "citation_marker"
    DONE = "done"
    ERROR = "error"


class StreamingGenerator:
    """
    Streaming generator with SSE events and citations.
    
    Generates structured events for frontend consumption:
    - status: Progress updates
    - citation: Source metadata
    - chunk: Answer text chunks
    - citation_marker: Citation markers in text
    - done: Completion with metadata
    - error: Error events
    
    Example event stream:
        {"type": "status", "content": "Retrieving context..."}
        {"type": "citation", "data": {"id": "[1]", "file": "sample.pdf", "page": 5}}
        {"type": "chunk", "content": "PVD 코팅이 "}
        {"type": "chunk", "content": "적합합니다 "}
        {"type": "citation_marker", "id": "[1]"}
        {"type": "done", "metadata": {"total_tokens": 150, "latency_ms": 850}}
    """
    
    def __init__(
        self,
        llm_client: BaseGenerator,
        safety_checker: Optional[SafetyChecker] = None
    ):
        """
        Initialize streaming generator.
        
        Args:
            llm_client: LLM client for generation
            safety_checker: Optional safety checker
        """
        self.llm_client = llm_client
        self.safety_checker = safety_checker
        
        logger.info("StreamingGenerator initialized")
    
    async def generate_with_citations(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str = "ko",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate answer with citations and stream events.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks with metadata
            language: Response language ("ko" or "en")
            temperature: LLM temperature (optional)
            max_tokens: Max tokens (optional)
            
        Yields:
            SSE event dicts with type and content/data
        """
        import time
        start_time = time.time()
        
        try:
            # Reset citation formatter
            citation_formatter = CitationFormatter()
            
            # Step 1: Send status - Adding citations
            yield self._create_event(EventType.STATUS, "sources를 준비 중...")
            
            # Step 2: Add sources and send citation events
            # CRITICAL: Track the actual citation numbers assigned by formatter
            chunk_citation_map = []  # List of (chunk, citation_num) tuples
            
            for chunk in context_chunks:
                citation_num = citation_formatter.add_source(chunk)
                chunk_citation_map.append((chunk, citation_num))
                
                # Send citation metadata to frontend
                yield self._create_event(
                    EventType.CITATION,
                    data={
                        "id": f"[{citation_num}]",
                        "file": chunk.get('source_file', 'Unknown'),
                        "page": chunk.get('page_number'),
                        "source_id": chunk.get('child_id'),
                        "parent_text": chunk.get('parent_text', '')[:200] + "..." if len(chunk.get('parent_text', '')) > 200 else chunk.get('parent_text', ''),  # Truncate for display
                        "child_text": chunk.get('child_text', '')[:100] + "..." if len(chunk.get('child_text', '')) > 100 else chunk.get('child_text', '')  # Truncate for display
                    }
                )
            
            # Step 3: Build prompt with ACTUAL citation numbers
            yield self._create_event(EventType.STATUS, "답변 생성 중...")
            
            system_prompt = self._build_system_prompt(language)
            user_prompt = self._build_user_prompt(query, chunk_citation_map, language)
            
            # Step 4: Stream generation
            full_answer = ""
            chunk_count = 0
            logger.debug(f"Starting LLM streaming - query length: {len(query)} chars")
            
            try:
                async for chunk in self.llm_client.stream_generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=None,  # Don't set temperature - Azure models reject it
                    max_tokens=max_tokens
                ):
                    if chunk.content:
                        full_answer += chunk.content
                        chunk_count += 1
                        yield self._create_event(EventType.CHUNK, chunk.content)
                    
                    if chunk.finish_reason:
                        logger.debug(f"Generation finished: {chunk.finish_reason}, chunks={chunk_count}, chars={len(full_answer)}")
                
                # Check if we got any content
                if chunk_count == 0:
                    logger.error(
                        f"LLM returned 0 chunks! "
                        f"Query length: {len(query)}, "
                        f"Prompt length: {len(user_prompt)}, "
                        f"Context chunks: {len(chunk_citation_map)}"
                    )
                    # Return error event for empty response
                    yield self._create_event(
                        EventType.ERROR,
                        data={
                            "message": "LLM returned empty response",
                            "type": "empty_response"
                        }
                    )
                    return
                else:
                    logger.info(f"LLM generation complete: {chunk_count} chunks, {len(full_answer)} chars")
            
            except Exception as e:
                logger.error(f"LLM streaming exception: {type(e).__name__}: {e}", exc_info=True)
                yield self._create_event(
                    EventType.ERROR,
                    data={  # Use 'data' consistently
                        "message": f"LLM generation failed: {str(e)}",
                        "type": "llm_error"
                    }
                )
                return
            
            # Step 5: Safety check (if enabled)
            if self.safety_checker:
                yield self._create_event(EventType.STATUS, "안전성 검사 중...")
                
                # Extract context chunks (not text) for safety checker
                context_chunks_for_safety = [chunk for chunk, _ in chunk_citation_map]
                
                # Extract citations from the answer
                import re
                citation_pattern = r'\[\d+\]'
                citation_matches = re.findall(citation_pattern, full_answer)
                citations_list = [{"number": int(m.strip('[]'))} for m in citation_matches] if citation_matches else None
                
                # Call with correct signature and AWAIT the coroutine
                # Signature: check(answer, context_chunks, citations)
                safety_result = await self.safety_checker.check(
                    answer=full_answer,
                    context_chunks=context_chunks_for_safety,
                    citations=citations_list
                )
                
                if not safety_result.passed:
                    logger.warning(f"Safety check failed: {safety_result.issues}")
                    yield self._create_event(
                        EventType.ERROR,
                        data={
                            "message": "답변이 안전성 검사를 통과하지 못했습니다.",
                            "issues": safety_result.issues,  # Use 'issues' not 'reason'
                            "confidence": safety_result.confidence,
                            "type": "safety_check_failed"
                        }
                    )
                    return
            
            # Step 6: Send completion event
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            yield self._create_event(
                EventType.DONE,
                metadata={
                    "latency_ms": latency_ms,
                    "answer_length": len(full_answer),
                    "citation_count": len(citation_formatter.sources),
                    "safety_passed": True if not self.safety_checker else safety_result.passed
                }
            )
        
        except Exception as e:
            logger.error(f"Streaming generation error: {e}", exc_info=True)
            yield self._create_event(
                EventType.ERROR,
                data={"message": str(e), "type": type(e).__name__}
            )
    
    def _build_system_prompt(self, language: str) -> str:
        """
        Build system prompt for LLM.
        
        Args:
            language: Response language
            
        Returns:
            System prompt string
        """
        if language == "ko":
            return """당신은 전문적인 기술 문서 Q&A 어시스턴트입니다.

제공된 컨텍스트를 바탕으로 정확하고 간결하게 답변하세요.

중요한 규칙:
1. 답변에 반드시 출처 번호를 표시하세요 (예: [1], [2])
2. 컨텍스트에 없는 정보는 추측하지 마세요
3. 확실하지 않으면 "제공된 정보로는 확인할 수 없습니다"라고 말하세요
4. 전문 용어를 정확하게 사용하세요
5. 답변할 수 없는 경우에도, 확인한 출처를 반드시 인용하세요 (예: [1][2])

기억하세요: 모든 사실적 주장에는 반드시 출처 [1], [2] 등이 포함되어야 합니다."""
        else:
            return """You are a professional technical documentation Q&A assistant.

Provide accurate and concise answers based on the provided context.

CRITICAL RULES:
1. ALWAYS cite sources using [1], [2] notation - this is MANDATORY
2. Do not make up information not in the context
3. If uncertain, say "Cannot be determined from the provided information"
4. Use technical terms accurately
5. Even if you cannot answer, cite the sources you checked: [1][2]

Remember: Every factual claim MUST have a citation [1], [2], etc."""
    
    def _build_user_prompt(
        self,
        query: str,
        chunk_citation_map: List[tuple],  # List of (chunk, citation_num) tuples
        language: str
    ) -> str:
        """
        Build user prompt with context and query using ACTUAL citation numbers.
        
        Args:
            query: User query
            chunk_citation_map: List of (chunk_dict, citation_num) tuples
            language: Response language
            
        Returns:
            User prompt string
        """
        # Build context section with ACTUAL citation numbers
        context_parts = []
        for chunk, citation_num in chunk_citation_map:
            parent_text = chunk.get('parent_text', chunk.get('child_text', ''))
            # Use the actual citation number assigned by formatter (not enumerate index)
            context_parts.append(f"[{citation_num}] {parent_text}")
        
        context_text = "\n\n".join(context_parts)
        
        # Build prompt
        if language == "ko":
            prompt = f"""다음 컨텍스트를 바탕으로 질문에 답변하세요.

컨텍스트:
{context_text}

질문: {query}

답변 (출처 번호 [1], [2] 등을 반드시 표시):"""
        else:
            prompt = f"""Answer the question based on the following context.

Context:
{context_text}

Question: {query}

Answer (MUST cite sources using [1], [2], [3] etc. for every fact):"""
        
        return prompt
    
    def _create_event(
        self,
        event_type: EventType,
        content: Any = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create SSE event dict.
        
        Args:
            event_type: Type of event
            content: Content string (for status, chunk)
            data: Data dict (for citation, error)
            metadata: Metadata dict (for done)
            
        Returns:
            Event dict ready for JSON serialization
        """
        event = {"type": event_type.value}
        
        if content is not None:
            event["content"] = content
        
        if data is not None:
            event["data"] = data
        
        if metadata is not None:
            event["metadata"] = metadata
        
        return event


# Convenience function
def create_streaming_generator(
    llm_client: BaseGenerator,
    safety_checker: Optional[SafetyChecker] = None
) -> StreamingGenerator:
    """
    Create streaming generator with all components.
    
    Args:
        llm_client: LLM client
        safety_checker: Optional safety checker
        
    Returns:
        StreamingGenerator instance
    """
    return StreamingGenerator(
        llm_client=llm_client,
        safety_checker=safety_checker
    )