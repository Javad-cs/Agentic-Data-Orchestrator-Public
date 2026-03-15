"""Wrapper for Fast Lane RAG agent"""
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Calculate paths
_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[3]
_rag_agent_path = str(_project_root / "rag_agent")

def _import_rag_agent_classes():
    """Import rag_agent classes safely by temporarily manipulating sys.path."""
    for key in list(sys.modules.keys()):
        if key == 'src' or key.startswith('src.'):
            del sys.modules[key]
    
    sys.path.insert(0, _rag_agent_path)
    
    from src.agents.fast_lane import FastLane
    from src.config.models import SystemConfig
    return FastLane, SystemConfig

FastLane, SystemConfig = _import_rag_agent_classes()

from .base import BaseTool, ToolResponse


class RAGTool(BaseTool):
    """Fast Lane RAG tool with lazy async initialization"""
    
    def __init__(self):
        try:
            self.config = SystemConfig()
            self.agent = None
            self._initialized = False
            logger.info("RAG tool config loaded")
        except Exception as e:
            logger.error(f"Failed to load RAG config: {e}", exc_info=True)
            raise
    
    async def _ensure_initialized(self):
        if not self._initialized:
            try:
                logger.info("Initializing Fast Lane agent...")
                self.agent = FastLane(config=self.config)
                await self.agent.initialize()
                self._initialized = True
                logger.info("Fast Lane agent initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Fast Lane: {e}", exc_info=True)
                raise
    
    def name(self) -> str:
        return "fast_lane_agent"
    
    def description(self) -> str:
        return """Search documents for qualitative information.
        Use when query needs: definitions, policies, context, descriptions.
        Input: Natural language question.
        Output: Answer from documents with citations."""
    
    async def run(
        self, 
        query: str, 
        top_k: int = 5, 
        language: str = "ko"
    ) -> ToolResponse:
        """
        Run RAG agent with lazy initialization.
        
        Args:
            query: Natural language question
            top_k: Number of chunks to retrieve (default: 5)
            language: Query language (default: "ko")
        """
        logger.debug(f"RAG tool processing query: {query[:100]}... (top_k={top_k}, lang={language})")
        
        await self._ensure_initialized()
        
        try:
            # invoke_tool returns:
            # {
            #     "answer": str,
            #     "context_chunks": List[Dict],
            #     "citations_used": List[int],
            #     "success": bool,
            #     "error": Optional[str]
            # }
            result = await self.agent.invoke_tool(
                query=query,
                top_k=top_k,
                language=language
            )
            
            # Check success field
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown RAG error")
                logger.warning(f"RAG tool failed: {error_msg}")
                return ToolResponse(
                    answer=result.get("answer", ""),  # May have partial answer
                    success=False,
                    error=error_msg,
                    metadata={
                        "citations": result.get("citations_used", []),
                        "chunks_retrieved": len(result.get("context_chunks", [])),
                        "context_chunks": result.get("context_chunks", [])
                    }
                )
            
            answer = result.get("answer", "")
            context_chunks = result.get("context_chunks", [])
            citations_used = result.get("citations_used", [])
            
            logger.info(f"RAG tool succeeded: {len(answer)} chars, {len(citations_used)} citations")
            
            return ToolResponse(
                answer=answer,
                success=True,
                metadata={
                    "citations": citations_used,
                    "chunks_retrieved": len(context_chunks),
                    "context_chunks": context_chunks
                }
            )
            
        except Exception as e:
            logger.error(f"RAG tool exception: {e}", exc_info=True)
            return ToolResponse(
                answer="",
                success=False,
                error=f"RAG tool error: {str(e)}"
            )