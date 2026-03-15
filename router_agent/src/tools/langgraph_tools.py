"""
LangGraph tool wrappers for Slow Lane integration.

Converts RAGTool and SQLTool to async functions compatible with LangGraph.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LangGraphToolWrapper:
    """Wrapper to make router tools work with LangGraph"""
    
    def __init__(self, rag_tool=None, sql_tool=None):
        """
        Initialize with tool instances.
        
        Args:
            rag_tool: RAGTool instance
            sql_tool: SQLTool instance
        """
        if not rag_tool and not sql_tool:
            raise ValueError("Must provide at least one tool (rag_tool or sql_tool)")
        
        self.rag_tool = rag_tool
        self.sql_tool = sql_tool
        
        logger.info(f"LangGraph tools initialized: "
                   f"RAG={'[SUCCESS]' if rag_tool else '[FAILED]'}, "
                   f"SQL={'[SUCCESS]' if sql_tool else '[FAILED]'}")
    
    async def rag_search(
        self, 
        query: str, 
        top_k: int = 5, 
        language: str = "ko"
    ) -> Dict[str, Any]:
        """
        Search documents using RAG tool.
        
        Args:
            query: Natural language question
            top_k: Number of chunks to retrieve
            language: Query language
            
        Returns:
            Dict with success, answer, and metadata
        """
        if not self.rag_tool:
            return {
                "success": False,
                "answer": "",
                "error": "RAG tool not available",
                "context_chunks": [],
                "citations_used": []
            }
        
        try:
            result = await self.rag_tool.run(
                query=query,
                top_k=top_k,
                language=language
            )
            
            return {
                "success": result.success,
                "answer": result.answer,
                "error": result.error,
                "context_chunks": result.metadata.get("context_chunks", []),
                "citations_used": result.metadata.get("citations", [])
            }
        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            return {
                "success": False,
                "answer": "",
                "error": str(e),
                "context_chunks": [],
                "citations_used": []
            }
    
    async def sql_query(self, query: str) -> Dict[str, Any]:
        """
        Query database using SQL tool.
        
        Args:
            query: Natural language question
            
        Returns:
            Dict with success, answer, sql_query, and metadata
        """
        if not self.sql_tool:
            return {
                "success": False,
                "answer": "",
                "error": "SQL tool not available",
                "sql_query": "",
                "context_chunks": [],
                "citations_used": []
            }
        
        try:
            result = await self.sql_tool.run(query)
            
            return {
                "success": result.success,
                "answer": result.answer,
                "error": result.error,
                "sql_query": result.metadata.get("sql_query", ""),
                "rows_returned": result.metadata.get("rows_returned", 0),
                "context_chunks": [],  # SQL doesn't have document chunks
                "citations_used": []   # SQL doesn't have citations
            }
        except Exception as e:
            logger.error(f"SQL query failed: {e}", exc_info=True)
            return {
                "success": False,
                "answer": "",
                "error": str(e),
                "sql_query": "",
                "context_chunks": [],
                "citations_used": []
            }
    
    async def get_table_context(self, query: str, max_tables: int = 5) -> str:
        """
        Get table context for Slow Lane planner.
        
        Provides SQL table summaries to help planner make decisions.
        
        Args:
            query: User query
            max_tables: Maximum tables to include
            
        Returns:
            Formatted string with table summaries, or empty if SQL not available
        """
        if not self.sql_tool:
            return ""
        
        try:
            return await self.sql_tool.get_table_context(query, max_tables)
        except Exception as e:
            logger.error(f"Failed to get table context: {e}", exc_info=True)
            return ""
    
    def get_available_tools(self) -> list:
        """Get list of available tool names"""
        tools = []
        if self.rag_tool:
            tools.append("rag_tool")
        if self.sql_tool:
            tools.append("sql_tool")
        return tools
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available tools for planner"""
        descriptions = {}
        
        if self.rag_tool:
            descriptions["rag_tool"] = self.rag_tool.description()
        
        if self.sql_tool:
            descriptions["sql_tool"] = self.sql_tool.description()
        
        return descriptions