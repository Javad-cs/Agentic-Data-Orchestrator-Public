"""
Router logger with async I/O and flexible type handling.

Fixes:
1. Handles both object attributes and dict keys (future-proof)
2. Async file I/O to avoid blocking event loop
3. Graceful error handling for malformed results
"""
import json
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union


class RouterLogger:
    """
    Async JSON logger for router observability.
    
    Features:
    - Non-blocking async file writes
    - Handles both objects (dataclass) and dicts
    - Graceful degradation on errors
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.log_dir / f"router_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    def _safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """
        Safely get value from object or dict.
        
        Handles:
        - Objects with attributes (dataclass, ToolResponse)
        - Dictionaries
        - None/missing values
        
        Args:
            obj: Object or dict to extract from
            key: Attribute/key name
            default: Default value if not found
            
        Returns:
            Value or default
        """
        if obj is None:
            return default
        
        # Try object attribute (dataclass, NamedTuple, etc.)
        if hasattr(obj, key):
            return getattr(obj, key, default)
        
        # Try dictionary key
        if isinstance(obj, dict):
            return obj.get(key, default)
        
        # Fallback
        return default
    
    async def log_routing_decision(
        self,
        query: str,
        scores: Dict[str, float],
        selected_path: str,
        reasoning: str
    ):
        """Log routing decision (async)"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "routing_decision",
            "query": query,
            "scores": scores,
            "selected_path": selected_path,
            "reasoning": reasoning
        }
        await self._write_async(entry)
    
    async def log_result(
        self,
        query: str,
        path: str,
        result: Union[Any, Dict[str, Any]]
    ):
        """
        Log execution result (async).
        
        Handles both ToolResponse objects and dicts.
        """
        # Extract fields safely (works for objects and dicts)
        success = self._safe_get(result, 'success', True)
        answer = self._safe_get(result, 'answer', '')
        error = self._safe_get(result, 'error', None)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "result",
            "query": query,
            "path": path,
            "success": success,
            "answer_length": len(answer) if answer else 0,
            "has_error": error is not None
        }
        await self._write_async(entry)
    
    async def log_error(self, message: str, context: Dict[str, Any] = None):
        """Log error (async)"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "message": message,
            "context": context or {}
        }
        await self._write_async(entry)
    
    async def log_tool_call(
        self,
        tool_name: str,
        query: str,
        success: bool,
        latency_ms: float
    ):
        """
        Log individual tool call (for Slow Lane observability).
        
        Useful for debugging multi-step queries.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_call",
            "tool_name": tool_name,
            "query": query,
            "success": success,
            "latency_ms": latency_ms
        }
        await self._write_async(entry)
    
    async def _write_async(self, entry: Dict[str, Any]):
        """
        Write log entry asynchronously (non-blocking).
        
        Uses aiofiles to avoid blocking the event loop.
        Falls back to sync write if aiofiles unavailable.
        """
        try:
            # Async file write (non-blocking)
            async with aiofiles.open(self.log_file, 'a') as f:
                await f.write(json.dumps(entry) + '\n')
        except Exception as e:
            # Fallback to sync write (better than losing logs)
            print(f"️  Async log write failed, using sync fallback: {e}")
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def log_sync(self, entry: Dict[str, Any]):
        """
        Synchronous log write (for non-async contexts).
        
        Use sparingly - prefer async methods when possible.
        """
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')