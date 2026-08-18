"""Base tool interface - async with proper defaults"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ToolResponse:
    """Standard response from any tool"""
    answer: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

class BaseTool(ABC):
    """Base class for all tools"""
    
    @abstractmethod
    def name(self) -> str:
        """Tool identifier"""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """What this tool does"""
        pass
    
    @abstractmethod
    async def run(self, query: str) -> ToolResponse:
        """Execute tool"""
        pass