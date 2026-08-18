from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result from LLM generation"""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int]
    metadata: Dict[str, Any]


@dataclass
class StreamChunk:
    """Single chunk from streaming generation"""
    content: str
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseGenerator(ABC):
    """
    Abstract base class for LLM generators.
    
    Supports both streaming and non-streaming generation.
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate completion (non-streaming).
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens (overrides config)
            model: Model name (overrides config)
            
        Returns:
            GenerationResult with full response
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate completion with streaming.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens (overrides config)
            model: Model name (overrides config)
            
        Yields:
            StreamChunk objects as they arrive
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the current model name"""
        pass
    
    @abstractmethod
    async def close(self):
        """Cleanup resources (close HTTP clients, etc.)"""
        pass