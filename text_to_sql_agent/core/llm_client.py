"""
Abstract LLM client interface and Azure OpenAI implementation.
Designed to be modular - easy to swap providers.
Includes robust error handling and 'parameter learning' to avoid repeated latency penalties.
Enhanced with intelligent rate limit handling.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass
import logging
import time
import re
from openai import AsyncAzureOpenAI, AzureOpenAI
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class LLMMessage:
    """Represents a message in the conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs 
    ) -> LLMResponse:
        """Synchronous generation."""
        pass
    
    @abstractmethod
    async def generate_async(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs 
    ) -> LLMResponse:
        """Asynchronous generation."""
        pass


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI implementation of LLM client."""
    
    PROTECTED_KEYS = {"model", "messages"}
    DISALLOWED_KEYS = {"stream", "n", "response_format"} 
    
    def __init__(
        self,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        self.model = model or settings.default_model
        self.endpoint = endpoint or settings.azure_openai_endpoint
        self.api_key = api_key or settings.azure_openai_key
        self.api_version = api_version or settings.azure_openai_api_version
        
        # MEMORY: Store params that this model rejected so we don't try them again
        self.unsupported_params: Set[str] = set()
        
        # Check if model is likely O1 (reasoning) which typically rejects temperature
        if self.model.lower().startswith("o1") or "reasoning" in self.model.lower():
            logger.info(f"Model '{self.model}' detected as Reasoning model. Disabling temperature.")
            self.unsupported_params.add("temperature")

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        
        self.async_client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
    
    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _prepare_base_params(self, messages: List[LLMMessage], temperature: Optional[float], **kwargs) -> Dict[str, Any]:
        formatted_messages = self._format_messages(messages)
        
        params = {
            "model": self.model,
            "messages": formatted_messages,
        }
        
        # Only add temperature if it hasn't failed before
        if temperature is not None:
            if "temperature" not in self.unsupported_params:
                params["temperature"] = temperature
            
        for k, v in kwargs.items():
            if k in self.PROTECTED_KEYS:
                logger.warning(f"Ignoring attempt to override protected parameter '{k}'")
                continue
            if k in self.DISALLOWED_KEYS:
                raise ValueError(f"Parameter '{k}' is not currently supported by this client.")
            
            # Skip known bad params
            if k in self.unsupported_params:
                continue
                
            params[k] = v
            
        return params

    def _parse_response(self, response: Any) -> LLMResponse:
        if not getattr(response, "choices", None):
            raise RuntimeError(f"LLM response invalid: missing 'choices'. Response: {response}")

        usage = getattr(response, "usage", None)
        tokens = usage.total_tokens if usage else None
        
        content = response.choices[0].message.content or ""
        
        return LLMResponse(
            content=content,
            model=response.model,
            tokens_used=tokens,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response,
        )

    def _is_unsupported_param_error(self, error_msg: str, param_name: str) -> bool:
        error_msg = error_msg.lower()
        param_name = param_name.lower()
        triggers = [
            "unsupported parameter", 
            "unsupported value", 
            "does not support", 
            "unexpected keyword", 
            "unknown parameter",
            "invalid_request_error"
        ]
        if any(t in error_msg for t in triggers) and param_name in error_msg:
            return True
        return False

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """Check if exception is a rate limit (429) error."""
        if hasattr(exception, 'status_code') and exception.status_code == 429:
            return True
        error_str = str(exception).lower()
        return '429' in error_str or 'too many requests' in error_str or 'rate limit' in error_str

    def _extract_retry_after(self, exception: Exception) -> int:
        """
        Extract Retry-After value from rate limit exception.
        
        Tries multiple methods to find the wait time:
        1. Direct attribute on exception
        2. Response headers
        3. Parse from error message
        4. Fallback to 30 seconds
        """
        # Try direct attribute (OpenAI SDK sometimes includes this)
        if hasattr(exception, 'retry_after'):
            try:
                return int(exception.retry_after)
            except (ValueError, TypeError):
                pass
        
        # Try response headers if available
        if hasattr(exception, 'response') and hasattr(exception.response, 'headers'):
            retry_after = exception.response.headers.get('Retry-After') or \
                         exception.response.headers.get('retry-after')
            if retry_after:
                try:
                    return int(retry_after)
                except (ValueError, TypeError):
                    pass
        
        # Parse from error message as fallback
        error_str = str(exception)
        match = re.search(r'retry.*?(\d+)\s*second', error_str, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass
        
        # Default fallback
        return 30

    def _call_with_retry(self, call_func, **call_params):
        """
        Execute API call with rate limit retry logic.
        
        Handles 429 rate limit errors by:
        1. Detecting rate limit error
        2. Extracting exact wait time from API response
        3. Waiting the specified duration
        4. Retrying once
        
        Args:
            call_func: Function to call (self.client.chat.completions.create or async variant)
            **call_params: Parameters to pass to the API call
            
        Returns:
            API response
            
        Raises:
            Exception: If call fails after retry or for non-rate-limit errors
        """
        try:
            return call_func(**call_params)
        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._extract_retry_after(e)
                logger.warning(f"Rate limit hit. Waiting {retry_after}s as instructed by API before retry.")
                time.sleep(retry_after)
                
                # Retry once with same parameters
                try:
                    return call_func(**call_params)
                except Exception as retry_error:
                    logger.error(f"Rate limit retry failed: {retry_error}")
                    raise retry_error
            raise

    async def _call_with_retry_async(self, call_func, **call_params):
        """
        Async version of _call_with_retry.
        
        Same logic as sync version but uses asyncio.sleep for proper async behavior.
        """
        import asyncio
        
        try:
            return await call_func(**call_params)
        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._extract_retry_after(e)
                logger.warning(f"Rate limit hit. Waiting {retry_after}s as instructed by API before retry.")
                await asyncio.sleep(retry_after)
                
                try:
                    return await call_func(**call_params)
                except Exception as retry_error:
                    logger.error(f"Rate limit retry failed: {retry_error}")
                    raise retry_error
            raise

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs 
    ) -> LLMResponse:
        """Synchronous generation with persistent parameter learning and rate limit handling."""
        base_params = self._prepare_base_params(messages, temperature, **kwargs)
        max_tok = max_tokens or 2000
        
        token_strategies = [
            ("max_completion_tokens", max_tok),
            ("max_tokens", max_tok)
        ]
        
        last_error = None

        for token_key, token_val in token_strategies:
            call_params = base_params.copy()
            call_params[token_key] = token_val
            
            for attempt in range(3):
                try:
                    # Use rate limit wrapper
                    response = self._call_with_retry(
                        self.client.chat.completions.create,
                        **call_params
                    )
                    return self._parse_response(response)
                
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    
                    # Skip rate limit errors - already handled in _call_with_retry
                    if self._is_rate_limit_error(e):
                        raise
                    
                    if token_key in error_msg and ("unsupported" in error_msg or "unknown" in error_msg):
                        break 
                    
                    stripped = False
                    candidates = ["temperature", "seed", "top_p", "presence_penalty", "frequency_penalty"]
                    
                    for bad_param in candidates:
                        if bad_param in call_params and self._is_unsupported_param_error(error_msg, bad_param):
                            logger.warning(f"Azure rejected '{bad_param}'. Permanently ignoring it for this client.")
                            
                            # 1. Remove from current call
                            call_params.pop(bad_param)
                            
                            # 2. Add to blocklist so we never send it again
                            self.unsupported_params.add(bad_param)
                            
                            stripped = True
                            break 
                    
                    if stripped:
                        continue 
                    
                    raise e
        
        if last_error:
            raise last_error
        
        raise RuntimeError("LLM generation failed silently.")

    async def generate_async(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Asynchronous generation with persistent parameter learning and rate limit handling."""
        base_params = self._prepare_base_params(messages, temperature, **kwargs)
        max_tok = max_tokens or 2000
        
        token_strategies = [
            ("max_completion_tokens", max_tok),
            ("max_tokens", max_tok)
        ]
        
        last_error = None

        for token_key, token_val in token_strategies:
            call_params = base_params.copy()
            call_params[token_key] = token_val
            
            for attempt in range(3):
                try:
                    # Use async rate limit wrapper
                    response = await self._call_with_retry_async(
                        self.async_client.chat.completions.create,
                        **call_params
                    )
                    return self._parse_response(response)
                
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    
                    # Skip rate limit errors - already handled in _call_with_retry_async
                    if self._is_rate_limit_error(e):
                        raise
                    
                    if token_key in error_msg and ("unsupported" in error_msg or "unknown" in error_msg):
                        break
                    
                    stripped = False
                    candidates = ["temperature", "seed", "top_p", "presence_penalty", "frequency_penalty"]
                    
                    for bad_param in candidates:
                        if bad_param in call_params and self._is_unsupported_param_error(error_msg, bad_param):
                            logger.warning(f"Azure rejected '{bad_param}'. Permanently ignoring it for this client.")
                            
                            call_params.pop(bad_param)
                            self.unsupported_params.add(bad_param)
                            
                            stripped = True
                            break
                    
                    if stripped:
                        continue
                    
                    raise e
        
        if last_error:
            raise last_error
            
        raise RuntimeError("LLM generation failed silently.")

    def __repr__(self) -> str:
        return f"AzureOpenAIClient(model={self.model}, endpoint={self.endpoint[:30]}...)"


def create_llm_client(
    model: Optional[str] = None,
    provider: str = "azure",
) -> BaseLLMClient:
    if provider == "azure":
        return AzureOpenAIClient(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")