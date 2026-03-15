import logging
from typing import AsyncIterator, Optional, List, Set, Dict, Any
from openai import AsyncAzureOpenAI

from .base import BaseGenerator, GenerationResult, StreamChunk
from src.config.models import LLMConfig

logger = logging.getLogger(__name__)


class AzureOpenAIClient(BaseGenerator):
    """
    Azure OpenAI client with streaming support.
    
    Uses official OpenAI SDK (same as text_to_sql project).
    Includes parameter learning to avoid unsupported param errors.
    """
    
    # Parameters that can cause issues with certain models
    PROTECTED_KEYS = {"model", "messages"}
    DISALLOWED_KEYS = {"n", "response_format"}  # Not supported yet
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Azure OpenAI client.
        
        Args:
            config: LLMConfig with Azure credentials and settings
        """
        self.config = config
        
        # Validate Azure config
        if config.provider != "azure":
            raise ValueError(f"AzureOpenAIClient requires provider='azure', got '{config.provider}'")
        
        # Memory: Store params this model rejected
        self.unsupported_params: Set[str] = set()
        
        # Create async OpenAI client (official SDK)
        self.client = AsyncAzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version
        )
        
        logger.info(f"AzureOpenAI client initialized (model={config.default_model})")
    
    def get_model_name(self) -> str:
        """Return the default model name"""
        return self.config.default_model
    
    def _prepare_params(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare API call parameters, excluding known unsupported params.
        """
        params = {
            "model": model or self.config.default_model,
            "messages": messages,
        }

        if params["model"] == "gpt-5-mini":
            self.unsupported_params.add("temperature")
        
        # Add temperature if not blocked
        if temperature is not None and "temperature" not in self.unsupported_params:
            params["temperature"] = temperature
        elif "temperature" not in self.unsupported_params:
            params["temperature"] = self.config.temperature
        
        # Add max_tokens (try max_completion_tokens first for newer models)
        max_tok = max_tokens or self.config.max_tokens
        if "max_completion_tokens" not in self.unsupported_params:
            params["max_completion_tokens"] = max_tok
        elif "max_tokens" not in self.unsupported_params:
            params["max_tokens"] = max_tok
        
        # Add other kwargs if not blocked
        for k, v in kwargs.items():
            if k in self.PROTECTED_KEYS:
                logger.warning(f"Ignoring attempt to override protected parameter '{k}'")
                continue
            if k in self.DISALLOWED_KEYS:
                logger.warning(f"Parameter '{k}' is not currently supported, skipping")
                continue
            if k in self.unsupported_params:
                continue
            
            params[k] = v
        
        return params
    
    def _is_unsupported_param_error(self, error_msg: str, param_name: str) -> bool:
        """Check if error is due to unsupported parameter"""
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
        return any(t in error_msg for t in triggers) and param_name in error_msg
    
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
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare params
        params = self._prepare_params(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=False
        )
        
        model_name = params["model"]
        logger.debug(f"Generating completion (model={model_name}, temp={params.get('temperature', 'N/A')})")
        
        # Retry with parameter stripping if needed
        last_error = None
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(**params)
                
                # Parse response
                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = choice.finish_reason
                usage = response.usage
                
                logger.info(f"Generated {usage.completion_tokens if usage else 0} tokens")
                
                return GenerationResult(
                    content=content,
                    model=model_name,
                    finish_reason=finish_reason,
                    usage={
                        'prompt_tokens': usage.prompt_tokens if usage else 0,
                        'completion_tokens': usage.completion_tokens if usage else 0,
                        'total_tokens': usage.total_tokens if usage else 0
                    } if usage else {},
                    metadata={}
                )
            
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                
                # Try to strip unsupported params
                stripped = False
                candidates = ["temperature", "max_completion_tokens", "max_tokens", "seed", "top_p"]
                
                for bad_param in candidates:
                    if bad_param in params and self._is_unsupported_param_error(error_msg, bad_param):
                        logger.warning(f"Azure rejected '{bad_param}'. Removing and retrying...")
                        params.pop(bad_param)
                        self.unsupported_params.add(bad_param)
                        stripped = True
                        break
                
                if stripped:
                    continue  # Retry without the bad param
                
                # If not a param error, raise immediately
                logger.error(f"LLM generation error: {e}")
                raise Exception(f"LLM generation failed: {str(e)[:200]}")
        
        # If we exhausted retries
        if last_error:
            raise last_error
        
        raise RuntimeError("LLM generation failed silently")
    
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
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare params
        params = self._prepare_params(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=True
        )
        
        model_name = params["model"]
        logger.debug(f"Streaming generation (model={model_name}, temp={params.get('temperature', 'N/A')})")
        
        # Retry with parameter stripping if needed
        last_error = None
        for attempt in range(3):
            try:
                # Stream from OpenAI SDK
                logger.debug(f"Starting OpenAI stream (attempt {attempt + 1}/3)...")
                stream = await self.client.chat.completions.create(**params)
                
                chunks_yielded = 0
                content_chars = 0
                async for chunk in stream:
                    if not chunk.choices:
                        logger.debug("Received chunk with no choices, skipping")
                        continue
                    
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    content = delta.content or ""
                    finish_reason = choice.finish_reason
                    
                    if content:
                        content_chars += len(content)
                    
                    # Yield if there's content OR a finish_reason
                    # (final chunk may have finish_reason but no content)
                    if content or finish_reason:
                        chunks_yielded += 1
                        yield StreamChunk(
                            content=content,
                            finish_reason=finish_reason
                        )
                        
                # Log completion
                logger.info(f"Stream completed: {chunks_yielded} chunks, {content_chars} chars (attempt {attempt + 1}/3)")

                # if stream ended but produced no text, retry with backoff
                if content_chars == 0:
                    logger.warning(f"Empty stream (0 chars) on attempt {attempt + 1}/3 - retrying...")
                    if attempt < 2:  # Not the last attempt
                        import asyncio
                        await asyncio.sleep(0.5 * (attempt + 1))  # 0.5s, 1s backoff
                    continue  # Try next attempt

                # If we got here, streaming succeeded with content
                return
            
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Retry on empty-content streams (transient)
                if "empty streaming content" in error_msg:
                    logger.warning(f"Empty streaming content detected (attempt {attempt + 1}/3); retrying...")
                    continue
                
                # Try to strip unsupported params
                stripped = False
                candidates = ["temperature", "max_completion_tokens", "max_tokens", "seed", "top_p"]
                
                for bad_param in candidates:
                    if bad_param in params and self._is_unsupported_param_error(error_msg, bad_param):
                        logger.warning(f"Azure rejected '{bad_param}' in streaming. Removing and retrying...")
                        params.pop(bad_param)
                        self.unsupported_params.add(bad_param)
                        stripped = True
                        break
                
                if stripped:
                    continue  # Retry without the bad param
                
                # If not a param error, raise immediately
                logger.error(f"Streaming error: {e}")
                raise Exception(f"LLM streaming failed: {str(e)[:200]}")
        
        # If we exhausted retries
        if last_error:
            raise last_error
        
        raise RuntimeError("All 3 streaming attempts returned empty content")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.close()
        logger.debug("Azure OpenAI client closed")


# Factory function for easy instantiation
def create_llm_client(config: LLMConfig) -> BaseGenerator:
    """
    Factory function to create LLM client based on config.
    
    Args:
        config: LLMConfig
        
    Returns:
        BaseGenerator instance (currently only AzureOpenAIClient)
    """
    if config.provider == "azure":
        return AzureOpenAIClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")