"""
Simplified LLM client for router
Adapted from rag_agent/src/generation/llm_client.py
Only needs non-streaming generation with JSON mode
"""
import logging
import json
from typing import Optional, Dict, Any, Set
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)


class RouterLLMClient:
    """
    Lightweight Azure OpenAI client for routing decisions.
    
    Simplified from RAG agent's client - only supports:
    - Non-streaming generation
    - JSON mode for structured outputs
    - Temperature = 0 for deterministic routing
    """
    
    PROTECTED_KEYS = {"model", "messages"}
    DISALLOWED_KEYS = {"n", "stream"}  # Router doesn't use these
    
    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        azure_api_version: str = "2024-08-01-preview",
        default_model: str = "gpt-4o-mini"
    ):
        """
        Initialize router LLM client.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: API key
            azure_api_version: API version
            default_model: Model for routing (gpt-4o-mini recommended)
        """
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_api_version = azure_api_version
        self.default_model = default_model
        
        # Track unsupported parameters (learned at runtime)
        self.unsupported_params: Set[str] = set()
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version
        )
        
        logger.info(f"Router LLM client initialized (model={default_model})")
    
    def _prepare_params(
        self,
        messages: list,
        temperature: float,
        json_mode: bool,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare API parameters, excluding unsupported ones"""
        params = {
            "model": model or self.default_model,
            "messages": messages,
        }
        
        # Temperature (deterministic for routing)
        if "temperature" not in self.unsupported_params:
            params["temperature"] = temperature
        
        # Max tokens (routing responses are short)
        if "max_completion_tokens" not in self.unsupported_params:
            params["max_completion_tokens"] = 500  # Routing decisions are brief
        elif "max_tokens" not in self.unsupported_params:
            params["max_tokens"] = 500
        
        # JSON mode (critical for structured routing output)
        if json_mode and "response_format" not in self.unsupported_params:
            params["response_format"] = {"type": "json_object"}
        
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
            "unknown parameter"
        ]
        return any(t in error_msg for t in triggers) and param_name in error_msg
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        temperature: float = 0.0,
        model: Optional[str] = None
    ) -> str:
        """
        Generate completion for routing decision.
        
        Args:
            system_prompt: Router system prompt with rules
            user_prompt: Query to route
            json_mode: Enforce JSON output (True for routing)
            temperature: Sampling temp (0.0 for deterministic)
            model: Override default model
            
        Returns:
            Generated text (JSON string if json_mode=True)
            
        Raises:
            Exception: If generation fails after retries
        """
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Prepare params
        params = self._prepare_params(
            messages=messages,
            temperature=temperature,
            json_mode=json_mode,
            model=model
        )
        
        model_name = params["model"]
        logger.debug(f"Router LLM call (model={model_name}, json_mode={json_mode})")
        
        # Retry with parameter stripping if needed
        last_error = None
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(**params)
                
                # Extract response
                choice = response.choices[0]
                content = choice.message.content or ""
                
                # Validate JSON if json_mode enabled
                if json_mode:
                    try:
                        json.loads(content)  # Validate
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from LLM (attempt {attempt+1}): {e}")
                        if attempt < 2:  # Retry if not last attempt
                            continue
                        raise ValueError(f"LLM returned invalid JSON: {content[:200]}")
                
                logger.debug(f"Router decision generated ({len(content)} chars)")
                return content
            
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Try to strip unsupported params
                stripped = False
                candidates = ["temperature", "max_completion_tokens", "max_tokens", "response_format"]
                
                for bad_param in candidates:
                    if bad_param in params and self._is_unsupported_param_error(error_msg, bad_param):
                        logger.warning(f"Parameter '{bad_param}' unsupported. Removing and retrying...")
                        params.pop(bad_param)
                        self.unsupported_params.add(bad_param)
                        stripped = True
                        break
                
                if stripped:
                    continue  # Retry without bad param
                
                # Not a param error - raise immediately
                logger.error(f"Router LLM error: {e}")
                raise Exception(f"Router LLM generation failed: {str(e)[:200]}")
        
        # Exhausted retries
        if last_error:
            raise last_error
        
        raise RuntimeError("Router LLM generation failed silently")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.close()
        logger.debug("Router LLM client closed")


def create_router_llm_client(
    azure_endpoint: str,
    azure_api_key: str,
    azure_api_version: str = "2024-08-01-preview",
    model: str = "gpt-4o-mini"
) -> RouterLLMClient:
    """
    Factory function for router LLM client.
    
    Args:
        azure_endpoint: Azure OpenAI endpoint
        azure_api_key: API key
        azure_api_version: API version
        model: Model name (gpt-4o-mini recommended for routing)
        
    Returns:
        RouterLLMClient instance
    """
    return RouterLLMClient(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        default_model=model
    )