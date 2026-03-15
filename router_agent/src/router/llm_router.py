"""LLM-based router using Azure OpenAI client"""
import json
import yaml
from pathlib import Path
from typing import Dict, Tuple
from ..utils.llm_client import RouterLLMClient
from ..utils.logger import RouterLogger
from .prompts import format_router_prompt


class LLMRouter:
    """
    Routes queries using LLM to interpret intent.
    More robust than keyword matching.
    """
    
    # Tiebreaker priority: complex_dual > doc_only > fact_only > llm_only
    # Rationale: Prefer more capable paths that can handle complex queries
    TIEBREAK_PRIORITY = ["complex_dual", "doc_only", "fact_only", "llm_only"]
    
    def __init__(
        self,
        rules_path: str,
        azure_endpoint: str,
        azure_api_key: str,
        azure_api_version: str = "2024-10-01-preview",
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize LLM router.
        
        Args:
            rules_path: Path to routing_rules.yaml
            azure_endpoint: Azure endpoint URL
            azure_api_key: Azure API key
            azure_api_version: API version
            model: Model for routing (gpt-4o-mini recommended)
        """
        self.rules_path = Path(rules_path)
        self.rules_text = self._load_rules_as_text()
        
        # Initialize LLM client
        self.llm = RouterLLMClient(
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_api_version=azure_api_version,
            default_model=model
        )
        
        # Logger for observability
        self.logger = RouterLogger()
    
    def _load_rules_as_text(self) -> str:
        """Load YAML rules and format as readable text for LLM"""
        with open(self.rules_path) as f:
            config = yaml.safe_load(f)
        
        # Format rules as simple numbered list
        rules_text = []
        for i, rule in enumerate(config['rules'], 1):
            rule_str = f"{i}. {rule['description']}"
            rules_text.append(rule_str)
        
        # Add scoring instructions if present
        if 'scoring_instructions' in config:
            rules_text.append(f"\n{config['scoring_instructions']}")
        
        return "\n".join(rules_text)
    
    async def route(self, query: str) -> Tuple[str, Dict[str, float], str]:
        """
        Route query using LLM to interpret intent.
        
        Args:
            query: User query
            
        Returns:
            (selected_path, scores_dict, reasoning)
        """
        # 1. Prepare prompt
        prompt = format_router_prompt(query, self.rules_text)
        
        # 2. Call LLM with JSON mode
        try:
            response_text = await self.llm.generate(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                json_mode=True,
                temperature=0.0  # Deterministic routing
            )
            
            # 3. Parse JSON response
            response = json.loads(response_text)
            scores = {
                "llm_only": response.get("llm_only", 0),
                "fact_only": response.get("fact_only", 0),
                "doc_only": response.get("doc_only", 0),
                "complex_dual": response.get("complex_dual", 0)
            }
            reasoning = response.get("reasoning", "No reasoning provided")
            
        except json.JSONDecodeError as e:
            await self.logger.log_error(f"JSON parse error: {e}", {"response": response_text})
            # Fallback to safe default
            scores = {"llm_only": 0, "fact_only": 0, "doc_only": 0, "complex_dual": 5}
            reasoning = f"JSON parse error, defaulting to complex_dual: {str(e)}"
            
        except Exception as e:
            await self.logger.log_error(f"LLM routing failed: {e}")
            # Fallback to safe default
            scores = {"llm_only": 0, "fact_only": 0, "doc_only": 0, "complex_dual": 5}
            reasoning = f"Routing error, defaulting to complex_dual: {str(e)}"
        
        # 4. Select winner with tiebreaker
        max_score = max(scores.values())
        tied_paths = [k for k, v in scores.items() if v == max_score]
        
        if len(tied_paths) > 1:
            # Tiebreaker: use priority order (complex_dual > doc_only > fact_only > llm_only)
            selected_path = min(tied_paths, key=lambda k: self.TIEBREAK_PRIORITY.index(k))
            
            # Log tie for observability
            await self.logger.log_routing_decision(
                query=query,
                scores=scores,
                selected_path=selected_path,
                reasoning=f"TIE DETECTED among {tied_paths}. Priority tiebreaker selected: {selected_path}. Original reasoning: {reasoning}"
            )
        else:
            selected_path = tied_paths[0]
            
            # 5. Log decision (normal case)
            await self.logger.log_routing_decision(
                query=query,
                scores=scores,
                selected_path=selected_path,
                reasoning=reasoning
            )
        
        return selected_path, scores, reasoning
    
    async def close(self):
        """Cleanup resources"""
        await self.llm.close()