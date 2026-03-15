"""
Query Masker using LLM.
Based on the MCS-SQL paper [LPKP2024] Appendix B.2.
Replaces schema-specific terms (Table, Column, Value) with generic tokens.
Includes robust output cleaning (Longest Line Heuristic) and System/User split.
"""

import logging
import re
import hashlib
from typing import Dict, List
from core import BaseLLMClient, LLMMessage

logger = logging.getLogger(__name__)

class QueryMasker:
    """
    Masks natural language queries using an LLM to abstract away 
    schema-specific details, improving few-shot retrieval.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        # Simple in-memory cache: (question, schema_hash) -> masked_question
        self._cache: Dict[str, str] = {}

    def mask(self, question: str, schema_context: str) -> str:
        """
        Masks the question using the provided schema context.
        """
        # 1. Stable Cache Key
        schema_hash = hashlib.md5(schema_context.encode("utf-8")).hexdigest()
        cache_key = f"{question}||{schema_hash}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 2. Build Messages (System + User split for robustness)
        messages = self._build_messages(question, schema_context)
        
        try:
            # 3. Call LLM
            # Set temperature=0.3 to avoid strict deterministic lock-up (empty output)
            response = self.llm_client.generate(
                messages,
                max_tokens=300, 
                temperature=0.3 
            )
            
            # 4. Clean Output
            raw = response.content.strip()
            masked = self._clean_output(raw)
            
            # 5. Fallback Check
            if not masked:
                # Only log raw if empty to debug "mute" models
                logger.warning(f"Masking returned empty. Raw: '{raw}'")
                return self._cheap_mask(question)
                
            self._cache[cache_key] = masked
            return masked
            
        except Exception as e:
            logger.error(f"Masking failed: {e}. Using regex fallback (not cached).")
            return self._cheap_mask(question)

    def _cheap_mask(self, text: str) -> str:
        """
        Deterministic regex fallback.
        """
        # Mask numbers
        text = re.sub(r"\b\d+(\.\d+)?\b", "[VALUE]", text)
        # Mask ISO dates
        text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[VALUE]", text)
        # Mask quoted strings
        text = re.sub(r"'[^']*'", "[VALUE]", text)
        text = re.sub(r'"[^"]*"', "[VALUE]", text)
        return text

    def _clean_output(self, text: str) -> str:
        """
        Robustly cleans the LLM output.
        Implements 'Longest Line' heuristic to avoid truncation.
        """
        # Remove common prefixes
        text = re.sub(r"^(Masked Question:|Answer:|Output:)\s*", "", text, flags=re.IGNORECASE)
        
        # Remove markdown
        text = text.replace("```", "").strip()
        
        # Split into lines and remove empty ones
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return ""
            
        # HEURISTIC: Return the longest line. 
        # This handles cases where the model says "Here is the result:\n[The Masked Question]"
        return max(lines, key=len)

    def _build_messages(self, question: str, schema_context: str) -> List[LLMMessage]:
        """Constructs the chat messages for the model."""
        
        system_content = """You are a Query Masking Agent. Your job is to generalize a natural language question by replacing specific schema terms with generic tokens.
Rules:
1. Replace table names with [TABLE].
2. Replace column names with [COLUMN].
3. Replace specific values (numbers, dates, strings, entities) with [VALUE].
4. Output ONLY the masked question. Do not explain. Do not use prefixes."""

        user_content = f"""### Examples

<example>
### Schema:
# customers (CustomerID, Segment, Currency)
# transactions (TransactionID, Date, Amount, Price)
### Question: For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.
### Masked Question: For all the [TABLE] who paid more than [VALUE] per unit of [COLUMN] [VALUE]. Give their consumption status in the [VALUE].
</example>

<example>
### Schema:
# drivers (driverId, nationality, dob)
### Question: How many Australian drivers who were born in 1980?
### Masked Question: How many [VALUE] [TABLE] who were born in [VALUE]?
</example>

### Task
<SCHEMA>
{schema_context}
</SCHEMA>

### Question: {question}
### Masked Question:"""

        return [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=user_content)
        ]