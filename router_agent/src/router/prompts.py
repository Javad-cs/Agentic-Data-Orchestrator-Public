"""Prompts for LLM-based routing - following the paper's exact approach"""

ROUTER_SYSTEM_PROMPT = """You are an expert question analyzer. Based on the given question and the following rules, you need to score each of the four possible answer paths from 0-10.

Question: "{query}"

Rules:
{rules_text}

Answer Paths:
1. LLM_ONLY: Use general knowledge to answer (for common knowledge questions, definitions, explanations)
2. FACT_ONLY: Search structured data/tables for specific facts, numbers, or data points
3. DOC_ONLY: Search documents/paragraphs for contextual information
4. COMPLEX_DUAL: Use both structured data and documents (for complex questions requiring multiple sources)

IMPORTANT: All scores must be integers between 0 and 10 (inclusive). Do not use scores outside this range.

Please analyze the question according to the rules above and provide scores for each path in the following JSON format:
{{
    "llm_only": <score>,
    "fact_only": <score>,
    "doc_only": <score>,
    "complex_dual": <score>,
    "reasoning": "<brief explanation>"
}}

Only return the JSON object, no other text."""


def format_router_prompt(query: str, rules_text: str) -> dict:
    """
    Format the routing prompt following the paper's approach.
    
    Args:
        query: User query to route
        rules_text: Formatted rules from YAML
        
    Returns:
        Dictionary with system and user prompts
    """
    return {
        "system": ROUTER_SYSTEM_PROMPT.format(
            query=query,
            rules_text=rules_text
        ),
        "user": "Provide routing scores:"
    }