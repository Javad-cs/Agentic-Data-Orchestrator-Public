"""
Literal extraction from natural language questions.
Extracts entities, numbers, dates, and quoted strings for LSH literal matching.
"""

from typing import List, Set, Optional
import re
from dataclasses import dataclass
from core import BaseLLMClient, create_llm_client, LLMMessage


@dataclass
class ExtractedLiteral:
    """A literal extracted from a question."""
    value: str
    type: str  # "quoted", "number", "date", "capitalized", "entity"
    source: str  # "heuristic" or "llm"
    
    def __repr__(self):
        return f"Literal('{self.value}', type={self.type})"


class LiteralExtractor:
    """
    Extract literals from natural language questions.
    
    Uses heuristic extraction (fast, no cost) with optional LLM fallback.
    Paper-aligned: Extract literals to query LSH index for field matching.
    """
    
    def __init__(
        self, 
        llm_client: Optional[BaseLLMClient] = None,
        use_llm_fallback: bool = False
    ):
        """
        Initialize literal extractor.
        
        Args:
            llm_client: LLM client for optional fallback
            use_llm_fallback: Whether to use LLM if heuristics find few literals
        """
        self.llm_client = llm_client
        self.use_llm_fallback = use_llm_fallback
    
    def extract(
        self, 
        question: str,
        min_literals: int = 1
    ) -> List[ExtractedLiteral]:
        """
        Extract literals from question.
        
        Args:
            question: Natural language question
            min_literals: Minimum literals to extract (triggers LLM fallback if needed)
            
        Returns:
            List of ExtractedLiteral
        """
        literals = []
        
        # Step 1: Heuristic extraction (fast, free)
        literals.extend(self._extract_heuristic(question))
        
        # Step 2: Optional LLM fallback (if too few literals found)
        if self.use_llm_fallback and len(literals) < min_literals:
            if self.llm_client is None:
                self.llm_client = create_llm_client()
            
            llm_literals = self._extract_llm(question)
            literals.extend(llm_literals)
        
        # Deduplicate
        seen = set()
        unique_literals = []
        for lit in literals:
            key = (lit.value.lower(), lit.type)
            if key not in seen:
                seen.add(key)
                unique_literals.append(lit)
        
        return unique_literals
    
    def _extract_heuristic(self, question: str) -> List[ExtractedLiteral]:
        """
        Extract literals using heuristic rules.
        
        Fast and free - no LLM needed.
        """
        literals = []
        
        # 1. Quoted strings: "Fresno County", 'District 5'
        quoted = self._extract_quoted(question)
        literals.extend([
            ExtractedLiteral(q, "quoted", "heuristic") 
            for q in quoted
        ])
        
        # 2. Numbers: 500, 1.5, 2020
        numbers = self._extract_numbers(question)
        literals.extend([
            ExtractedLiteral(n, "number", "heuristic") 
            for n in numbers
        ])
        
        # 3. Dates: 2020-01-01, 01/15/2020
        dates = self._extract_dates(question)
        literals.extend([
            ExtractedLiteral(d, "date", "heuristic") 
            for d in dates
        ])
        
        # 4. Capitalized phrases: "Fresno County", "District Five"
        # (Only if not already captured in quotes)
        quoted_values = {q.lower() for q in quoted}
        capitalized = self._extract_capitalized(question)
        literals.extend([
            ExtractedLiteral(c, "capitalized", "heuristic") 
            for c in capitalized 
            if c.lower() not in quoted_values
        ])
        
        return literals
    
    def _extract_quoted(self, question: str) -> List[str]:
        """Extract quoted strings."""
        # Match both single and double quotes
        pattern = r'["\']([^"\']+)["\']'
        matches = re.findall(pattern, question)
        return [m.strip() for m in matches if m.strip()]
    
    def _extract_numbers(self, question: str) -> List[str]:
        """Extract numbers (integers and floats)."""
        # Match integers and decimals, but not parts of dates
        # Avoid matching years in dates like "2020-01-01"
        pattern = r'(?<!\d)(?<![/-])\b(\d+\.?\d*)\b(?![/-])'
        matches = re.findall(pattern, question)
        return [m for m in matches if m]
    
    def _extract_dates(self, question: str) -> List[str]:
        """Extract date-like strings."""
        patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}/\d{2}/\d{2}\b',  # YYYY/MM/DD
            r'\b\d{2}-\d{2}-\d{4}\b',  # DD-MM-YYYY
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, question)
            dates.extend(matches)
        
        return dates
    
    def _extract_capitalized(self, question: str) -> List[str]:
        """
        Extract capitalized phrases (proper nouns).
        
        Examples:
        - "Fresno County"
        - "San Francisco"
        - "District Five"
        """
        # Match sequences of capitalized words
        # At least 2 chars per word to avoid single letters
        pattern = r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b'
        matches = re.findall(pattern, question)
        
        # Filter out common question words
        stop_words = {
            "Show", "List", "Find", "Get", "What", "Which", "Where", 
            "How", "Why", "When", "Who", "Are", "Is", "The"
        }
        
        return [m for m in matches if m not in stop_words and len(m) > 2]
    
    def _extract_llm(self, question: str) -> List[ExtractedLiteral]:
        """
        Extract literals using LLM (fallback when heuristics find too few).
        
        Uses small/cheap model with strict JSON output.
        """
        prompt = f"""Extract literal values from this question that might appear in a database.

Question: {question}

Extract:
- Quoted strings
- Numbers
- Dates
- Proper nouns (names, places, organizations)
- Any specific values the user is searching for

Return ONLY a JSON array of strings, nothing else. No explanation.

Example format: ["value1", "value2", "value3"]

If no literals found, return: []"""

        try:
            messages = [LLMMessage(role="user", content=prompt)]
            response = self.llm_client.generate(messages, max_tokens=200)
            
            if not response.content:
                return []
            
            # Parse JSON
            import json
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r'```(?:json)?\s*\n?', '', content)
                content = re.sub(r'\n?```\s*$', '', content)
            
            literals_list = json.loads(content)
            
            return [
                ExtractedLiteral(lit, "entity", "llm")
                for lit in literals_list
                if isinstance(lit, str) and lit.strip()
            ]
        
        except Exception as e:
            print(f"Warning: LLM literal extraction failed: {e}")
            return []
    
    def extract_simple(self, question: str) -> List[str]:
        """
        Simple interface: just return literal values as strings.
        
        Args:
            question: Natural language question
            
        Returns:
            List of literal value strings
        """
        literals = self.extract(question)
        return [lit.value for lit in literals]


# Convenience function
def extract_literals(question: str) -> List[str]:
    """
    Quick literal extraction (heuristic only).
    
    Args:
        question: Natural language question
        
    Returns:
        List of literal strings
        
    Examples:
        >>> extract_literals('Show schools in "Fresno County" with enrollment over 500')
        ['Fresno County', '500', 'Fresno County']
    """
    extractor = LiteralExtractor(use_llm_fallback=False)
    return extractor.extract_simple(question)