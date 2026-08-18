#!/usr/bin/env python3
"""
Standalone test for router parsing robustness.
Tests regex-based parser with various LLM response formats.
"""

import re
from typing import Literal
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """Router decision result"""
    lane: Literal["fast", "slow"]
    confidence: float
    reasoning: str


def parse_decision(response: str, default_lane: str = "fast") -> RoutingDecision:
    """
    Parse LLM response using regex (robust version).
    """
    # Default values
    lane = default_lane
    confidence = 0.5
    reasoning = "Unable to parse LLM response"
    
    # Clean response (remove markdown code blocks)
    clean_response = re.sub(r'```[a-z]*\s*|\s*```', '', response, flags=re.IGNORECASE)
    
    # Extract LANE (case-insensitive, flexible spacing)
    lane_match = re.search(
        r'LANE\s*[:：]\s*(fast|slow)',
        clean_response,
        re.IGNORECASE
    )
    if lane_match:
        lane = lane_match.group(1).lower()
    
    # Extract CONFIDENCE
    conf_match = re.search(
        r'CONFIDENCE\s*[:：]\s*([0-9]*\.?[0-9]+)',
        clean_response,
        re.IGNORECASE
    )
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass
    
    # Extract REASONING (everything after "REASONING:" until end)
    reasoning_match = re.search(
        r'REASONING\s*[:：]\s*(.+?)$',  # Capture to end of string
        clean_response,
        re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        # Collapse multiple newlines into single space
        reasoning = re.sub(r'[\n\r]+', ' ', reasoning).strip()
        # Remove trailing noise
        reasoning = re.sub(r'\s*(hope this helps|let me know).*$', '', reasoning, flags=re.IGNORECASE).strip()
    
    return RoutingDecision(
        lane=lane,  # type: ignore
        confidence=confidence,
        reasoning=reasoning
    )


def main():
    """Test parser with various response formats"""
    
    print("=" * 70)
    print("ROUTER PARSING ROBUSTNESS TEST")
    print("=" * 70)
    print()
    
    test_cases = [
        # Case 1: Perfect format
        (
            "perfect",
            """LANE: fast
CONFIDENCE: 0.9
REASONING: Simple factual query"""
        ),
        
        # Case 2: Markdown code block
        (
            "markdown_block",
            """```
LANE: slow
CONFIDENCE: 0.85
REASONING: Complex comparison required
```"""
        ),
        
        # Case 3: Extra whitespace
        (
            "whitespace",
            """
LANE:    fast
CONFIDENCE:   0.95  
REASONING:    Single concept query   
"""
        ),
        
        # Case 4: Case insensitive
        (
            "case_insensitive",
            """lane: slow
confidence: 0.7
reasoning: Multi-step analysis needed"""
        ),
        
        # Case 5: Korean colon (：)
        (
            "korean_colon",
            """LANE： fast
CONFIDENCE： 0.88
REASONING： 단순한 사실 확인 질문"""
        ),
        
        # Case 6: Mixed case with noise
        (
            "noisy",
            """Here's my analysis:

Lane: SLOW
Confidence: 0.92
Reasoning: Requires integration of multiple concepts

Hope this helps!"""
        ),
        
        # Case 7: Missing confidence
        (
            "missing_confidence",
            """LANE: fast
REASONING: Direct question"""
        ),
        
        # Case 8: Invalid lane (should use default)
        (
            "invalid_lane",
            """LANE: medium
CONFIDENCE: 0.8
REASONING: Moderate complexity"""
        ),
        
        # Case 9: Multiline reasoning
        (
            "multiline_reasoning",
            """LANE: slow
CONFIDENCE: 0.87
REASONING: This query requires
multiple steps and comparisons"""
        ),
    ]
    
    passed = 0
    failed = 0
    verbose = True  # Show full reasoning for verification
    
    for name, response in test_cases:
        print(f" Test: {name}")
        print(f"   Input: {response[:50].replace(chr(10), ' ')}...")
        
        try:
            decision = parse_decision(response)
            
            print(f"    Parsed successfully:")
            print(f"      Lane: {decision.lane}")
            print(f"      Confidence: {decision.confidence:.2f}")
            
            if verbose:
                # Show full reasoning to verify multiline capture
                print(f"      Reasoning (full): {decision.reasoning}")
            else:
                print(f"      Reasoning: {decision.reasoning[:50]}...")
            
            passed += 1
        except Exception as e:
            print(f"    Parse error: {e}")
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()