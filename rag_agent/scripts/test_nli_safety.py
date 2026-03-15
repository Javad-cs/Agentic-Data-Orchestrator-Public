#!/usr/bin/env python3
"""
Test NLI Safety Checker.

Tests:
1. Citation checking
2. No-answer phrase detection
3. NLI entailment checking (if model available)
4. Overall safety validation

Usage:
    python scripts/test_nli_safety.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.nli_safety_check import NLISafetyChecker


async def main():
    """Test NLI safety checker"""
    
    print("=" * 70)
    print("NLI SAFETY CHECKER TEST")
    print("=" * 70)
    print()
    
    # Initialize checker
    print(" Initializing NLI safety checker...")
    checker = NLISafetyChecker(
        nli_model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_threshold=0.5,
        use_nli=True,  # Enable deep NLI checking
        check_citations=True,
        min_answer_length=10
    )
    print(" Checker initialized")
    print()
    
    # Sample context
    context_chunks = [
        {
            "parent_text": "스테인레스강 가공 시 절삭 속도는 60-80 m/min이 권장됩니다. "
                          "PVD 코팅 공구를 사용하면 공구 수명이 2-3배 증가합니다.",
            "child_text": "절삭 속도 60-80 m/min 권장"
        },
        {
            "parent_text": "냉각수를 충분히 사용하면 가공 품질이 향상됩니다. "
                          "특히 고속 가공에서는 필수적입니다.",
            "child_text": "냉각수 사용으로 품질 향상"
        }
    ]
    
    # Test cases
    test_cases = [
        # Case 1: Good answer with citations (PASS)
        {
            "name": "good_with_citations",
            "answer": "스테인레스강 가공 시 절삭 속도는 60-80 m/min이 권장됩니다[1]. "
                     "PVD 코팅 공구를 사용하면 공구 수명이 증가합니다[1].",
            "citations": [{"number": 1, "source": "doc1"}],
            "expected": True
        },
        
        # Case 2: Too short (FAIL)
        {
            "name": "too_short",
            "answer": "60-80",
            "citations": None,
            "expected": False
        },
        
        # Case 3: No citations (FAIL)
        {
            "name": "no_citations",
            "answer": "스테인레스강 가공 시 절삭 속도는 60-80 m/min이 권장됩니다. "
                     "PVD 코팅 공구를 사용하면 좋습니다.",
            "citations": None,
            "expected": False
        },
        
        # Case 4: Has "I don't know" phrase (FAIL)
        {
            "name": "no_answer_phrase",
            "answer": "죄송하지만 정보가 없습니다[1].",
            "citations": [{"number": 1, "source": "doc1"}],
            "expected": False
        },
        
        # Case 5: Hallucinated content (FAIL - if NLI works)
        {
            "name": "hallucinated",
            "answer": "스테인레스강 가공 시 절삭 속도는 200 m/min이 권장됩니다[1]. "
                     "다이아몬드 공구를 사용하면 가장 좋습니다[1].",
            "citations": [{"number": 1, "source": "doc1"}],
            "expected": False  # Should fail NLI check
        },
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print("-" * 70)
        print(f" Test: {test_case['name']}")
        print(f"   Answer: {test_case['answer'][:60]}...")
        print(f"   Expected: {'PASS' if test_case['expected'] else 'FAIL'}")
        print()
        
        # Run safety check
        result = await checker.check(
            answer=test_case['answer'],
            context_chunks=context_chunks,
            citations=test_case['citations']
        )
        
        # Display result
        actual = "PASS" if result.passed else "FAIL"
        emoji = "" if result.passed else ""
        
        print(f"   {emoji} Result: {actual}")
        print(f"      Confidence: {result.confidence:.2f}")
        
        if result.has_issues:
            print(f"      Issues:")
            for issue in result.issues:
                print(f"        - {issue}")
        
        if result.nli_scores:
            avg_nli = sum(result.nli_scores) / len(result.nli_scores)
            print(f"      NLI avg score: {avg_nli:.3f}")
        
        # Check if matches expectation
        if result.passed == test_case['expected']:
            print(f"    Test PASSED (matched expectation)")
            passed += 1
        else:
            print(f"    Test FAILED (expected {test_case['expected']}, got {result.passed})")
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    # Note about NLI model
    if not checker.use_nli:
        print()
        print("️  Note: NLI model not loaded. Deep entailment checks skipped.")
        print("   Install: pip install sentence-transformers")


if __name__ == "__main__":
    asyncio.run(main())