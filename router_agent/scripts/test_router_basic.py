"""
Basic router test - verify all components work together.

Tests:
1. Settings load from .env
2. LLM client connects to Azure
3. Router makes routing decisions
4. Logging works
"""
import asyncio
import sys
from pathlib import Path

# Add router_agent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.router.llm_router import LLMRouter


async def main():
    print("=" * 70)
    print("ROUTER BASIC TEST")
    print("=" * 70)
    print()
    
    # Test 1: Load settings
    print(" Test 1: Loading settings...")
    try:
        settings = get_settings()
        print(f" Settings loaded")
        print(f"   - Endpoint: {settings.llm__azure_endpoint[:40]}...")
        print(f"   - Model: {settings.router_model}")
        print(f"   - Rules: {settings.routing_rules_path}")
        print()
    except Exception as e:
        print(f" Settings failed: {e}")
        print("   Make sure .env file exists and has required values")
        return
    
    # Test 2: Initialize router
    print(" Test 2: Initializing router...")
    try:
        router = LLMRouter(
            rules_path=settings.routing_rules_path,
            azure_endpoint=settings.llm__azure_endpoint,
            azure_api_key=settings.llm__azure_api_key,
            azure_api_version=settings.llm__azure_api_version,
            model=settings.router_model
        )
        print(" Router initialized")
        print()
    except Exception as e:
        print(f" Router init failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Route test queries
    print(" Test 3: Routing test queries...")
    print()
    
    test_cases = [
        {
            "query": "When is Seollal public holiday in Korea",
            "expected": "llm_only",
            "description": "General Knowledge (Holidays)"
        },
        {
            "query": "Show last year's overseas export sales by product, by currency.",
            "expected": "fact_only",
            "description": "Specific database query from provided example"
        },
        {
            "query": "Can I receive travel expenses when working from the Cheongju plant to the Jincheon plant?",
            "expected": "doc_only",
            "description": "Document-specific question"
        },
        {
            "query": "Please calculate the travel expenses for my 4-day, 3-night business trip to Japan. Are overseas travel expenses paid in the local currency or in Korean Won?",
            "expected": "complex_dual",
            "description": "Multi-hop query"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected = test["expected"]
        description = test["description"]
        
        print(f"Test {i}/{len(test_cases)}: {description}")
        print(f"Query: {query}")
        
        try:
            selected_path, scores, reasoning = await router.route(query)
            
            # Check if correct
            status = "[Correct]" if selected_path == expected else "[Wrong]"
            results.append(selected_path == expected)
            
            print(f"{status} Expected: {expected} | Got: {selected_path}")
            print(f"   Scores: {scores}")
            print(f"   Reasoning: {reasoning}...")
            print()
            
        except Exception as e:
            print(f" Routing failed: {e}")
            results.append(False)
            import traceback
            traceback.print_exc()
            print()
    
    # Test 4: Check logs
    print("=" * 70)
    print(" Test 4: Checking logs...")
    log_file = Path(settings.log_dir) / f"router_{asyncio.get_event_loop().time()}.jsonl"
    
    # Find the most recent log file
    log_dir = Path(settings.log_dir)
    if log_dir.exists():
        log_files = sorted(log_dir.glob("router_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if log_files:
            latest_log = log_files[-1]
            with open(latest_log) as f:
                lines = f.readlines()
            print(f" Found log file: {latest_log}")
            print(f"   Entries: {len(lines)}")
            print()
        else:
            print("  No log files found")
            print()
    else:
        print("  Log directory doesn't exist")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    correct = sum(results)
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"Tests Passed: {correct}/{total} ({accuracy:.0f}%)")
    
    if accuracy == 100:
        print(" All tests passed!")
    elif accuracy >= 75:
        print("  Most tests passed, review failures")
    else:
        print(" Many tests failed, check configuration")
    
    print()
    
    # Cleanup
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())