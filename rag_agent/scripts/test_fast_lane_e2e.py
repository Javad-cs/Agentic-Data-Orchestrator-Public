#!/usr/bin/env python3
"""
Fast Lane End-to-End Integration Test

Tests the complete Fast Lane pipeline with all features:
1. Query Expansion (3 variants)
2. Hybrid Retrieval (Dense + Sparse + RRF)
3. Multi-query RRF Merge
4. Reranking (Cohere - if available)
5. Answer Generation with Citations
6. NLI Safety Check

Usage:
    python scripts/test_fast_lane_e2e.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.agents.fast_lane import create_fast_lane


async def main():
    """Test complete Fast Lane pipeline"""
    
    print("=" * 70)
    print("FAST LANE END-TO-END INTEGRATION TEST")
    print("=" * 70)
    print()
    
    # Load configuration
    print(" Loading configuration...")
    try:
        config = SystemConfig()
        print(" Configuration loaded")
        print()
    except Exception as e:
        print(f" Configuration error: {e}")
        print("   Make sure .env file is properly configured")
        return
    
    # Display feature flags
    print(" Feature Configuration:")
    print(f"   Query Expansion: {config.fast_lane.query_expansion.enabled}")
    print(f"   - Variants: {config.fast_lane.query_expansion.num_variants}")
    print(f"   - Parallel: {config.fast_lane.query_expansion.parallel}")
    print()
    print(f"   Reranker: {config.fast_lane.reranker.enabled}")
    if config.fast_lane.reranker.enabled:
        print(f"   - Provider: {config.fast_lane.reranker.provider}")
        print(f"   - Model: {config.fast_lane.reranker.cohere_model}")
    print()
    print(f"   Safety Check: {config.fast_lane.safety_check.enabled}")
    print(f"   - Use NLI: {config.fast_lane.safety_check.use_nli}")
    print(f"   - NLI Model: {config.fast_lane.safety_check.nli_model}")
    print(f"   - NLI Threshold: {config.fast_lane.safety_check.nli_threshold}")
    print()
    print(f"   Router: {config.router.enabled}")
    print(f"   - Model: {config.router.model_name}")
    print()
    
    # Initialize Fast Lane
    print(" Initializing Fast Lane...")
    try:
        fast_lane = await create_fast_lane(config)
        print(" Fast Lane initialized successfully")
        print()
    except Exception as e:
        print(f" Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries
    test_queries = [
        # Simple factual query (should be fast)
        {
            "query": "스테인레스강 가공 시 권장 절삭 속도는?",
            "language": "ko",
            "expected": "Fast, factual answer with citations"
        },
        
        # Technical query
        {
            "query": "PVD 코팅의 장점은 무엇인가요?",
            "language": "ko",
            "expected": "Technical explanation with citations"
        },
        
        # English query
        {
            "query": "What are the benefits of using coolant in machining?",
            "language": "en",
            "expected": "English answer with citations"
        },
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print("=" * 70)
        print(f"TEST {i}/{len(test_queries)}")
        print("=" * 70)
        print(f"Query: {test_case['query']}")
        print(f"Language: {test_case['language']}")
        print(f"Expected: {test_case['expected']}")
        print()
        
        # Track metrics
        import time
        start_time = time.time()
        
        answer_chunks = []
        citations = []
        citation_details = {}  # Track citation number -> citation data
        metadata = {}
        status_updates = []
        
        try:
            # Run Fast Lane query
            async for event in fast_lane.query(
                query=test_case['query'],
                top_k=5,
                language=test_case['language'],
                streaming=True
            ):
                event_type = event.get("type")
                
                # Track status updates
                if event_type == "status":
                    status = event.get("content", "")
                    status_updates.append(status)
                    print(f"    Status: {status}")
                
                # Collect answer chunks
                elif event_type == "chunk":
                    chunk = event.get("content", "")
                    answer_chunks.append(chunk)
                    print(chunk, end="", flush=True)
                
                # Collect citations and track details
                elif event_type == "citation":
                    citation_data = event.get("data", {})
                    citations.append(citation_data)
                    # Store citation details by ID for later lookup
                    citation_id = citation_data.get("id", "")
                    citation_details[citation_id] = citation_data
                
                # Get final metadata
                elif event_type == "done":
                    metadata = event.get("metadata", {})
                    if answer_chunks:  # Only print newline if we got chunks
                        print()  # Newline after content
                
                # Handle errors
                elif event_type == "error":
                    error_data = event.get("data", {})
                    print(f"\n    Error: {error_data.get('message', 'Unknown error')}")
                    break
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Analyze which citations were actually used
            full_answer = ''.join(answer_chunks)
            import re
            citation_pattern = r'\[(\d+)\]'
            used_citation_numbers = set(re.findall(citation_pattern, full_answer))
            
            # Display results
            print()
            print("-" * 70)
            print("RESULTS:")
            print(f"   ️  Total Time: {total_time:.2f}s")
            print(f"    Answer Length: {len(full_answer)} chars")
            print(f"    Citations Retrieved: {len(citations)}")
            print(f"    Citations Actually Used: {len(used_citation_numbers)}")
            
            # Show which citations were used
            if used_citation_numbers:
                print(f"      Used: {sorted([int(n) for n in used_citation_numbers])}")
            
            # Show citation details
            if citation_details:
                print()
                print("    Citation Details:")
                for cite_num in sorted([int(n) for n in used_citation_numbers]):
                    cite_id = f"[{cite_num}]"
                    if cite_id in citation_details:
                        cite = citation_details[cite_id]
                        file_name = cite.get('file', 'Unknown')
                        page_num = cite.get('page')
                        parent_text = cite.get('parent_text', '')
                        
                        page_info = f", page {page_num}" if page_num else ""
                        print(f"      [{cite_num}] {file_name}{page_info}")
                        
                        # Show parent text preview if available
                        if parent_text:
                            # Clean up the text for display
                            preview = parent_text.replace('\n', ' ').strip()
                            print(f"          Preview: {preview}")
                        else:
                            # Fallback to child_text if no parent
                            child_text = cite.get('child_text', '')
                            if child_text:
                                preview = child_text.replace('\n', ' ').strip()
                                print(f"          Preview: {preview}")
            
            if metadata:
                print()
                print(f"    Metadata:")
                for key, value in metadata.items():
                    print(f"      - {key}: {value}")
            
            # Check if meets performance target
            if total_time < 4.0:
                print(f"    Performance: PASS (<4s target)")
            else:
                print(f"   ️  Performance: SLOW (>{total_time:.1f}s, target <4s)")
            
            # Check if has citations (check actual usage, not just retrieval)
            if used_citation_numbers:
                print(f"    Citations: USED ({len(used_citation_numbers)} citations)")
            else:
                print(f"   ️  Citations: NOT USED (0 citations in answer)")
            
            print()
        
        except Exception as e:
            print(f"\n    Query error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Cleanup
    print("=" * 70)
    print("CLEANUP")
    print("=" * 70)
    print(" Closing Fast Lane...")
    try:
        await fast_lane.close()
        print(" Fast Lane closed")
    except Exception as e:
        print(f"️  Cleanup error: {e}")
    
    print()
    print("=" * 70)
    print(" ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())