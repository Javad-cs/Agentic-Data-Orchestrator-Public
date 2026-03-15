#!/usr/bin/env python3
"""
Integration test for StreamingGenerator.

Tests the full flow: retrieve → generate → stream with citations.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.generation import (
    create_llm_client,
    CitationFormatter,
    SafetyChecker,
    create_streaming_generator,
    EventType
)


async def test_streaming_generation():
    """Test streaming generation with citations"""
    print("\n" + "="*70)
    print("TEST: Streaming Generator with Citations")
    print("="*70)
    
    # Load config
    config = SystemConfig()
    
    # Create components
    llm_client = create_llm_client(config.llm)
    citation_formatter = CitationFormatter()
    safety_checker = SafetyChecker(
        check_no_answer=True,
        check_citation=True,
        min_answer_length=10,
        use_nli=False
    )
    
    # Create streaming generator
    generator = create_streaming_generator(
        llm_client=llm_client,
        citation_formatter=citation_formatter,
        safety_checker=safety_checker
    )
    
    # Mock retrieved context (simulating retrieval results)
    context_chunks = [
        {
            'child_id': 'sample_elem_35_table_child_0',
            'parent_id': 'sample_elem_35',
            'source_file': 'data/inputs/sample.pdf',
            'page_number': 5,
            'parent_type': 'table',
            'parent_text': 'PC8110 PVD 코팅은 M10~M20 S10~S20 범위에서 스테인레스강 및 내열합금강의 고속가공에 적합합니다.'
        },
        {
            'child_id': 'sample_elem_116_table_child_0',
            'parent_id': 'sample_elem_116',
            'source_file': 'data/inputs/sample.pdf',
            'page_number': 12,
            'parent_type': 'table',
            'parent_text': 'PVD 코팅의 절삭속도는 오스테나이트계 스테인레스강(STS304)에서 150~280 m/min입니다.'
        }
    ]
    
    # Test query
    query = "스테인레스강 고속 가공에 적합한 코팅은?"
    
    print(f"\n Query: {query}")
    print(f" Context chunks: {len(context_chunks)}")
    print(f"\n Streaming events:\n")
    print("-" * 70)
    
    # Track events
    event_counts = {}
    full_answer = ""
    citations_received = []
    
    try:
        # Stream generation
        async for event in generator.generate_with_citations(
            query=query,
            context_chunks=context_chunks,
            language="ko"
        ):
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Display event
            if event_type == EventType.STATUS:
                print(f" STATUS: {event['content']}")
            
            elif event_type == EventType.CITATION:
                citation = event['data']
                citations_received.append(citation)
                print(f" CITATION: {citation['id']} → {citation['file']}, page {citation['page']}")
            
            elif event_type == EventType.CHUNK:
                content = event['content']
                full_answer += content
                print(content, end='', flush=True)
            
            elif event_type == EventType.CITATION_MARKER:
                marker = event['id']
                print(f" {marker}", end='', flush=True)
            
            elif event_type == EventType.DONE:
                metadata = event['metadata']
                print(f"\n\n DONE:")
                print(f"   Latency: {metadata['latency_ms']}ms")
                print(f"   Answer length: {metadata['answer_length']} chars")
                print(f"   Citations: {metadata['citation_count']}")
                print(f"   Safety passed: {metadata['safety_passed']}")
            
            elif event_type == EventType.ERROR:
                print(f"\n ERROR: {event['data']['message']}")
        
        print("-" * 70)
        
        # Summary
        print(f"\n Event Summary:")
        for event_type, count in event_counts.items():
            print(f"   {event_type}: {count}")
        
        print(f"\n Full Answer ({len(full_answer)} chars):")
        print(f"   {full_answer}")
        
        print(f"\n Citations Received: {len(citations_received)}")
        for cit in citations_received:
            print(f"   {cit['id']} → {cit['file']}, p.{cit['page']}")
        
        print("\n" + "="*70)
        print(" Test completed successfully!")
        print("="*70)
    
    finally:
        await llm_client.close()


async def main():
    """Run test"""
    try:
        await test_streaming_generation()
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())