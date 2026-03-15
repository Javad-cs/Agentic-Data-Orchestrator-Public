"""
End-to-end test for Slow Lane with multi-tool integration.

Tests SQL + RAG on manufacturing domain:
- SQL: PRIMARY_DATABASE/LINKED_DATABASE Oracle databases
- RAG: Manufacturing documentation (Korean)

Usage:
    docker compose exec -e USE_DOCKER=false agent-env python3 -m router_agent.scripts.test_slow_lane_multi_tool
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add paths
_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[2]

sys.path.insert(0, str(_project_root / "router_agent" / "src"))
sys.path.insert(0, str(_project_root / "rag_agent" / "src"))

# Imports from router_agent
from tools.rag_tool import RAGTool
from tools.sql_tool import SQLTool
from tools.langgraph_tools import LangGraphToolWrapper

# Imports from rag_agent
from src.config.models import SystemConfig
from src.agents.slow_lane import SlowLane

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_table_context_retrieval():
    """Test 1: Table context retrieval works"""
    print("\n" + "="*70)
    print("TEST 1: Table Context Retrieval")
    print("="*70)
    
    # Initialize SQL tool (no db_name - uses Oracle from settings)
    sql_tool = SQLTool()
    
    # Trigger initialization
    print("Initializing SQL tool (builds indices and table metadata)...\n")
    await sql_tool._ensure_initialized()
    
    # Test table context retrieval with Korean manufacturing query
    query = "연삭1반 레오페리 6호기 가동률"
    
    print(f"Query: {query}\n")
    print("Retrieving table context...\n")
    
    table_context = await sql_tool.get_table_context(query, max_tables=3)
    
    print(table_context)
    
    if table_context and "AVAILABLE SQL TABLES" in table_context:
        print("\n Test 1 PASSED: Table context retrieved successfully")
    else:
        print("\n Test 1 FAILED: Table context retrieval failed")


async def test_json_parsing_robustness():
    """Test 2: JSON parsing handles malformed responses"""
    print("\n" + "="*70)
    print("TEST 2: JSON Parsing Robustness")
    print("="*70)
    
    from src.agents.slow_lane import SlowLane
    
    config = SystemConfig()
    
    sql_tool = SQLTool()  # Changed: no db_name
    tools = LangGraphToolWrapper(sql_tool=sql_tool)
    slow_lane = SlowLane(config=config, tools=tools)
    
    # Test cases
    test_cases = [
        # Case 1: Valid JSON
        ('{"plan": [{"step_id": 1, "tool": "sql_tool", "query": "test", "rationale": "test"}]}', True),
        
        # Case 2: JSON with code fence
        ('```json\n{"plan": [{"step_id": 1, "tool": "sql_tool", "query": "test", "rationale": "test"}]}\n```', True),
        
        # Case 3: JSON with explanation before
        ('Here is the plan:\n{"plan": [{"step_id": 1, "tool": "sql_tool", "query": "test", "rationale": "test"}]}', True),
        
        # Case 4: JSON with explanation after
        ('{"plan": [{"step_id": 1, "tool": "sql_tool", "query": "test", "rationale": "test"}]}\nThis plan will work.', True),
        
        # Case 5: Multiple JSON objects (should extract first)
        ('{"debug": true}\n{"plan": [{"step_id": 1, "tool": "sql_tool", "query": "test", "rationale": "test"}]}', False),
        
        # Case 6: Nested braces
        ('{"plan": [{"step_id": 1, "tool": "sql_tool", "query": "What is {field}?", "rationale": "test"}]}', True),
        
        # Case 7: Invalid JSON
        ('This is not JSON at all', False),
    ]
    
    passed = 0
    for i, (response, should_succeed) in enumerate(test_cases, 1):
        plan = slow_lane._parse_json_plan(response)
        
        success = len(plan) > 0
        
        if success == should_succeed:
            print(f"  Case {i}: {'PASS' if should_succeed else 'CORRECTLY FAILED'}")
            passed += 1
        else:
            print(f"  Case {i}: {'UNEXPECTED FAILURE' if should_succeed else 'UNEXPECTED SUCCESS'}")
            print(f"     Response: {response[:80]}...")
            print(f"     Extracted plan: {plan}")
    
    await slow_lane.close()
    
    if passed == len(test_cases):
        print(f"\n Test 2 PASSED: {passed}/{len(test_cases)} cases handled correctly")
    else:
        print(f"\n Test 2 PARTIAL: {passed}/{len(test_cases)} cases passed")

async def test_sql_only_query():
    """Test 3: SQL-only query on manufacturing database"""
    print("\n" + "="*70)
    print("TEST 3: SQL-Only Query (Manufacturing Database)")
    print("="*70)
    
    query = "지난주 연삭1반 레오페리 6호기 가동률 알려줘"
    
    print(f"Query: {query}\n")
    
    # Initialize tools - no db_name needed (uses Oracle from settings)
    sql_tool = SQLTool()
    tools = LangGraphToolWrapper(sql_tool=sql_tool)
    
    config = SystemConfig()
    slow_lane = SlowLane(config=config, tools=tools)
    
    # Execute query
    print("Executing query...\n")
    
    events = []
    async for event in slow_lane.query(query, language="ko", streaming=True):  # Changed to "ko"
        events.append(event)
        
        if event["type"] == "status":
            print(f"[STATUS] {event['content']}")
        elif event["type"] == "chunk":
            print(event["content"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\n\n[DONE] Metadata: {event['metadata']}")
        elif event["type"] == "error":
            print(f"\n[ERROR] {event['data']['message']}")
    
    # Validate
    done_events = [e for e in events if e["type"] == "done"]
    if done_events:
        metadata = done_events[0]["metadata"]
        print(f"\n Test 3 PASSED: {metadata['steps_taken']} steps, {metadata['sources_used']} sources")
    else:
        print("\n Test 3 FAILED: No answer generated")
    
    await slow_lane.close()


async def test_rag_only_query():
    """Test 4: RAG-only query on manufacturing docs"""
    print("\n" + "="*70)
    print("TEST 4: RAG-Only Query (Manufacturing Documents)")
    print("="*70)
    
    query = "PVD 코팅이란 무엇이고 어떤 장점이 있나요?"
    
    print(f"Query: {query}\n")
    
    # Initialize tools
    rag_tool = RAGTool()
    tools = LangGraphToolWrapper(rag_tool=rag_tool)
    
    config = SystemConfig()
    slow_lane = SlowLane(config=config, tools=tools)
    
    # Execute query
    print("Executing query...\n")
    
    events = []
    async for event in slow_lane.query(query, language="ko", streaming=True):  # Changed to "ko"
        events.append(event)
        
        if event["type"] == "status":
            print(f"[STATUS] {event['content']}")
        elif event["type"] == "chunk":
            print(event["content"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\n\n[DONE] Metadata: {event['metadata']}")
        elif event["type"] == "error":
            print(f"\n[ERROR] {event['data']['message']}")
    
    # Validate
    done_events = [e for e in events if e["type"] == "done"]
    if done_events:
        metadata = done_events[0]["metadata"]
        print(f"\n Test 4 PASSED: {metadata['steps_taken']} steps, {metadata['sources_used']} sources")
    else:
        print("\n Test 4 FAILED: No answer generated")
    
    await slow_lane.close()


async def test_multi_tool_query():
    """Test 5: Multi-tool query (SQL + RAG)"""
    print("\n" + "="*70)
    print("TEST 5: Multi-Tool Query (SQL + RAG)")
    print("="*70)
    
    query = "지난주 연삭1반 레오페리 6호기 가동률을 알려주고, PVD 코팅 기술에 대해서도 설명해줘"
    
    print(f"Query: {query}\n")
    print("This query requires:")
    print("  - SQL: Get equipment operation rate from PRIMARY_DATABASE/LINKED_DATABASE")
    print("  - RAG: Get PVD coating information from manufacturing docs")
    print()
    
    # Initialize both tools
    sql_tool = SQLTool()
    rag_tool = RAGTool()
    tools = LangGraphToolWrapper(sql_tool=sql_tool, rag_tool=rag_tool)
    
    config = SystemConfig()
    slow_lane = SlowLane(config=config, tools=tools)
    
    # Execute query
    print("Executing query...\n")
    
    events = []
    async for event in slow_lane.query(query, language="ko", streaming=True):  # Changed to "ko"
        events.append(event)
        
        if event["type"] == "status":
            print(f"[STATUS] {event['content']}")
        elif event["type"] == "chunk":
            print(event["content"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\n\n[DONE] Metadata: {event['metadata']}")
        elif event["type"] == "error":
            print(f"\n[ERROR] {event['data']['message']}")
    
    # Validate
    done_events = [e for e in events if e["type"] == "done"]
    if done_events:
        metadata = done_events[0]["metadata"]
        print(f"\n Test 5 PASSED: {metadata['steps_taken']} steps, {metadata['sources_used']} sources")
        
        # Check if both tools were used
        if metadata['steps_taken'] >= 2:
            print("   Multi-tool orchestration successful!")
    else:
        print("\n Test 5 FAILED: No answer generated")
    
    await slow_lane.close()


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("SLOW LANE MULTI-TOOL INTEGRATION TESTS")
    print("="*70)
    
    # Test 1: Table context (no LLM needed)
    await test_table_context_retrieval()
    
    # Test 2: JSON parsing (no LLM needed)
    await test_json_parsing_robustness()
    
    # Tests 3-5 require LLM
    print("\n  The following tests require LLM access and may incur costs.")
    print("Press Enter to continue or Ctrl+C to skip...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nTests skipped by user.")
        return
    
    # Test 3: SQL-only
    await test_sql_only_query()
    
    # Test 4: RAG-only
    await test_rag_only_query()
    
    # Test 5: Multi-tool
    await test_multi_tool_query()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())