"""
Test SQL tool independently.

Run from project root:
    cd Agentic-Data-Orchestrator/
    docker compose exec -e USE_DOCKER=false agent-env python3 -m router_agent.scripts.test_sql_tool
"""
import asyncio
from pathlib import Path

from router_agent.src.tools.sql_tool import SQLTool
from router_agent.src.config.settings import get_settings


async def main():
    print("=" * 70)
    print("SQL TOOL TEST")
    print("=" * 70)
    print()
    
    # Get settings
    settings = get_settings()
    
    # Test 1: Initialize tool
    print(f" Test 1: Initializing SQL tool (db={settings.PRIMARY_DB_NAME})...")
    print("  Note: First initialization takes 10-60s (profiling + indexing)")
    print()
    
    try:
        tool = SQLTool()
        print(" SQL tool created (lazy init)")
        print(f"   Name: {tool.name()}")
        print(f"   Description: {tool.description()[:80]}...")
        print()
    except Exception as e:
        print(f" Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Simple aggregation query
    print(" Test 2: Testing simple query...")
    test_query = "지난주 연삭1반 레오페리 6호기 가동률 알려줘"
    
    try:
        print(f"Query: {test_query}")
        print(" Running query (first run triggers lazy init)...")
        
        import time
        start = time.time()
        result = await tool.run(test_query)
        elapsed = time.time() - start
        
        if result.success:
            print(f" Query succeeded ({elapsed:.1f}s)")
            print(f"   Answer: {result.answer}")
            print(f"   SQL: {result.metadata.get('sql_query', 'N/A')[:100]}...")
            print(f"   Rows returned: {result.metadata.get('rows_returned', 0)}")
        else:
            print(f" Query failed: {result.error}")
        print()
    except Exception as e:
        print(f" Query error: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test 3: Invalid query (error handling)
    print(" Test 3: Testing error handling (nonsense query)...")
    try:
        result = await tool.run("xyzabc nonsense query 12345")
        
        if not result.success:
            print(" Correctly handled invalid query")
            print(f"   Error: {result.error[:100]}...")
        else:
            print("  Invalid query didn't fail as expected")
            print(f"   Answer: {result.answer}")
        print()
    except Exception as e:
        print(f" Exception caught: {e}")
        print()
    
    print("=" * 70)
    print("SQL TOOL TEST COMPLETE")
    print("=" * 70)
    print()
    print(" Note: SQL tool is now initialized. Subsequent queries will be faster.")


if __name__ == "__main__":
    asyncio.run(main())