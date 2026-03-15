"""
Test RAG tool independently.

Run from project root:
    cd Agentic-Data-Orchestrator/
    docker compose exec -e USE_DOCKER=false agent-env python3 -m router_agent.scripts.test_rag_tool
"""
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load rag_agent's .env file
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / "rag_agent" / ".env")

from router_agent.src.tools.rag_tool import RAGTool


async def main():
    print("=" * 70)
    print("RAG TOOL TEST")
    print("=" * 70)
    print()
    
    # Test 1: Initialize tool
    print(" Test 1: Initializing RAG tool...")
    try:
        tool = RAGTool()
        print(" RAG tool initialized")
        print(f"   Name: {tool.name()}")
        print(f"   Description: {tool.description()[:80]}...")
        print()
    except Exception as e:
        print(f" Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Simple query
    print(" Test 2: Testing simple query...")
    test_query = "What is PVD coating?"
    
    try:
        print(f"Query: {test_query}")
        result = await tool.run(test_query)
        
        if result.success:
            print(" Query succeeded")
            print(f"   Answer length: {len(result.answer)} chars")
            print(f"   Answer preview: {result.answer[:200]}...")
            print(f"   Citations: {result.metadata.get('citations', [])}")
            print(f"   Chunks retrieved: {result.metadata.get('chunks_retrieved', 0)}")
            print(f"   Confidence: {result.metadata.get('confidence', 'N/A')}")
        else:
            print(f" Query failed: {result.error}")
        print()
    except Exception as e:
        print(f" Query error: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test 3: Korean query (if applicable)
    print(" Test 3: Testing Korean query...")
    korean_query = "스테인레스강 가공 시 권장 절삭 속도는?"
    
    try:
        print(f"Query: {korean_query}")
        result = await tool.run(korean_query)
        
        if result.success:
            print(" Query succeeded")
            print(f"   Answer length: {len(result.answer)} chars")
            print(f"   Answer preview: {result.answer[:200]}...")
            print(f"   Citations: {len(result.metadata.get('citations', []))}")
            print(f"   Chunks retrieved: {result.metadata.get('chunks_retrieved', 0)}")
        else:
            print(f" Query failed: {result.error}")
        print()
    except Exception as e:
        print(f" Query error: {e}")
        print()
    
    # Test 4: Empty query (error handling)
    print(" Test 4: Testing error handling (empty query)...")
    try:
        result = await tool.run("")
        
        if not result.success:
            print(" Correctly handled empty query")
            print(f"   Error: {result.error}")
        else:
            print("  Empty query didn't fail as expected")
        print()
    except Exception as e:
        print(f" Exception caught: {e}")
        print()
    
    print("=" * 70)
    print("RAG TOOL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())