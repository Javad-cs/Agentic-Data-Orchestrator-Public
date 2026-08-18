#!/usr/bin/env python3
"""
Test Slow Lane agent with complex queries.

Usage:
    python scripts/test_slow_lane.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.agents.fast_lane import create_fast_lane
from src.agents.slow_lane import create_slow_lane
from src.utils.logging_config import setup_logging

setup_logging()

async def main():
    print("=" * 70)
    print("SLOW LANE TEST")
    print("=" * 70)
    print()
    
    # Load config
    print(" Loading configuration...")
    config = SystemConfig()
    print(" Configuration loaded")
    print()
    
    fast_lane = None
    slow_lane = None
    
    try:
        # Initialize Fast Lane (needed as tool)
        print(" Initializing Fast Lane...")
        fast_lane = await create_fast_lane(config)
        print(" Fast Lane ready")
        print()
        
        # Initialize Slow Lane
        print(" Initializing Slow Lane...")
        slow_lane = await create_slow_lane(config, fast_lane)
        print(" Slow Lane ready")
        print()
        
        # Test complex query
        query = "PVD 코팅 공구의 권장 절삭 속도와 이송 조건은?"
        
        print(f"Query: {query}")
        print()
        print("-" * 70)
        
        async for event in slow_lane.query(query, language="ko", streaming=True):
            event_type = event.get("type")
            
            if event_type == "status":
                print(f"   {event.get('content')}")
            
            elif event_type == "chunk":
                print(event.get("content"), end="", flush=True)
            
            elif event_type == "done":
                metadata = event.get("metadata", {})
                print()
                print()
                print(f"   Steps: {metadata.get('steps_taken')}")
                print(f"   Facts: {metadata.get('facts_found')}")
                print(f"   Sources: {metadata.get('sources_used')}")
            
            elif event_type == "error":
                print(f"    Error: {event.get('data', {}).get('message')}")
        
        print()
        print("=" * 70)
    
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        #  CRITICAL: Always cleanup, even on error
        print("\n" + "=" * 70)
        print("CLEANUP")
        print("=" * 70)
        
        if slow_lane:
            print(" Closing Slow Lane...")
            try:
                await slow_lane.close()
                print(" Slow Lane closed")
            except Exception as e:
                print(f"️  Slow Lane cleanup error: {e}")
        
        if fast_lane:
            print(" Closing Fast Lane...")
            try:
                await fast_lane.close()
                print(" Fast Lane closed")
            except Exception as e:
                print(f"️  Fast Lane cleanup error: {e}")
        
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())