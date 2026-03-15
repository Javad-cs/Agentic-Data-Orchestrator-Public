import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from scripts.db.milvus_schema import create_milvus_collection
from pymilvus import connections
import asyncpg


async def clean_postgres():
    """Clean all ingestion data from PostgreSQL"""
    print(" Cleaning PostgreSQL...")
    
    config = SystemConfig()
    
    conn = await asyncpg.connect(config.database.postgres_dsn)
    
    try:
        # Delete in correct order (respects foreign keys)
        await conn.execute("DELETE FROM bm25_index")
        await conn.execute("DELETE FROM children")
        await conn.execute("DELETE FROM parents")
        await conn.execute("DELETE FROM ingestion_log")
        await conn.execute("DELETE FROM bm25_df")
        await conn.execute("UPDATE bm25_stats SET stat_value = 0")
        
        print(" PostgreSQL cleaned!")
    finally:
        await conn.close()


def clean_milvus():
    """Clean and recreate Milvus collection"""
    print(" Cleaning Milvus...")
    
    create_milvus_collection(
        collection_name="rag_children",
        drop_existing=True
    )
    
    # Verify
    from pymilvus import Collection
    collection = Collection("rag_children")
    collection.load()
    print(f" Milvus cleaned! Vectors: {collection.num_entities}")
    
    connections.disconnect("default")


async def main():
    print("=" * 60)
    print("  CLEANING ALL DATABASES")
    print("=" * 60)
    
    # Clean PostgreSQL
    await clean_postgres()
    
    # Clean Milvus
    clean_milvus()
    
    print("=" * 60)
    print(" All databases cleaned!")
    print("=" * 60)
    print("\nReady for fresh ingestion:")
    print("  python scripts/ingest_document.py data/inputs/sample.pdf")


if __name__ == "__main__":
    asyncio.run(main())