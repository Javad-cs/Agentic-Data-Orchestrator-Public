# scripts/clean_milvus.py

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.db.milvus_schema import create_milvus_collection
from pymilvus import connections


def main():
    print(" Starting Milvus cleanup...")
    
    # Drop existing and recreate collection
    create_milvus_collection(
        collection_name="rag_children", 
        drop_existing=True  # This wipes old data
    )
    
    print(" Milvus cleaned and recreated successfully!")
    
    # Verify
    from pymilvus import Collection
    collection = Collection("rag_children")
    collection.load()
    print(f" Current vector count: {collection.num_entities}")
    
    # Cleanup connection
    connections.disconnect("default")
    print(" Disconnected from Milvus")


if __name__ == "__main__":
    main()