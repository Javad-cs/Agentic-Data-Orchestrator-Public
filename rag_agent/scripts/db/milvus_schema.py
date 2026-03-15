from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from urllib.parse import urlparse
from typing import Optional
import sys


def parse_milvus_uri(uri: str) -> tuple[str, str]:
    """
    Robustly parse Milvus URI into host and port.
    
    Handles:
    - http://localhost:19530
    - localhost:19530
    - milvus-standalone:19530
    - http://milvus:8080
    
    Returns:
        (host, port) tuple
    """
    # Add scheme if missing
    if not uri.startswith(('http://', 'https://')):
        uri = f'http://{uri}'
    
    parsed = urlparse(uri)
    
    host = parsed.hostname or 'localhost'
    port = str(parsed.port) if parsed.port else '19530'
    
    return host, port


def create_milvus_collection(
    collection_name: str = "rag_children",
    dimension: int = 4096,
    milvus_uri: str = "http://localhost:19530",
    drop_existing: bool = False,
    index_type: str = "HNSW"  # Changed default from IVF_FLAT
):
    """
    Create Milvus collection for dense vector search.
    
    Args:
        collection_name: Name of collection
        dimension: Vector dimension (4096 for solar-embedding-1-large)
        milvus_uri: Milvus connection URI
        drop_existing: If True, drop existing collection
        index_type: Index type ('HNSW' or 'IVF_FLAT')
            - HNSW: Faster queries, more memory (recommended for RAG)
            - IVF_FLAT: Slower queries, less memory (use for 10M+ vectors)
    """
    # Parse URI robustly
    host, port = parse_milvus_uri(milvus_uri)
    
    print(f" Connecting to Milvus at {host}:{port}...")
    
    try:
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
    except Exception as e:
        print(f" Failed to connect to Milvus: {e}")
        print(f"   Make sure Milvus is running at {host}:{port}")
        return None
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        if drop_existing:
            print(f"️  Dropping existing collection '{collection_name}'...")
            utility.drop_collection(collection_name)
        else:
            print(f" Collection '{collection_name}' already exists")
            collection = Collection(collection_name)
            print(f"   Total entities: {collection.num_entities}")
            return collection
    
    print(f" Creating collection '{collection_name}'...")
    
    # Define schema
    fields = [
        FieldSchema(
            name="child_id",
            dtype=DataType.VARCHAR,
            max_length=100,
            is_primary=True,
            description="Unique child chunk ID (matches PostgreSQL children.child_id)"
        ),
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=dimension,
            description="Dense embedding from Upstage solar-embedding-1-large API"
        ),
        FieldSchema(
            name="parent_id",
            dtype=DataType.VARCHAR,
            max_length=100,
            description="Reference to parent chunk (matches PostgreSQL parents.parent_id)"
        ),
        FieldSchema(
            name="source_file",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="Source document file path"
        ),
        FieldSchema(
            name="page_number",
            dtype=DataType.INT32,
            description="Page number in source document"
        ),
        FieldSchema(
            name="chunk_type",
            dtype=DataType.VARCHAR,
            max_length=20,
            description="Type: text_chunk or table_row_group"
        )
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="RAG children chunks with dense embeddings from Upstage API",
        enable_dynamic_field=True  # Future-proofing: allow adding fields later
    )
    
    # Create collection
    collection = Collection(
        name=collection_name,
        schema=schema
    )
    
    print(f" Creating {index_type} index on dense_vector...")
    
    # Index parameters based on type
    if index_type.upper() == "HNSW":
        # HNSW: Faster queries, recommended for RAG agents
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,  # Max number of connections per layer (8-64, 16 is balanced)
                "efConstruction": 200  # Build time/quality tradeoff (100-500)
            }
        }
        print(f"   Using HNSW (M=16, efConstruction=200)")
        print(f"    Optimized for low latency queries")
        
    elif index_type.upper() == "IVF_FLAT":
        # IVF_FLAT: Lower memory, good for large datasets
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 1024  # Number of clusters (sqrt(N) to 4*sqrt(N))
            }
        }
        print(f"   Using IVF_FLAT (nlist=1024)")
        print(f"    Optimized for memory efficiency")
        
    else:
        raise ValueError(f"Unknown index_type: {index_type}. Use 'HNSW' or 'IVF_FLAT'")
    
    collection.create_index(
        field_name="dense_vector",
        index_params=index_params
    )
    
    print(f" Collection '{collection_name}' created successfully!")
    print(f"   - Dimension: {dimension}")
    print(f"   - Metric: COSINE similarity")
    print(f"   - Index: {index_type}")
    print(f"   - Schema alignment:  Matches PostgreSQL")
    
    # Load collection into memory
    print(f" Loading collection into memory...")
    collection.load()
    print(f" Collection loaded and ready for queries")
    
    return collection


def get_collection_info(
    collection_name: str = "rag_children",
    milvus_uri: str = "http://localhost:19530"
):
    """Get detailed information about a collection"""
    host, port = parse_milvus_uri(milvus_uri)
    
    try:
        connections.connect(alias="default", host=host, port=port)
    except Exception as e:
        print(f" Failed to connect: {e}")
        return
    
    if not utility.has_collection(collection_name):
        print(f" Collection '{collection_name}' does not exist")
        print(f"\nAvailable collections:")
        for name in utility.list_collections():
            print(f"   - {name}")
        return
    
    collection = Collection(collection_name)
    
    print(f"\n{'='*60}")
    print(f" Collection: {collection_name}")
    print(f"{'='*60}")
    print(f"Total entities: {collection.num_entities:,}")
    print(f"\nSchema:")
    for field in collection.schema.fields:
        field_info = f"   - {field.name} ({field.dtype})"
        if field.dtype == DataType.FLOAT_VECTOR:
            field_info += f" [dim={field.params.get('dim', 'N/A')}]"
        elif field.dtype == DataType.VARCHAR:
            field_info += f" [max_length={field.params.get('max_length', 'N/A')}]"
        if field.is_primary:
            field_info += " [PRIMARY KEY]"
        print(field_info)
    
    print(f"\nIndexes:")
    for index in collection.indexes:
        print(f"   - Field: {index.field_name}")
        print(f"     Type: {index.params.get('index_type', 'N/A')}")
        print(f"     Metric: {index.params.get('metric_type', 'N/A')}")
        if 'params' in index.params:
            print(f"     Params: {index.params['params']}")
    
    # Check load status using utility
    try:
        load_state = utility.load_state(collection_name)
        status = "Loaded " if load_state == utility.LoadState.Loaded else f"Not loaded ({load_state.name}) ️"
    except:
        status = "Unknown"

    print(f"\nStatus: {status}")
    print(f"{'='*60}\n")


def drop_collection(
    collection_name: str = "rag_children",
    milvus_uri: str = "http://localhost:19530"
):
    """Drop a collection"""
    host, port = parse_milvus_uri(milvus_uri)
    
    try:
        connections.connect(alias="default", host=host, port=port)
    except Exception as e:
        print(f" Failed to connect: {e}")
        return
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f" Collection '{collection_name}' dropped successfully")
    else:
        print(f"️  Collection '{collection_name}' does not exist")


def test_connection(milvus_uri: str = "http://localhost:19530"):
    """Test connection to Milvus"""
    host, port = parse_milvus_uri(milvus_uri)
    
    print(f" Testing connection to {host}:{port}...")
    
    try:
        connections.connect(alias="default", host=host, port=port)
        print(f" Connection successful!")
        
        # List collections
        collections = utility.list_collections()
        print(f"\n Available collections ({len(collections)}):")
        for name in collections:
            coll = Collection(name)
            print(f"   - {name} ({coll.num_entities:,} entities)")
        
        connections.disconnect("default")
        return True
        
    except Exception as e:
        print(f" Connection failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"   1. Is Milvus running? Check with: docker ps | grep milvus")
        print(f"   2. Is the URI correct? Current: {milvus_uri}")
        print(f"   3. Is port {port} accessible?")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Milvus collection management for RAG system"
    )
    parser.add_argument(
        "--action",
        choices=["create", "info", "drop", "test"],
        default="create",
        help="Action to perform"
    )
    parser.add_argument(
        "--collection",
        default="rag_children",
        help="Collection name"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=4096,
        help="Vector dimension (4096 for solar-embedding-1-large)"
    )
    parser.add_argument(
        "--uri",
        default="http://localhost:19530",
        help="Milvus URI (e.g., http://localhost:19530)"
    )
    parser.add_argument(
        "--index-type",
        choices=["HNSW", "IVF_FLAT"],
        default="HNSW",
        help="Index type (HNSW=fast queries, IVF_FLAT=low memory)"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing collection before creating"
    )
    
    args = parser.parse_args()
    
    if args.action == "create":
        create_milvus_collection(
            collection_name=args.collection,
            dimension=args.dimension,
            milvus_uri=args.uri,
            drop_existing=args.drop_existing,
            index_type=args.index_type
        )
    elif args.action == "info":
        get_collection_info(args.collection, args.uri)
    elif args.action == "drop":
        drop_collection(args.collection, args.uri)
    elif args.action == "test":
        test_connection(args.uri)