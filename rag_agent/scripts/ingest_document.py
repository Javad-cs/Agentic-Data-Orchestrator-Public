import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.ingestion.pipeline import IngestionPipeline


async def main():
    """Ingest a document from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest a document into RAG system")
    parser.add_argument("file", help="Path to document file")
    parser.add_argument("--acl-users", nargs="*", help="User IDs with access")
    parser.add_argument("--acl-groups", nargs="*", help="Group IDs with access")
    
    args = parser.parse_args()
    
    # Validate file
    if not Path(args.file).exists():
        print(f" File not found: {args.file}")
        sys.exit(1)
    
    # Load config
    try:
        config = SystemConfig()
    except Exception as e:
        print(f" Failed to load config: {e}")
        sys.exit(1)
    
    # Create pipeline
    pipeline = IngestionPipeline(config)
    
    try:
        # Initialize
        print(" Initializing pipeline...")
        await pipeline.initialize()
        
        # Prepare metadata
        metadata = {}
        if args.acl_users:
            metadata['acl_users'] = args.acl_users
        if args.acl_groups:
            metadata['acl_groups'] = args.acl_groups
        
        # Ingest
        result = await pipeline.ingest_document(args.file, metadata)
        
        # Print results
        print("\n" + "="*60)
        print(" INGESTION RESULTS")
        print("="*60)
        print(f"Status: {result.status}")
        print(f"Job ID: {result.job_id}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"\nStatistics:")
        print(f"  Parents: {result.total_parents}")
        print(f"  Children: {result.total_children}")
        print(f"  Embedded: {result.total_embedded}")
        print(f"  Indexed (BM25): {result.total_indexed}")
        
        if result.errors:
            print(f"\n Errors:")
            for error in result.errors:
                print(f"  - {error}")
        else:
            print("\n Success!")
        
        print("="*60)
    
    finally:
        # Cleanup
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())