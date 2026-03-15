import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.retrieval.hybrid_retriever import HybridRetriever


async def main():
    """Test hybrid retrieval with and without reranking"""
    # Load config
    config = SystemConfig()
    
    # Create retriever
    retriever = HybridRetriever(config)
    
    try:
        # Initialize
        print(" Initializing retriever...")
        await retriever.initialize()
        
        # Test query
        query = "스테인레스강 고속 가공에 적합한 코팅"
        
        # Test WITHOUT reranking
        print(f"\n{'='*70}")
        print(f" Query: {query}")
        print(f"{'='*70}")
        print("\n WITHOUT RERANKING (RRF only):")
        print("-" * 70)
        
        results_no_rerank = await retriever.retrieve(
            query=query,
            top_k=5,
            rerank=False
        )
        
        for i, result in enumerate(results_no_rerank, 1):
            print(f"\n[{i}] RRF Score: {result['rrf_score']:.4f}")
            print(f"    Dense: {result['dense_score']:.4f} (rank {result['dense_rank']})")
            print(f"    Sparse: {result['sparse_score']:.4f} (rank {result['sparse_rank']})")
            print(f"    Child ID: {result['child_id']}")
            print(f"    Source: {result.get('source_file', 'N/A')}")
            print(f"    Full Text:\n    {result['child_text']}") 
            print(f"    " + "-" * 60)
        
        # Test WITH reranking
        print(f"\n{'='*70}")
        print(" WITH RERANKING (RRF + Cross-encoder):")
        print("-" * 70)
        
        results_with_rerank = await retriever.retrieve(
            query=query,
            top_k=5,
            rerank=True,
            rerank_top_n=20
        )
        
        for i, result in enumerate(results_with_rerank, 1):
            score = result.get('rerank_score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            
            print(f"\n[{i}] Rerank Score: {score_str}")
            print(f"    Original RRF rank: {result.get('original_rrf_rank', 'N/A')}")
            print(f"    RRF Score: {result['rrf_score']:.4f}")
            print(f"    Dense: {result['dense_score']:.4f} (rank {result['dense_rank']})")
            print(f"    Sparse: {result['sparse_score']:.4f} (rank {result['sparse_rank']})")
            print(f"    Child ID: {result['child_id']}")
            print(f"    Full Text:\n    {result['child_text']}") 
            print(f"    " + "-" * 60)
        
        # Compare rankings
        print(f"\n{'='*70}")
        print(" RANKING COMPARISON:")
        print("-" * 70)
        
        for i in range(min(5, len(results_no_rerank))):
            no_rerank_id = results_no_rerank[i]['child_id']
            with_rerank_id = results_with_rerank[i]['child_id']
            
            if no_rerank_id != with_rerank_id:
                # Safely handle scores here too
                score = results_with_rerank[i].get('rerank_score', 0)
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)

                print(f"\nPosition {i+1}:  CHANGED")
                print(f"  Without rerank: {no_rerank_id}")
                print(f"  With rerank:    {with_rerank_id}")
                print(f"  Rerank score:   {score_str}")
            else:
                print(f"\nPosition {i+1}:  SAME ({no_rerank_id})")

    finally:
        await retriever.close()


if __name__ == "__main__":
    asyncio.run(main())