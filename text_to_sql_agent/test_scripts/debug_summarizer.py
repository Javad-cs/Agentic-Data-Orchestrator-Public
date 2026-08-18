"""Debug the summarizer to see what LLM actually returns."""

from test_database import create_sample_db
from core import connect_database, create_llm_client, LLMMessage
from profiling import ColumnProfiler


def debug_llm_response():
    """Test what the LLM is actually returning."""
    print(" Debugging LLM Response...\n")
    
    # Create sample database
    db_path = create_sample_db()
    
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler()
        llm = create_llm_client()
        
        # Profile the 'id' column
        profile = profiler.profile_column(db, "users", "id", "INTEGER")
        
        # Build context manually
        context = f"""Table: {profile.table_name}
Column: {profile.column_name}
Data Type: {profile.data_type}
Total Records: {profile.total_records}
NULL Values: {profile.null_count}
Distinct Values: {profile.distinct_count}
Range: {profile.min_value} to {profile.max_value}
Most Common Values: {', '.join(str(v) for v, _ in profile.top_k_values[:5])}"""
        
        print(" Context being sent to LLM:")
        print(context)
        print("\n" + "="*60 + "\n")
        
        # Try simple prompt
        user_prompt = f"""You are analyzing a database column. Based on the statistics below, write a brief 1-2 sentence description.

Column Statistics:
{context}

Description:"""
        
        print(" Prompt being sent:")
        print(user_prompt)
        print("\n" + "="*60 + "\n")
        
        messages = [
            LLMMessage(role="user", content=user_prompt),
        ]
        
        print(" Calling LLM...")
        response = llm.generate(messages, max_tokens=200)
        
        print(f"\n Raw response object:")
        print(f"   Type: {type(response)}")
        print(f"   Content type: {type(response.content)}")
        print(f"   Content length: {len(response.content)}")
        print(f"   Content repr: {repr(response.content)}")
        print(f"   Model: {response.model}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Finish reason: {response.finish_reason}")
        
        print(f"\n Content:")
        print(f"'{response.content}'")
        
        print(f"\n After strip:")
        print(f"'{response.content.strip()}'")
    
    db_path.unlink()


if __name__ == "__main__":
    debug_llm_response()