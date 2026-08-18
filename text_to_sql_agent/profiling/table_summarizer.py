"""
LLM-based summarization of table profiles.
Generates grounded table descriptions from column descriptions.
Includes disk caching to avoid re-summarizing tables.
"""

from typing import Optional, Dict, List
import json
import hashlib
from pathlib import Path
from core import LLMMessage, BaseLLMClient, create_llm_client
from .table_statistics import TableProfile
from .field_metadata import FieldMetadata


class TableMetadata:
    """
    Metadata for a table with LLM-generated description.
    
    Similar to FieldMetadata but for tables.
    """
    
    def __init__(
        self,
        profile: TableProfile,
        description: str,
        column_summaries: Dict[str, str]
    ):
        """
        Initialize table metadata.
        
        Args:
            profile: Table profile with statistics
            description: LLM-generated table description
            column_summaries: Map of column_name -> short_description
        """
        self.profile = profile
        self.description = description
        self.column_summaries = column_summaries
    
    def __repr__(self):
        return f"TableMetadata({self.profile.table_name}, {len(self.column_summaries)} columns)"


class TableSummarizer:
    """
    Generates natural language table descriptions from column descriptions.
    
    Grounding strategy: Aggregates existing FieldMetadata descriptions
    rather than generating from scratch.
    """
    
    def __init__(
        self, 
        llm_client: Optional[BaseLLMClient] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Initialize table summarizer.
        
        Args:
            llm_client: LLM client to use (creates default if None)
            cache_dir: Directory for caching summaries (default: .cache/table_summaries)
            use_cache: Whether to use caching
        """
        self.llm = llm_client or create_llm_client()
        self.use_cache = use_cache
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "text_to_sql" / "table_summaries"
        
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, table_profile: TableProfile) -> str:
        """
        Generate cache key from table profile.
        
        Uses table name and column names to create unique hash.
        """
        key_data = {
            "table": table_profile.table_name,
            "columns": table_profile.get_column_names(),
            "total_rows": table_profile.total_rows
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached summary if exists."""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save summary to cache."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to cache table summary: {e}")
    
    def summarize(
        self, 
        table_profile: TableProfile,
        field_metadata_list: List[FieldMetadata]
    ) -> TableMetadata:
        """
        Generate table description from column descriptions.
        
        GROUNDING: Uses existing FieldMetadata.short_description for each column.
        LLM only aggregates/summarizes - doesn't generate new facts.
        
        Args:
            table_profile: Table profile with statistics
            field_metadata_list: Pre-computed field metadata with descriptions
            
        Returns:
            TableMetadata with LLM-generated table description
        """
        # Check cache first
        cache_key = self._get_cache_key(table_profile)
        cached = self._load_from_cache(cache_key)
        
        if cached:
            # Rebuild column_summaries dict from cache
            column_summaries = {}
            for field_meta in field_metadata_list:
                col_name = field_meta.profile.column_name
                column_summaries[col_name] = field_meta.short_description or ""
            
            return TableMetadata(
                profile=table_profile,
                description=cached.get("description", ""),
                column_summaries=column_summaries
            )
        
        # Build column summaries dict
        column_summaries = {}
        for field_meta in field_metadata_list:
            col_name = field_meta.profile.column_name
            column_summaries[col_name] = field_meta.short_description or ""
        
        # Generate table description
        description = self._generate_table_description(
            table_profile,
            field_metadata_list
        )
        
        # Cache the result
        cache_data = {
            "description": description,
            "table_name": table_profile.table_name
        }
        self._save_to_cache(cache_key, cache_data)
        
        return TableMetadata(
            profile=table_profile,
            description=description,
            column_summaries=column_summaries
        )
    
    def _generate_table_description(
        self,
        table_profile: TableProfile,
        field_metadata_list: List[FieldMetadata]
    ) -> str:
        """
        Generate ONE sentence table description from column descriptions.
        
        Grounded approach: LLM only summarizes existing column descriptions.
        """
        # Build context from column descriptions
        column_descriptions = []
        for field_meta in field_metadata_list:
            col_name = field_meta.profile.column_name
            col_desc = field_meta.short_description or f"{col_name} column"
            column_descriptions.append(f"- {col_name}: {col_desc}")
        
        column_context = "\n".join(column_descriptions)
        
        user_prompt = f"""You are a database expert. Given these column descriptions for the table "{table_profile.table_name}", write ONE clear sentence summarizing what this table stores.

Column Descriptions:
{column_context}

RULES:
1. Write EXACTLY ONE sentence
2. Start with: "Stores..." or "Contains..." or "Records..."
3. Prioritize information from the column descriptions
4. The table name "{table_profile.table_name}" can provide semantic hints if column descriptions are generic
5. Be specific about what data is in the table
6. Mention key data types if relevant (e.g., "including numeric compensation data")
7. Do NOT invent domain knowledge beyond what's in the columns and table name

Table Description:"""
        
        messages = [
            LLMMessage(role="user", content=user_prompt)
        ]
        
        # Retry with increasing token limits
        token_limits = [200, 300, 500]
        
        for max_tokens in token_limits:
            try:
                response = self.llm.generate(messages, max_tokens=max_tokens)
                description = response.content.strip() if response.content else ""
                
                # If we got content, return it
                if description:
                    return description
                
                # If finish_reason is 'length', try again with more tokens
                if response.finish_reason == 'length':
                    print(f"[RETRY] Table description hit token limit ({max_tokens}), retrying with more...")
                    continue
                else:
                    # Some other issue, don't retry
                    break
                    
            except Exception as e:
                print(f"[ERROR] Table description generation failed for {table_profile.table_name}: {e}")
                break
        
        # All retries failed - use fallback
        print(f"[WARN] Empty table description for {table_profile.table_name}, using fallback")
        key_columns = table_profile.get_column_names()[:3]
        return f"Stores data with columns: {', '.join(key_columns)}"

    
    async def summarize_async(
        self,
        table_profile: TableProfile,
        field_metadata_list: List[FieldMetadata]
    ) -> TableMetadata:
        """Async version of summarize."""
        
        # Check cache first
        cache_key = self._get_cache_key(table_profile)
        cached = self._load_from_cache(cache_key)
        
        if cached:
            # Rebuild column_summaries dict from cache
            column_summaries = {}
            for field_meta in field_metadata_list:
                col_name = field_meta.profile.column_name
                column_summaries[col_name] = field_meta.short_description or ""
            
            return TableMetadata(
                profile=table_profile,
                description=cached.get("description", ""),
                column_summaries=column_summaries
            )
        
        # Build column summaries
        column_summaries = {}
        for field_meta in field_metadata_list:
            col_name = field_meta.profile.column_name
            column_summaries[col_name] = field_meta.short_description or ""
        
        # Generate description asynchronously
        description = await self._generate_table_description_async(
            table_profile,
            field_metadata_list
        )
        
        # Cache the result
        cache_data = {
            "description": description,
            "table_name": table_profile.table_name
        }
        self._save_to_cache(cache_key, cache_data)
        
        return TableMetadata(
            profile=table_profile,
            description=description,
            column_summaries=column_summaries
        )
    
    async def _generate_table_description_async(
        self,
        table_profile: TableProfile,
        field_metadata_list: List[FieldMetadata]
    ) -> str:
        """Async version of table description generation."""
        
        # Build context
        column_descriptions = []
        for field_meta in field_metadata_list:
            col_name = field_meta.profile.column_name
            col_desc = field_meta.short_description or f"{col_name} column"
            column_descriptions.append(f"- {col_name}: {col_desc}")
        
        column_context = "\n".join(column_descriptions)
        
        user_prompt = f"""You are a database expert. Given these column descriptions for the table "{table_profile.table_name}", write ONE clear sentence summarizing what this table stores.

Column Descriptions:
{column_context}

RULES:
1. Write EXACTLY ONE sentence
2. Start with: "Stores..." or "Contains..." or "Records..."
3. Prioritize information from the column descriptions
4. The table name "{table_profile.table_name}" can provide semantic hints if column descriptions are generic
5. Be specific about what data is in the table
6. Mention key data types if relevant (e.g., "including numeric compensation data")
7. Do NOT invent domain knowledge beyond what's in the columns and table name

Table Description:"""
        
        messages = [
            LLMMessage(role="user", content=user_prompt)
        ]
        
        # Retry with increasing token limits
        token_limits = [200, 300, 500]
        
        for max_tokens in token_limits:
            try:
                response = await self.llm.generate_async(messages, max_tokens=max_tokens)
                description = response.content.strip() if response.content else ""
                
                # If we got content, return it
                if description:
                    return description
                
                # If finish_reason is 'length', try again with more tokens
                if response.finish_reason == 'length':
                    print(f"[RETRY ASYNC] Table description hit token limit ({max_tokens}), retrying with more...")
                    continue
                else:
                    # Some other issue, don't retry
                    break
                    
            except Exception as e:
                print(f"[ERROR] Async table description failed for {table_profile.table_name}: {e}")
                break
        
        # All retries failed - use fallback
        print(f"[WARN] Async empty table description for {table_profile.table_name}, using fallback")
        key_columns = table_profile.get_column_names()[:3]
        return f"Stores data with columns: {', '.join(key_columns)}"
