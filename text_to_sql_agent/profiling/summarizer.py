"""
LLM-based summarization of column profiles.
Converts statistical profiles into natural language descriptions.
Includes disk caching to avoid re-summarizing columns.
"""

from typing import Optional, Dict
import json
import hashlib
from pathlib import Path
from core import LLMMessage, BaseLLMClient, create_llm_client
from .statistics import ColumnProfile
from .field_metadata import FieldMetadata  # Use unified FieldMetadata


class ProfileSummarizer:
    """Generates natural language summaries from column profiles using LLM."""
    
    def __init__(
        self, 
        llm_client: Optional[BaseLLMClient] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Initialize summarizer.
        
        Args:
            llm_client: LLM client to use (creates default if None)
            cache_dir: Directory for caching summaries (default: .cache/summaries)
            use_cache: Whether to use caching
        """
        self.llm = llm_client or create_llm_client()
        self.use_cache = use_cache
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "text_to_sql" / "summaries"
        
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, profile: ColumnProfile) -> str:
        """
        Generate cache key from profile.
        
        Uses db_name, table, column, and key statistics to create unique hash.
        """
        # Convert top_values to JSON-serializable format
        top_values_serializable = []
        if profile.top_k_values:
            for val_tuple in profile.top_k_values[:3]:
                if isinstance(val_tuple, (list, tuple)) and len(val_tuple) >= 2:
                    value, count = val_tuple[0], val_tuple[1]
                    # Convert non-serializable types to strings
                    if hasattr(value, 'isoformat'):  # datetime/date
                        value = value.isoformat()
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        value = str(value)
                    top_values_serializable.append([value, count])
        
        key_data = {
            "table": profile.table_name,
            "column": profile.column_name,
            "type": profile.data_type,
            "distinct": profile.distinct_count,
            "null": profile.null_count,
            "total": profile.total_records,
            "top_values": top_values_serializable
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
            # Don't fail if caching fails
            print(f"Warning: Failed to cache summary: {e}")
    
    def summarize(self, profile: ColumnProfile) -> 'FieldMetadata':
        """
        Generate short and long descriptions for a column.
        
        Uses cache if available to avoid redundant LLM calls.
        
        Args:
            profile: Column profile with statistics
            
        Returns:
            FieldMetadata with LLM-generated descriptions
        """
        # Check cache first
        cache_key = self._get_cache_key(profile)
        cached = self._load_from_cache(cache_key)
        
        if cached:
            # Return cached result
            return FieldMetadata(
                profile=profile,
                short_description=cached.get("short_description"),
                long_description=cached.get("long_description"),
                prompt=cached.get("prompt"),
                raw_response=cached.get("raw_response")
            )
        # Build context from profile
        context = self._build_context(profile)
        
        # Generate short description
        short_desc = self._generate_short_description(profile, context)
        
        # Generate long description
        long_desc = self._generate_long_description(profile, context, short_desc)
        
        # Cache the result
        cache_data = {
            "short_description": short_desc,
            "long_description": long_desc,
            "prompt": context,
            "raw_response": None  # Could store if needed
        }
        self._save_to_cache(cache_key, cache_data)
        
        return FieldMetadata(
            profile=profile,
            short_description=short_desc,
            long_description=long_desc,
            prompt=context,
        )
    
    def _build_context(self, profile: ColumnProfile) -> str:
        """Build English-language context from profile statistics."""
        lines = []
        
        lines.append(f"Table: {profile.table_name}")
        lines.append(f"Column: {profile.column_name}")
        lines.append(f"Data Type: {profile.data_type}")
        lines.append(f"Total Records: {profile.total_records}")
        
        # NULL information
        if profile.null_count > 0:
            null_pct = (profile.null_count / profile.total_records) * 100
            lines.append(f"NULL Values: {profile.null_count} ({null_pct:.1f}%)")
        else:
            lines.append("NULL Values: 0 (all values present)")
        
        # Distinct values
        if profile.distinct_count > 0:
            lines.append(f"Distinct Values: {profile.distinct_count}")
            
            # Check if it's likely a key
            if profile.distinct_count == profile.non_null_count:
                lines.append("Note: All non-NULL values are unique (possibly a key)")
        
        # Range information
        if profile.min_value is not None and profile.max_value is not None:
            lines.append(f"Range: {profile.min_value} to {profile.max_value}")
        
        # Length information for strings
        if profile.min_length is not None:
            if profile.min_length == profile.max_length:
                lines.append(f"Length: Always {profile.min_length} characters")
            else:
                lines.append(f"Length: {profile.min_length} to {profile.max_length} characters")
        
        # Pattern detection
        if profile.common_pattern:
            lines.append(f"Common Pattern: {profile.common_pattern}")
        
        # Data characteristics
        characteristics = []
        if profile.is_numeric:
            characteristics.append("numeric")
        if profile.is_date_like:
            characteristics.append("date-like")
        if characteristics:
            lines.append(f"Data Characteristics: {', '.join(characteristics)}")
        
        # Top values
        if profile.top_k_values:
            top_vals = [str(val) for val, _ in profile.top_k_values[:5]]
            lines.append(f"Most Common Values: {', '.join(top_vals)}")
        return "\n".join(lines)
        
    def _generate_short_description(
        self, profile: ColumnProfile, context: str
    ) -> str:
        """Generate a short (1-2 sentence) description."""
        
        user_prompt = f"""You are a database expert helping generate precise field descriptions for a Korean text-to-SQL system. These descriptions must convey both statistical properties and semantic meaning so an LLM can understand the schema.

Analyze this column and write a clear 1-2 sentence description **IN KOREAN**:

Column Statistics:
{context}

WRITING RULES:
1. **WRITE ONLY IN KOREAN.** (Output must be 100% Korean)
2. Always use specific names instead of pronouns - write "{profile.column_name} 컬럼은..."
3. Avoid ambiguous references - be explicit about what you're describing
4. The column name itself may hint at meaning (e.g., "CDSCode" = County-District-School)
5. Include: what the column stores, data format, key characteristics (range, uniqueness, NULLs, patterns)
6. If semantics are unclear, say "appears to" / "likely" and do not invent domain details.
7. Be accurate and grounded in the provided statistics.

Description:"""
        
        messages = [
            LLMMessage(role="user", content=user_prompt),
        ]
        
        token_limits = [1000, 1500, 2000]

        for max_tokens in token_limits:
            try:
                response = self.llm.generate(messages, max_tokens=max_tokens)
                
                content = response.content or ""
                result = content.strip()
                
                # If we got content, return it
                if result:
                    return result
                
                # If finish_reason is 'length', try again with more tokens
                if response.finish_reason == 'length':
                    print(f"[RETRY] Hit token limit ({max_tokens}), retrying with more...")
                    continue
                else:
                    # Some other issue, don't retry
                    break
                    
            except Exception as e:
                print(f"[ERROR] Short description generation failed: {e}")
                break
        
        # All retries failed
        print(f"[WARN] Empty short long description for {profile.table_name}.{profile.column_name}")
        print(f"[WARN] Tried token limits: {token_limits}")
        return ""
    
    def _generate_long_description(
        self, profile: ColumnProfile, context: str, short_desc: str
    ) -> str:
        """Generate a long (detailed) description."""
        
        if not short_desc:
            print(f"[SKIP] Skipping long description (short was empty)")
            return ""
        
        user_prompt = f"""You are a database expert helping generate precise field descriptions for a Korean text-to-SQL system. Expand the short description with more detail.

Column Statistics:
{context}

Short description: {short_desc}

WRITING RULES:
1. **WRITE ONLY IN KOREAN.**
2. Use specific names instead of pronouns - write "{profile.column_name} 컬럼은..."
3. Maintain the same interpretation from the short description
4. Add specific details: exact data format, value ranges, most common entries, patterns
5. DO NOT give recommendations or analysis - only describe what the column contains
6. Be accurate and grounded in the provided statistics.

Write 3-4 clear sentences **IN KOREAN** expanding the short description.

Detailed description:"""
        
        messages = [
            LLMMessage(role="user", content=user_prompt),
        ]
        
        token_limits = [1000, 1500, 2000]

        for max_tokens in token_limits:
            try:
                response = self.llm.generate(messages, max_tokens=max_tokens)
                
                content = response.content or ""
                result = content.strip()
                
                # If we got content, return it
                if result:
                    return result
                
                # If finish_reason is 'length', try again with more tokens
                if response.finish_reason == 'length':
                    print(f"[RETRY] Hit token limit ({max_tokens}), retrying with more...")
                    continue
                else:
                    # Some other issue, don't retry
                    break
                    
            except Exception as e:
                print(f"[ERROR] Long description generation failed: {e}")
                break
        
        # All retries failed
        print(f"[WARN] Empty long description for {profile.table_name}.{profile.column_name}")
        print(f"[WARN] Tried token limits: {token_limits}")
        return ""
                
    async def summarize_async(self, profile: ColumnProfile) -> FieldMetadata:
        """Async version of summarize."""
        context = self._build_context(profile)
        
        short_desc = await self._generate_short_description_async(profile, context)
        long_desc = await self._generate_long_description_async(profile, context, short_desc)
        
        return FieldMetadata(
            profile=profile,
            short_description=short_desc,
            long_description=long_desc,
            prompt=context,
        )
    
    async def _generate_short_description_async(
        self, profile: ColumnProfile, context: str
    ) -> str:
        """Async version of short description generation."""
        
        user_prompt = f"""You are a database expert helping generate precise field descriptions for a Korean text-to-SQL system. These descriptions must convey both statistical properties and semantic meaning so an LLM can understand the schema.

Analyze this column and write a clear 1-2 sentence description **IN KOREAN**:

Column Statistics:
{context}

WRITING RULES:
1. **WRITE ONLY IN KOREAN.** (Output must be 100% Korean)
2. Always use specific names instead of pronouns - write "{profile.column_name} 컬럼은..."
3. Avoid ambiguous references - be explicit about what you're describing
4. The column name itself may hint at meaning (e.g., "CDSCode" = County-District-School)
5. Include: what the column stores, data format, key characteristics (range, uniqueness, NULLs, patterns)
6. If semantics are unclear, say "appears to" / "likely" and do not invent domain details.
7. Be accurate and grounded in the provided statistics.

Description:"""
        
        messages = [
            LLMMessage(role="user", content=user_prompt),
        ]
        
        token_limits = [1000, 1500, 2000]
        
        for max_tokens in token_limits:
            try:
                response = await self.llm.generate_async(messages, max_tokens=max_tokens)
                
                content = response.content or ""
                result = content.strip()
                
                # If we got content, return it
                if result:
                    return result
                
                # If finish_reason is 'length', try again with more tokens
                if response.finish_reason == 'length':
                    print(f"[RETRY ASYNC] Hit token limit ({max_tokens}), retrying with more...")
                    continue
                else:
                    # Some other issue, don't retry
                    break
                    
            except Exception as e:
                print(f"[ERROR] Async short description failed: {e}")
                break
        
        # All retries failed
        print(f"[WARN] Async Empty short description for {profile.table_name}.{profile.column_name}")
        print(f"[WARN] Tried token limits: {token_limits}")
        return ""
    
    async def _generate_long_description_async(
        self, profile: ColumnProfile, context: str, short_desc: str
    ) -> str:
        """Async version of long description generation."""
        
        if not short_desc:
            return ""
        
        user_prompt = f"""You are a database expert helping generate precise field descriptions for a Korean text-to-SQL system. Expand the short description with more detail.

Column Statistics:
{context}

Short description: {short_desc}

WRITING RULES:
1. **WRITE ONLY IN KOREAN.**
2. Use specific names instead of pronouns - write "{profile.column_name} 컬럼은..."
3. Maintain the same interpretation from the short description
4. Add specific details: exact data format, value ranges, most common entries, patterns
5. DO NOT give recommendations or analysis - only describe what the column contains
6. Be accurate and grounded in the provided statistics.

Write 3-4 clear sentences **IN KOREAN** expanding the short description.

Detailed description:"""
        
        messages = [
            LLMMessage(role="user", content=user_prompt),
        ]
        
        token_limits = [1000, 1500, 2000]
        
        for max_tokens in token_limits:
            try:
                response = await self.llm.generate_async(messages, max_tokens=max_tokens)
                
                content = response.content or ""
                result = content.strip()
                
                # If we got content, return it
                if result:
                    return result
                
                # If finish_reason is 'length', try again with more tokens
                if response.finish_reason == 'length':
                    print(f"[RETRY ASYNC] Hit token limit ({max_tokens}), retrying with more...")
                    continue
                else:
                    # Some other issue, don't retry
                    break
                    
            except Exception as e:
                print(f"[ERROR] Async long description failed: {e}")
                break
        
        # All retries failed
        print(f"[WARN] Async Empty long description for {profile.table_name}.{profile.column_name}")
        print(f"[WARN] Tried token limits: {token_limits}")
        return ""