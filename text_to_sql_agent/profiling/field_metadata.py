"""
Updated FieldMetadata with SME description support.
This should replace the FieldMetadata in summarizer.py
"""

from typing import Optional
from dataclasses import dataclass
from .statistics import ColumnProfile


@dataclass
class FieldMetadata:
    """
    Complete metadata for a field including profile, LLM summaries, and SME descriptions.
    
    Paper-aligned: Combines SME (subject matter expert) descriptions with 
    LLM-generated summaries for comprehensive field understanding.
    """
    
    # Original profile (statistics)
    profile: ColumnProfile
    
    # LLM-generated descriptions
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    
    # SME-provided descriptions (from BIRD dataset)
    sme_description: Optional[str] = None
    sme_source: Optional[str] = None  # "csv" or "json"
    
    # Raw prompt and response (for debugging)
    prompt: Optional[str] = None
    raw_response: Optional[str] = None
    
    @property
    def minimal_description(self) -> str:
        """
        Minimal profile: short LLM description only.
        Used in focused_minimal and full_minimal variants.
        """
        return self.short_description or ""
    
    @property
    def maximal_description(self) -> str:
        """
        Maximal profile: long LLM description only.
        Used in focused_maximal and full_maximal variants.
        """
        return self.long_description or ""
    
    @property
    def full_description(self) -> str:
        """
        Full profile: SME description + long LLM description (combined).
        Used in focused_full variant and for FAISS indexing.
        
        Paper-aligned: The "full profile" in the paper combines
        SME knowledge with LLM-generated statistical summaries.
        """
        parts = []
        
        if self.sme_description:
            parts.append(self.sme_description)
        
        if self.long_description:
            parts.append(self.long_description)
        
        return "\n\n".join(parts) if parts else self.maximal_description
    
    @property
    def best_description(self) -> str:
        """
        Best available description for general use.
        Priority: full > maximal > minimal
        """
        return self.full_description or self.maximal_description or self.minimal_description
    
    def __repr__(self):
        sources = []
        if self.sme_description:
            sources.append(f"SME({self.sme_source})")
        if self.long_description:
            sources.append("LLM-long")
        if self.short_description:
            sources.append("LLM-short")
        
        source_str = "+".join(sources) if sources else "none"
        return f"FieldMetadata({self.profile.table_name}.{self.profile.column_name}, sources={source_str})"