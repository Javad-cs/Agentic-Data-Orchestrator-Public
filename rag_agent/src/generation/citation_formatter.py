import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CitationFormatter:
    """
    Format citations in numbered bracket style: [1], [2], etc.
    
    Features:
    - Maps sources to citation numbers
    - Formats citation list at end of response
    - Handles deduplication (same source = same number)
    """
    
    def __init__(self):
        """Initialize citation formatter"""
        self.citation_map: Dict[str, int] = {}  # source_id -> citation_number
        self.sources: List[Dict[str, Any]] = []  # List of source metadata
        self.next_citation_num = 1
    
    def reset(self):
        """Reset citation state (call between queries)"""
        self.citation_map = {}
        self.sources = []
        self.next_citation_num = 1
    
    def add_source(self, source: Dict[str, Any]) -> int:
        """
        Add a source and get its citation number.
        
        Args:
            source: Source metadata dict with keys:
                - child_id: Unique ID
                - parent_id: Parent document ID
                - source_file: File path
                - page_number: Page number (optional)
                - parent_type: Type (text/table)
                
        Returns:
            Citation number for this source
        """
        source_id = source.get('child_id') or source.get('parent_id')
        
        if not source_id:
            logger.warning("Source missing both child_id and parent_id, skipping")
            return 0
        
        # Check if we've seen this source before
        if source_id in self.citation_map:
            return self.citation_map[source_id]
        
        # New source - assign next number
        citation_num = self.next_citation_num
        self.citation_map[source_id] = citation_num
        self.sources.append({
            'number': citation_num,
            'source_id': source_id,
            'file': source.get('source_file', 'Unknown'),
            'page': source.get('page_number'),
            'type': source.get('parent_type', 'text')
        })
        
        self.next_citation_num += 1
        
        return citation_num
    
    def add_sources(self, sources: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple sources and get their citation numbers.
        
        Args:
            sources: List of source metadata dicts
            
        Returns:
            List of citation numbers
        """
        return [self.add_source(source) for source in sources]
    
    def format_citation_marker(self, citation_num: int) -> str:
        """
        Format a citation marker.
        
        Args:
            citation_num: Citation number
            
        Returns:
            Formatted marker like "[1]"
        """
        return f"[{citation_num}]"
    
    def format_citation_list(self, language: str = "ko") -> str:
        """
        Format the full citation list.
        
        Args:
            language: "ko" for Korean, "en" for English
            
        Returns:
            Formatted citation list as string
        """
        if not self.sources:
            return ""
        
        # Header
        if language == "ko":
            header = "\n\n참고 문서:"
        else:
            header = "\n\nSources:"
        
        # Format each source
        lines = [header]
        for source in self.sources:
            citation_text = self._format_single_citation(source, language)
            lines.append(citation_text)
        
        return "\n".join(lines)
    
    def _format_single_citation(self, source: Dict[str, Any], language: str) -> str:
        """
        Format a single citation entry.
        
        Args:
            source: Source metadata
            language: "ko" or "en"
            
        Returns:
            Formatted citation string
        """
        num = source['number']
        file = source['file']
        page = source.get('page')
        
        # Extract filename from path
        filename = file.split('/')[-1] if '/' in file else file
        
        # Format with page number if available
        if page is not None:
            if language == "ko":
                return f"[{num}] {filename}, {page}페이지"
            else:
                return f"[{num}] {filename}, page {page}"
        else:
            return f"[{num}] {filename}"
    
    def inject_citations_into_text(
        self,
        text: str,
        source_mapping: Dict[str, int]
    ) -> str:
        """
        Inject citation markers into generated text.
        
        This is a placeholder for future LLM-based citation injection.
        For now, citations should be added by the LLM during generation.
        
        Args:
            text: Generated text
            source_mapping: Mapping of key phrases to citation numbers
            
        Returns:
            Text with citations injected
        """
        # Week 1: Return text as-is (LLM should add citations)
        # Week 2+: Implement smart citation injection
        return text
    
    def get_citation_metadata(self) -> List[Dict[str, Any]]:
        """
        Get all citation metadata for frontend.
        
        Returns:
            List of citation metadata dicts
        """
        return [
            {
                'id': f"[{s['number']}]",
                'file': s['file'],
                'page': s.get('page'),
                'source_id': s['source_id'],
                'type': s['type']
            }
            for s in self.sources
        ]
    
    def validate_citations_in_text(self, text: str) -> Tuple[bool, List[int]]:
        """
        Check if all citations in text are valid.
        
        Args:
            text: Generated text with citations
            
        Returns:
            (all_valid, invalid_citation_nums)
        """
        # Find all citation markers in text
        pattern = r'\[(\d+)\]'
        cited_nums = [int(m) for m in re.findall(pattern, text)]
        
        # Check if all cited numbers are in our sources
        valid_nums = set(s['number'] for s in self.sources)
        invalid_nums = [n for n in cited_nums if n not in valid_nums]
        
        return len(invalid_nums) == 0, invalid_nums