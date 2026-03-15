import tiktoken
from dataclasses import dataclass
from typing import List, Tuple
from src.config.models import ChunkingConfig


@dataclass
class TextChunk:
    """Represents a text chunk (parent or child)"""
    chunk_id: str
    text: str
    chunk_type: str  # "parent" | "child"
    token_count: int
    
    # For parents
    is_split: bool = False
    split_part: int = 0
    total_parts: int = 1
    
    # For children
    parent_id: str = None
    chunk_index: int = 0  # Position within parent


class TextChunker:
    """
    Chunks text content with overlap and token-aware splitting.
    
    Strategy:
    - Parents: Split long text at paragraph boundaries (max 2000 tokens)
    - Children: Fixed-size chunks with overlap (400 tokens, 50 overlap)
    
    Unlike TableChunker, this handles prose/paragraphs, not structured data.
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize using ChunkingConfig.
        
        Args:
            config: ChunkingConfig with tokenizer and text settings
        """
        # Setup tokenizer (shared with TableChunker)
        try:
            self.encoder = tiktoken.encoding_for_model(config.tokenizer_model)
        except KeyError:
            print(f"️ Warning: Model '{config.tokenizer_model}' not found. Fallback to cl100k_base.")
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Text-specific settings
        self.parent_max_tokens = config.text.parent_max_tokens
        self.child_chunk_size = config.text.child_chunk_size
        self.child_overlap = config.text.child_overlap
        
        # Validation
        if self.child_overlap >= self.child_chunk_size:
            raise ValueError(
                f"child_overlap ({self.child_overlap}) must be < child_chunk_size ({self.child_chunk_size})"
            )
    
    def chunk_text(
        self,
        text: str,
        source_id: str
    ) -> Tuple[List[TextChunk], List[TextChunk]]:
        """
        Chunk text into parents and children.
        
        Args:
            text: Raw text content (can be multi-paragraph)
            source_id: Identifier for source element
            
        Returns:
            (parents, children) tuple
        """
        if not text.strip():
            return [], []
        
        # Step 1: Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # Step 2: Group paragraphs into parents (respecting token limit)
        parents = self._create_parents(paragraphs, source_id)
        
        # Step 3: Create children with overlap
        children = []
        for parent in parents:
            parent_children = self._create_children_from_parent(parent)
            children.extend(parent_children)
        
        return parents, children
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Handles:
        - Double newlines (standard paragraph breaks)
        - Single newlines (preserve if not part of paragraph)
        """
        # Split on double newlines
        paragraphs = text.split('\n\n')
        
        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _create_parents(
        self,
        paragraphs: List[str],
        source_id: str
    ) -> List[TextChunk]:
        """
        Group paragraphs into parent chunks (max 2000 tokens each).
        
        Strategy:
        - Combine paragraphs until token limit reached
        - Split at paragraph boundaries (don't break paragraphs)
        - If single paragraph > 2000 tokens, force split
        """
        parents = []
        current_paras = []
        current_tokens = 0
        part_num = 1
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # Special case: Single paragraph exceeds limit
            if para_tokens > self.parent_max_tokens:
                # Save current parent if exists
                if current_paras:
                    parent = self._create_parent(
                        paragraphs=current_paras,
                        source_id=source_id,
                        part_num=part_num,
                        is_split=True
                    )
                    parents.append(parent)
                    part_num += 1
                    current_paras = []
                    current_tokens = 0
                
                # Force split the large paragraph
                large_parents = self._split_large_paragraph(para, source_id, part_num)
                parents.extend(large_parents)
                part_num += len(large_parents)
                continue
            
            # Check if adding this paragraph exceeds limit
            if current_tokens + para_tokens > self.parent_max_tokens and current_paras:
                # Save current parent
                parent = self._create_parent(
                    paragraphs=current_paras,
                    source_id=source_id,
                    part_num=part_num,
                    is_split=True
                )
                parents.append(parent)
                part_num += 1
                
                # Start new parent
                current_paras = [para]
                current_tokens = para_tokens
            else:
                # Add to current parent
                current_paras.append(para)
                current_tokens += para_tokens
        
        # Save final parent
        if current_paras:
            is_split = len(parents) > 0  # Split if we already have parents
            parent = self._create_parent(
                paragraphs=current_paras,
                source_id=source_id,
                part_num=part_num,
                is_split=is_split
            )
            parents.append(parent)
        
        # Update total_parts
        total_parts = len(parents)
        for p in parents:
            p.total_parts = total_parts
        
        return parents
    
    def _split_large_paragraph(
        self,
        paragraph: str,
        source_id: str,
        start_part_num: int
    ) -> List[TextChunk]:
        """
        Force split a paragraph that exceeds parent_max_tokens.
        
        Strategy: Split at sentence boundaries if possible, otherwise mid-sentence.
        """
        # Split into sentences (simple heuristic)
        sentences = self._split_into_sentences(paragraph)
        
        parents = []
        current_sentences = []
        current_tokens = 0
        part_num = start_part_num
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.parent_max_tokens and current_sentences:
                # Save current parent
                parent_text = ' '.join(current_sentences)
                parent = TextChunk(
                    chunk_id=f"{source_id}_text_part{part_num}",
                    text=parent_text,
                    chunk_type="parent",
                    token_count=self._count_tokens(parent_text),
                    is_split=True,
                    split_part=part_num,
                    total_parts=0  # Will update later
                )
                parents.append(parent)
                part_num += 1
                
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Save final parent
        if current_sentences:
            parent_text = ' '.join(current_sentences)
            parent = TextChunk(
                chunk_id=f"{source_id}_text_part{part_num}",
                text=parent_text,
                chunk_type="parent",
                token_count=self._count_tokens(parent_text),
                is_split=True,
                split_part=part_num,
                total_parts=0
            )
            parents.append(parent)
        
        return parents
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Simple heuristic: Split on '. ', '! ', '? ' followed by capital letter.
        """
        import re
        
        # Split on sentence terminators followed by space and capital letter
        sentences = re.split(r'([.!?]\s+)(?=[A-Z가-힣])', text)
        
        # Recombine (the split includes the delimiter as separate element)
        result = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        
        return [s.strip() for s in result if s.strip()]
    
    def _create_parent(
        self,
        paragraphs: List[str],
        source_id: str,
        part_num: int,
        is_split: bool
    ) -> TextChunk:
        """Create a parent chunk from paragraphs."""
        parent_text = '\n\n'.join(paragraphs)
        
        chunk_id = f"{source_id}_text_part{part_num}" if is_split else f"{source_id}_text"
        
        return TextChunk(
            chunk_id=chunk_id,
            text=parent_text,
            chunk_type="parent",
            token_count=self._count_tokens(parent_text),
            is_split=is_split,
            split_part=part_num,
            total_parts=0  # Will be updated by caller
        )
    
    def _create_children_from_parent(self, parent: TextChunk) -> List[TextChunk]:
        """
        Create children with overlap from parent.
        
        Strategy:
        - Fixed-size chunks (400 tokens)
        - Overlap (50 tokens) for context continuity
        - Slide window through parent text
        """
        children = []
        parent_tokens = self.encoder.encode(parent.text)
        
        # Calculate stride (chunk_size - overlap)
        stride = self.child_chunk_size - self.child_overlap
        
        child_idx = 0
        start_pos = 0
        
        while start_pos < len(parent_tokens):
            # Extract chunk
            end_pos = min(start_pos + self.child_chunk_size, len(parent_tokens))
            chunk_tokens = parent_tokens[start_pos:end_pos]
            
            # Decode back to text
            chunk_text = self.encoder.decode(chunk_tokens)
            
            # Create child
            child = TextChunk(
                chunk_id=f"{parent.chunk_id}_child_{child_idx}",
                text=chunk_text,
                chunk_type="child",
                token_count=len(chunk_tokens),
                parent_id=parent.chunk_id,
                chunk_index=child_idx
            )
            children.append(child)
            
            # Move window
            start_pos += stride
            child_idx += 1
            
            # Avoid infinite loop on last tiny chunk
            if end_pos >= len(parent_tokens):
                break
        
        return children
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoder.encode(text))