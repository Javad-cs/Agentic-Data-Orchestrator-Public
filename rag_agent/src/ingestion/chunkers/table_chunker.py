import tiktoken
from dataclasses import dataclass
from typing import List, Tuple
from src.config.models import ChunkingConfig
import logging
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)

@dataclass
class TableChunk:
    """Represents a table chunk (parent or child)"""
    chunk_id: str
    text: str
    chunk_type: str  # "parent" | "child"
    token_count: int
    
    # Metadata
    is_split: bool = False
    split_part: int = 0
    total_parts: int = 1
    
    # Row Tracking
    # For Parents: Use start/end_row_idx (it defines a contiguous range of the original table)
    start_row_idx: int = 0
    end_row_idx: int = 0
    
    # For Children: Use row_indices (it defines the specific, potentially non-contiguous rows in this group)
    row_indices: List[int] = None
    
    parent_id: str = None
    has_header: bool = False

class TableChunker:
    """
    Chunks tables using a centralized ChunkingConfig.
    """
    
    def __init__(self, config: ChunkingConfig):
        # Store config for later use
        self.chunking_config = config
        
        # 1. Setup Tokenizer
        try:
            self.encoder = tiktoken.encoding_for_model(config.tokenizer_model)
        except KeyError:
            print(f"️ Warning: Model '{config.tokenizer_model}' not found. Fallback to cl100k_base.")
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # 2. Setup Table Settings
        self.parent_max_tokens = config.table.parent_max_tokens
        self.child_target_tokens = config.table.child_target_tokens
        self.min_rows_per_child = config.table.min_rows_per_child
        self.max_rows_per_child = config.table.max_rows_per_child
        
    def _detect_corrupted_table(self, table_text: str) -> bool:
        """
        Detect if table parsing failed (merged cells/headers).
        
        Heuristics:
        - First row >1500 tokens (likely a merged blob)
        - Missing proper separator row
        
        Returns:
            True if table appears corrupted
        """
        lines = table_text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check first row token count
        first_row_tokens = self._count_tokens(lines[0])
        if first_row_tokens > 1500:
            # Very likely corrupted - normal headers are 10-300 tokens
            return True
        
        # Additional check: if first row is >500 tokens AND no separator
        if first_row_tokens > 500 and len(lines) > 1:
            second_is_separator = lines[1].strip().replace('|', '').replace('-', '').strip() == ''
            if not second_is_separator:
                # Possibly corrupted
                return True
        
        return False

    def chunk_table(
        self, 
        table_markdown: str,
        source_id: str
    ) -> Tuple[List[TableChunk], List[TableChunk]]:
        """Chunk a markdown table into parents and children."""
        
        # Step 0: Check if table appears to be corrupted
        if self._detect_corrupted_table(table_markdown):
            logger.warning(
                f"Table {source_id} appears corrupted (layout analysis failure - "
                f"likely merged header). Using aggressive chunking."
            )
            
            # Not using text_chunker cause it assumes prose structure
            # Instead, chunking by tokens directly
            return self._chunk_corrupted_table_by_tokens(table_markdown, source_id)
    
        # Step 1: Parse table structure
        header, separator, rows = self._parse_markdown_table(table_markdown)
        
        # Step 2: Decide if split needed
        total_tokens = self._count_tokens(table_markdown)
        
        if total_tokens <= self.parent_max_tokens:
            parents = [self._create_parent(
                header, separator, rows, 0, len(rows), source_id, 1, 1, False
            )]
        else:
            parents = self._split_into_parents(header, separator, rows, source_id)
        
        # Step 3: Create children for each parent
        children = []
        for parent in parents:
            children.extend(self._create_children_from_parent(parent, header, separator, rows))
        
        return parents, children
    
    def _chunk_corrupted_table_by_tokens(
        self, 
        table_text: str, 
        source_id: str
    ) -> Tuple[List[TableChunk], List[TableChunk]]:
        """
        Emergency chunking for corrupted tables.
        
        Simply splits by tokens into safe-sized chunks.
        No structure preservation - data is already corrupted.
        """
        encoder = self.encoder
        
        tokens = encoder.encode(table_text)
        max_parent_tokens = 1800  # Safe limit
        max_child_tokens = 350
        
        parents = []
        children = []
        
        # Split into parent chunks
        parent_idx = 0
        for start in range(0, len(tokens), max_parent_tokens):
            end = min(start + max_parent_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoder.decode(chunk_tokens)
            
            parent = TableChunk(
                chunk_id=f"{source_id}_corrupted_part{parent_idx}",
                text=chunk_text,
                chunk_type="parent",
                token_count=len(chunk_tokens),
                row_indices=[],
                has_header=False
            )
            parents.append(parent)
            
            # Create children from this parent
            child_idx = 0
            for child_start in range(0, len(chunk_tokens), max_child_tokens):
                child_end = min(child_start + max_child_tokens, len(chunk_tokens))
                child_chunk = chunk_tokens[child_start:child_end]
                child_text = encoder.decode(child_chunk)
                
                child = TableChunk(
                    chunk_id=f"{parent.chunk_id}_child_{child_idx}",
                    text=child_text,
                    chunk_type="child",
                    token_count=len(child_chunk),
                    parent_id=parent.chunk_id,
                    row_indices=[],
                    has_header=False
                )
                children.append(child)
                child_idx += 1
            
            parent_idx += 1
        
        logger.info(
            f"Corrupted table chunked by tokens: {len(parents)} parents, "
            f"{len(children)} children (max {max(c.token_count for c in children)} tokens)"
        )
        
        return parents, children
    
    def _parse_markdown_table(self, markdown: str) -> Tuple[str, str, List[str]]:
        """Robust parsing with missing separator handling"""
        lines = [line.strip() for line in markdown.strip().split('\n') if line.strip()]
        
        # FIX 1: Raise error instead of returning empty
        if len(lines) < 2:
            raise ValueError(f"Invalid table: need at least header + 1 row/separator, got {len(lines)} lines")
            
        header = lines[0]
        potential_separator = lines[1]
        
        is_valid_separator = set(potential_separator).issubset({'|', '-', ':', ' '})
        
        if is_valid_separator:
            separator = potential_separator
            rows = lines[2:]
        else:
            col_count = header.count('|') - 1
            if col_count < 1: col_count = 1
            separator = "|" + "---|" * col_count
            rows = lines[1:]
            
        return header, separator, rows

    def _split_into_parents(self, header, separator, rows, source_id) -> List[TableChunk]:
        parents = []
        current_rows = []
        overhead_tokens = self._count_tokens(header) + self._count_tokens(separator)
        current_tokens = overhead_tokens
        row_start_idx = 0
        part_num = 1
        
        for i, row in enumerate(rows):
            row_tokens = self._count_tokens(row)
            
            if current_tokens + row_tokens > self.parent_max_tokens and current_rows:
                parents.append(self._create_parent(
                    header, separator, current_rows, row_start_idx, row_start_idx + len(current_rows),
                    source_id, part_num, 0, True
                ))
                current_rows = [row]
                current_tokens = overhead_tokens + row_tokens
                row_start_idx = i
                part_num += 1
            else:
                current_rows.append(row)
                current_tokens += row_tokens
        
        if current_rows:
            parents.append(self._create_parent(
                header, separator, current_rows, row_start_idx, row_start_idx + len(current_rows),
                source_id, part_num, 0, True
            ))
            
        total_parts = len(parents)
        for p in parents: p.total_parts = total_parts
        return parents

    def _create_parent(self, header, separator, rows, start, end, source_id, part, total, is_split):
        parent_text = "\n".join([header, separator] + rows)
        chunk_id = f"{source_id}_table_part{part}" if is_split else f"{source_id}_table"
        return TableChunk(
            chunk_id=chunk_id,
            text=parent_text,
            chunk_type="parent",
            token_count=self._count_tokens(parent_text),
            is_split=is_split,
            split_part=part,
            total_parts=total,
            start_row_idx=start,
            end_row_idx=end,
            has_header=True
        )

    def _create_children_from_parent(self, parent, header, separator, all_rows) -> List[TableChunk]:
        children = []
        parent_rows = all_rows[parent.start_row_idx : parent.end_row_idx]
        
        current_group = []
        current_indices = []
        overhead_tokens = self._count_tokens(header) + self._count_tokens(separator)
        current_tokens = overhead_tokens
        child_idx = 0
        
        MAX_CHILD_TOKENS = 3500  # Hard API limit
        
        for local_idx, row in enumerate(parent_rows):
            global_idx = parent.start_row_idx + local_idx
            row_tokens = self._count_tokens(row)
            
            # NEW: If single row exceeds limit, split it into pseudo-rows
            if row_tokens > MAX_CHILD_TOKENS:
                logger.warning(
                    f" Row {global_idx} exceeds {MAX_CHILD_TOKENS} tokens ({row_tokens}). "
                    f"Splitting by columns into pseudo-rows."
                )
                
                # Split into smaller rows by columns
                pseudo_rows = self._split_monster_row_by_columns(row, MAX_CHILD_TOKENS - overhead_tokens)
                
                # Process each pseudo-row as if it were a normal row
                for pseudo_row in pseudo_rows:
                    pseudo_tokens = self._count_tokens(pseudo_row)
                    
                    # Force flush if would exceed
                    if current_tokens + pseudo_tokens > MAX_CHILD_TOKENS and current_group:
                        children.append(self._create_child(
                            parent.chunk_id, header, separator, current_group, current_indices, child_idx
                        ))
                        child_idx += 1
                        current_group = []
                        current_indices = []
                        current_tokens = overhead_tokens
                    
                    # Add pseudo-row to current group
                    current_group.append(pseudo_row)
                    current_indices.append(global_idx)  # Same index (it's from same original row)
                    current_tokens += pseudo_tokens
                
                continue  # Done with this monster row
            
            # NEW: Force flush if accumulated tokens exceed hard limit
            if current_tokens > MAX_CHILD_TOKENS and current_group:
                children.append(self._create_child(
                    parent.chunk_id, header, separator, current_group, current_indices, child_idx
                ))
                child_idx += 1
                current_group = []
                current_indices = []
                current_tokens = overhead_tokens
            
            # ORIGINAL LOGIC (unchanged)
            would_exceed = (current_tokens + row_tokens > self.child_target_tokens)
            has_min = len(current_group) >= self.min_rows_per_child
            has_max = len(current_group) >= self.max_rows_per_child
            
            if (would_exceed and has_min) or has_max:
                if current_group:
                    children.append(self._create_child(
                        parent.chunk_id, header, separator, current_group, current_indices, child_idx
                    ))
                    child_idx += 1
                current_group = [row]
                current_indices = [global_idx]
                current_tokens = overhead_tokens + row_tokens
            else:
                current_group.append(row)
                current_indices.append(global_idx)
                current_tokens += row_tokens
        
        if current_group:
            children.append(self._create_child(
                parent.chunk_id, header, separator, current_group, current_indices, child_idx
            ))
        
        return children
    
    def _split_monster_row_by_columns(self, row: str, max_tokens: int) -> List[str]:
        """
        Split a giant row into smaller pseudo-rows by grouping columns.
        
        If a single cell exceeds limit, splits that cell by tokens with overlap.
        Returns list of pseudo-rows (each under token limit).
        """
        # Parse cells - keep ALL parts including empty
        parts = row.split('|')
        if len(parts) < 3:  # Need at least "| cell |"
            logger.error(f"Cannot parse row structure: {row[:100]}...")
            return []  # Skip unparseable row
        
        cells = [p.strip() for p in parts[1:-1]]  # Between first and last pipe
        
        pseudo_rows = []
        current_cells = []
        current_tokens = 0
        
        for cell in cells:
            cell_tokens = self._count_tokens(cell)
            
            # If single cell exceeds limit, split it by tokens
            if cell_tokens > max_tokens:
                # Flush current group first
                if current_cells:
                    pseudo_row = '| ' + ' | '.join(current_cells) + ' |'
                    pseudo_rows.append(pseudo_row)
                    current_cells = []
                    current_tokens = 0
                
                logger.warning(
                    f" Single cell exceeds {max_tokens} tokens ({cell_tokens}). "
                    f"Splitting cell by tokens with overlap."
                )
                
                # Split this cell into token chunks
                cell_chunks = self._split_cell_by_tokens(cell, max_tokens)
                
                # Each chunk becomes its own pseudo-row
                for chunk in cell_chunks:
                    pseudo_row = f'| {chunk} |'
                    pseudo_rows.append(pseudo_row)
                
                continue  # Done with this monster cell
            
            # Would adding this cell exceed limit?
            if current_tokens + cell_tokens > max_tokens and current_cells:
                # Flush current pseudo-row
                pseudo_row = '| ' + ' | '.join(current_cells) + ' |'
                pseudo_rows.append(pseudo_row)
                current_cells = []
                current_tokens = 0
            
            # Add cell (preserves empty cells)
            current_cells.append(cell)
            current_tokens += cell_tokens
        
        # Flush remaining
        if current_cells:
            pseudo_row = '| ' + ' | '.join(current_cells) + ' |'
            pseudo_rows.append(pseudo_row)
        
        logger.info(f" Split monster row into {len(pseudo_rows)} pseudo-rows")
        return pseudo_rows


    def _split_cell_by_tokens(self, cell: str, max_tokens: int) -> List[str]:
        """
        Split a single monster cell by tokens with overlap.
        
        Last resort for cells that exceed token limit.
        """
        tokens = self.encoder.encode(cell)
        
        if len(tokens) <= max_tokens:
            return [cell]  # Shouldn't happen, but safety check
        
        chunks = []
        overlap = 50
        stride = max_tokens - overlap
        pos = 0
        
        while pos < len(tokens):
            end = min(pos + max_tokens, len(tokens))
            chunk_tokens = tokens[pos:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            pos += stride
            if end >= len(tokens):
                break
        
        logger.info(f"  Split monster cell into {len(chunks)} token-based chunks")
        return chunks

    def _create_child(self, parent_id, header, separator, rows, row_indices, child_idx):
        child_text = "\n".join([header, separator] + rows)
        return TableChunk(
            chunk_id=f"{parent_id}_child_{child_idx}",
            text=child_text,
            chunk_type="child",
            token_count=self._count_tokens(child_text),
            parent_id=parent_id,
            row_indices=row_indices,
            has_header=True
        )

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        return len(self.encoder.encode(text))