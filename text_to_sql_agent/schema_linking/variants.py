"""
Schema variant generation for LLM consumption.
Generates 5 text representations as defined in the BIRD paper.
"""

from typing import List, Dict, Optional

from .types import FocusedField, SchemaVariant, SchemaRepresentation
from profiling.field_metadata import FieldMetadata


class SchemaVariantGenerator:
    """
    Generate schema variants for LLM prompting.
    
    Paper defines 5 variants:
    1. focused_minimal: Focused schema + short descriptions
    2. focused_maximal: Focused schema + long LLM descriptions  
    3. focused_full: Focused schema + SME + long LLM descriptions
    4. full_minimal: All fields + short descriptions
    5. full_maximal: All fields + long LLM descriptions
    
    The focused variants use semantically-relevant fields (FAISS + LSH).
    The full variants use all database fields.
    """
    
    @staticmethod
    def _key(table: str, column: str) -> tuple:
        """
        Canonicalize (table, column) key to prevent case/whitespace mismatches.
        
        Args:
            table: Table name
            column: Column name
            
        Returns:
            Normalized (table, column) tuple
        """
        return (table.strip().lower(), column.strip().lower())
    
    def __init__(self, metadata_map: Dict[tuple, FieldMetadata], debug: bool = False):
        """
        Initialize variant generator.
        
        Args:
            metadata_map: Dict mapping (table, column) to FieldMetadata
            debug: Whether to print debug warnings for missing metadata
        """
        # Canonicalize all keys in metadata_map to ensure consistent lookups
        self.metadata_map = {
            self._key(table, column): metadata
            for (table, column), metadata in metadata_map.items()
        }
        self.debug = debug
    
    def generate(
        self,
        variant: SchemaVariant,
        focused_fields: Optional[List[FocusedField]] = None,
        include_scores: bool = False
    ) -> SchemaRepresentation:
        """
        Generate a schema variant.
        
        Args:
            variant: Which variant to generate
            focused_fields: Required for focused_* variants
            include_scores: Whether to include FAISS/LSH scores in output
            
        Returns:
            SchemaRepresentation with formatted text
        """
        if variant.is_focused() and not focused_fields:
            raise ValueError(f"{variant} requires focused_fields")
        
        # Determine which fields to include
        if variant.is_focused():
            fields_to_include = focused_fields
            
            # Validate: check for missing metadata (using canonicalized keys)
            missing = [
                f for f in focused_fields 
                if self._key(f.table, f.column) not in self.metadata_map
            ]
            if missing and self.debug:
                print(f"Warning: {len(missing)} focused fields missing metadata:")
                for f in missing[:3]:
                    print(f"  - {f.table}.{f.column}")
                if len(missing) > 3:
                    print(f"  ... and {len(missing) - 3} more")
        else:
            # Full schema: all fields (using canonicalized keys from metadata_map)
            fields_to_include = [
                FocusedField(table=table, column=column, selected_by="full")
                for (table, column) in self.metadata_map.keys()
            ]
        
        # Generate text representation
        text = self._generate_text(variant, fields_to_include, include_scores)
        
        return SchemaRepresentation(
            variant=variant,
            fields=fields_to_include,
            text=text
        )
    
    def _generate_text(
        self,
        variant: SchemaVariant,
        fields: List[FocusedField],
        include_scores: bool
    ) -> str:
        """Generate formatted text for variant."""
        
        # Group fields by table
        tables: Dict[str, List[FocusedField]] = {}
        for field in fields:
            if field.table not in tables:
                tables[field.table] = []
            tables[field.table].append(field)
        
        # Generate text
        lines = []
        lines.append(f"=== Schema ({variant.value}) ===\n")
        
        for table_name in sorted(tables.keys()):
            lines.append(f"Table: {table_name}")
            
            for field in tables[table_name]:
                # Use canonicalized key for lookup
                metadata = self.metadata_map.get(self._key(field.table, field.column))
                
                # Handle missing metadata (field in focused schema but not profiled)
                if not metadata:
                    # Include field (recall > precision)
                    field_line = f"  - {field.column}"
                    
                    if include_scores and variant.is_focused():
                        scores = []
                        if field.faiss_score is not None:
                            scores.append(f"faiss={field.faiss_score:.2f}")
                        if field.lsh_score is not None:
                            scores.append(f"lsh={field.lsh_score:.2f}")
                        if scores:
                            field_line += f" ({', '.join(scores)})"
                    
                    # Only show marker in debug mode (clean prompts for production)
                    if self.debug:
                        field_line += " [metadata not available]"
                    
                    lines.append(field_line)
                    continue
                
                # Get appropriate description based on variant
                description = self._get_description(metadata, variant)
                
                # Format field line
                field_line = f"  - {field.column}"
                
                if include_scores and variant.is_focused():
                    scores = []
                    if field.faiss_score is not None:
                        scores.append(f"faiss={field.faiss_score:.2f}")
                    if field.lsh_score is not None:
                        scores.append(f"lsh={field.lsh_score:.2f}")
                    if scores:
                        field_line += f" ({', '.join(scores)})"
                
                if description:
                    field_line += f": {description}"
                
                lines.append(field_line)
            
            lines.append("")  # Blank line between tables
        
        return "\n".join(lines)
    
    def _get_description(
        self,
        metadata: FieldMetadata,
        variant: SchemaVariant
    ) -> str:
        """Get appropriate description for variant type."""
        
        profile_type = variant.profile_type()
        
        if profile_type == "minimal":
            # Use short description
            return metadata.minimal_description
        
        elif profile_type == "maximal":
            # Use long LLM description only
            return metadata.maximal_description
        
        elif profile_type == "full":
            # Use SME + long LLM description (combined)
            return metadata.full_description
        
        return ""
    
    def generate_all(
        self,
        focused_fields: List[FocusedField],
        include_scores: bool = False
    ) -> Dict[SchemaVariant, SchemaRepresentation]:
        """
        Generate all 5 variants.
        
        Args:
            focused_fields: Fields selected by focused schema builder
            include_scores: Whether to include scores
            
        Returns:
            Dict mapping variant to its representation
        """
        variants = {}
        
        for variant in SchemaVariant:
            try:
                rep = self.generate(variant, focused_fields, include_scores)
                variants[variant] = rep
            except Exception as e:
                print(f"Warning: Failed to generate {variant}: {e}")
        
        return variants
    
    def generate_compact(
        self,
        variant: SchemaVariant,
        focused_fields: Optional[List[FocusedField]] = None,
    ) -> str:
        """
        Generate compact representation (just table.column list).
        
        Useful for very token-constrained scenarios.
        """
        if variant.is_focused() and not focused_fields:
            raise ValueError(f"{variant} requires focused_fields")
        
        if variant.is_focused():
            fields_to_include = focused_fields
        else:
            fields_to_include = [
                FocusedField(table=table, column=column, selected_by="full")
                for (table, column) in self.metadata_map.keys()
            ]
        
        # Group by table
        tables: Dict[str, List[str]] = {}
        for field in fields_to_include:
            if field.table not in tables:
                tables[field.table] = []
            tables[field.table].append(field.column)
        
        # Format compactly
        lines = []
        for table_name in sorted(tables.keys()):
            columns = ", ".join(sorted(tables[table_name]))
            lines.append(f"{table_name}: {columns}")
        
        return "\n".join(lines)


def format_schema_for_sql_generation(
    representation: SchemaRepresentation,
    question: str,
    additional_context: str = ""
) -> str:
    """
    Format schema representation for SQL generation prompt.
    
    This wraps the schema with the question and any additional context
    to create a complete prompt for the LLM.
    
    Args:
        representation: Schema representation
        question: User's natural language question
        additional_context: Optional additional context (e.g., database info)
        
    Returns:
        Formatted prompt text
    """
    prompt_parts = []
    
    # Question
    prompt_parts.append(f"Question: {question}\n")
    
    # Schema
    prompt_parts.append(representation.text)
    
    # Additional context
    if additional_context:
        prompt_parts.append(f"\nAdditional Context:\n{additional_context}")
    
    # Instruction
    prompt_parts.append("\nGenerate a SQL query to answer the question using the provided schema.")
    
    return "\n".join(prompt_parts)