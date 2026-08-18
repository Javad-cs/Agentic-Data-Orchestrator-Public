"""
Schema augmentation for iterative refinement.
Adds fields containing missing literals to the schema.
"""

from typing import Set, Tuple, Dict, Optional, List
from schema_linking import SchemaRepresentation, SchemaVariant
from profiling.field_metadata import FieldMetadata


class SchemaAugmenter:
    """
    Augment schema with additional fields.
    
    Used in refinement loop when SQL uses literals but doesn't
    reference the fields containing those literals.
    """
    
    def __init__(self, metadata_map: Dict[Tuple[str, str], FieldMetadata]):
        """
        Initialize augmenter.
        
        Args:
            metadata_map: Map of (table, column) to FieldMetadata
        """
        self.metadata_map = metadata_map
    
    def augment_schema_text(
        self,
        original_schema: str,
        additional_fields: Set[Tuple[str, str]],
        variant: SchemaVariant,
        already_augmented: Optional[Set[Tuple[str, str]]] = None
    ) -> str:
        """
        Add fields to existing schema text.
        
        Args:
            original_schema: Original schema text
            additional_fields: Fields to add {(table, column), ...}
            variant: Schema variant (determines description level)
            already_augmented: Fields already added in previous iterations
            
        Returns:
            Augmented schema text
        """
        if already_augmented is None:
            already_augmented = set()
        
        # Only add NEW fields (not already augmented)
        new_fields = additional_fields - already_augmented
        
        if not new_fields:
            return original_schema
        
        # Group new fields by table
        by_table: Dict[str, List[Tuple[str, str]]] = {}
        for table, column in new_fields:
            if table not in by_table:
                by_table[table] = []
            by_table[table].append((table, column))
        
        # Build augmentation text
        augmentation = []
        augmentation.append("\n--- Additional Fields (from literal matching) ---\n")
        
        for table in sorted(by_table.keys()):
            augmentation.append(f"Table: {table}")
            
            for table, column in by_table[table]:
                # Get metadata
                metadata = self.metadata_map.get((table, column))
                
                if metadata:
                    # Get appropriate description based on variant
                    description = self._get_description(metadata, variant)
                    augmentation.append(f"  - {column}: {description}")
                else:
                    augmentation.append(f"  - {column}")
            
            augmentation.append("")
        
        # Combine original + augmentation
        return original_schema + "\n".join(augmentation)
    
    def _get_description(self, metadata: FieldMetadata, variant: SchemaVariant) -> str:
        """Get appropriate description based on variant."""
        profile_type = variant.profile_type()
        
        if profile_type == "minimal":
            return metadata.minimal_description
        elif profile_type == "maximal":
            return metadata.maximal_description
        elif profile_type == "full":
            return metadata.full_description
        
        return ""
    
    def create_augmented_representation(
        self,
        original_representation: SchemaRepresentation,
        additional_fields: Set[Tuple[str, str]]
    ) -> SchemaRepresentation:
        """
        Create new SchemaRepresentation with augmented fields.
        
        Args:
            original_representation: Original schema
            additional_fields: Fields to add
            
        Returns:
            New SchemaRepresentation
        """
        # Create new field list
        from schema_linking import FocusedField
        
        augmented_fields = list(original_representation.fields)
        
        # Add new fields
        for table, column in additional_fields:
            # Check if not already present
            if not any(f.table == table and f.column == column for f in augmented_fields):
                augmented_fields.append(
                    FocusedField(
                        table=table,
                        column=column,
                        selected_by="augmented"
                    )
                )
        
        # Generate new text
        augmented_text = self.augment_schema_text(
            original_representation.text,
            additional_fields,
            original_representation.variant
        )
        
        return SchemaRepresentation(
            variant=original_representation.variant,
            fields=augmented_fields,
            text=augmented_text
        )


def create_revision_prompt(
    question: str,
    original_sql: str,
    missing_literals: List[str],
    literal_fields: Dict[str, Set[Tuple[str, str]]]
) -> str:
    """
    Create prompt for LLM to revise SQL with missing literal fields.
    
    Args:
        question: Original question
        original_sql: Previously generated SQL
        missing_literals: Literals used but fields not referenced
        literal_fields: Map of literal → fields containing it
        
    Returns:
        Revision prompt
    """
    prompt_parts = []
    
    prompt_parts.append(f"Question: {question}\n")
    prompt_parts.append(f"Your previous SQL query:\n```sql\n{original_sql}\n```\n")
    prompt_parts.append("However, there are issues with literal usage:\n")
    
    for literal in missing_literals:
        fields = literal_fields.get(literal, set())
        if fields:
            field_list = ", ".join(f"{t}.{c}" for t, c in sorted(fields))
            prompt_parts.append(
                f"- You used the literal '{literal}' but didn't reference "
                f"the field(s) that contain this value: {field_list}"
            )
    
    prompt_parts.append(
        "\nPlease revise the SQL query to properly use the fields containing these literals. "
        "Important instructions:\n"
        "1. JOIN the necessary tables using appropriate key columns (e.g., id, foreign keys)\n"
        "2. Use explicit JOIN clauses - do not assume implicit joins\n"
        "3. Reference the correct fields that contain the literal values"
    )
    
    return "\n".join(prompt_parts)