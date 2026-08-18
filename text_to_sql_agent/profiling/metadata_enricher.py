"""
Helper to merge SME descriptions with LLM-generated metadata.
"""

from pathlib import Path
from typing import Optional
from .field_metadata import FieldMetadata
from .statistics import ColumnProfile
from schema_linking.sme_parser import SMEParser


class MetadataEnricher:
    """
    Enriches FieldMetadata with SME descriptions from BIRD dataset.
    
    Usage:
        enricher = MetadataEnricher(bird_data_path)
        metadata = enricher.enrich(db_name, metadata)
    """
    
    def __init__(self, bird_root_path: Path):
        """
        Initialize enricher.
        
        Args:
            bird_root_path: Path to BIRD dataset root 
                           (e.g., /path/to/dev_20240627)
        """
        self.sme_parser = SMEParser(bird_root_path)
    
    def enrich(
        self, 
        db_name: str, 
        metadata: FieldMetadata
    ) -> FieldMetadata:
        """
        Add SME description to metadata.
        
        Args:
            db_name: Database name (e.g., "superhero")
            metadata: FieldMetadata with LLM descriptions
            
        Returns:
            Updated FieldMetadata with sme_description added
        """
        profile = metadata.profile
        
        sme_desc = self.sme_parser.get_field_description(
            db_name,
            profile.table_name,
            profile.column_name
        )
        
        if sme_desc:
            metadata.sme_description = sme_desc.description
            metadata.sme_source = sme_desc.source
        
        return metadata
    
    def enrich_batch(
        self,
        db_name: str,
        metadata_list: list[FieldMetadata]
    ) -> list[FieldMetadata]:
        """
        Enrich multiple metadata objects at once.
        
        More efficient as it loads all SME descriptions once.
        """
        # Load all descriptions for database
        all_descriptions = self.sme_parser.load_database_descriptions(db_name)
        
        # Enrich each metadata
        for metadata in metadata_list:
            profile = metadata.profile
            key = (profile.table_name, profile.column_name)
            
            if key in all_descriptions:
                sme_desc = all_descriptions[key]
                metadata.sme_description = sme_desc.description
                metadata.sme_source = sme_desc.source
        
        return metadata_list