from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any
from pathlib import Path

@dataclass
class ParsedElement:
    """
    Represents a single parsed document element.
    
    This is the atomic unit returned by parsers.
    """
    element_type: Literal["text", "table", "heading", "figure", "list"]
    content: str  # Text content or markdown (for tables)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"ParsedElement(type={self.element_type}, content='{preview}')"


@dataclass
class ParsedDocument:
    """
    Complete parsed document with structured elements.
    
    Contains both individual elements and reconstructed full text.
    """
    elements: List[ParsedElement]
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"ParsedDocument(elements={len(self.elements)}, metadata={self.metadata})"
    
    def get_elements_by_type(self, element_type: str) -> List[ParsedElement]:
        """Filter elements by type"""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_tables(self) -> List[ParsedElement]:
        """Get all table elements"""
        return self.get_elements_by_type("table")
    
    def get_text_elements(self) -> List[ParsedElement]:
        """Get all text elements (excluding tables, figures)"""
        return [e for e in self.elements if e.element_type in ["text", "heading", "list"]]


class BaseParser(ABC):
    """
    Abstract base class for document parsers.
    
    All parsers must implement the parse() method.
    """
    
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a document file into structured elements.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParsedDocument with elements and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
            Exception: For API or parsing errors
        """
        pass
    
    def supports_file(self, file_path: str) -> bool:
        """
        Check if this parser supports the given file type.
        
        Default implementation checks file extension.
        Override for custom logic.
        """
        supported_extensions = self.get_supported_extensions()
        file_ext = Path(file_path).suffix.lower()
        return file_ext in supported_extensions
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (e.g., ['.pdf', '.docx'])"""
        pass