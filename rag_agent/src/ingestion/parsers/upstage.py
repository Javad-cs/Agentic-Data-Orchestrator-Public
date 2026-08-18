import requests
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup

from .base import BaseParser, ParsedDocument, ParsedElement


class UpstageParser(BaseParser):
    """
    Upstage Layout Analysis API integration.
    
    API Version: 1.1
    Endpoint: /v1/document-ai/layout-analysis
    
    Features:
    - Element-level structure detection
    - Table detection and extraction
    - Bounding box coordinates
    - Multi-format support (PDF, DOCX, PPTX, XLSX, HWP)
    
    Usage:
        parser = UpstageParser(api_key="your_key")
        doc = parser.parse("document.pdf")
    """
    
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.xlsx', '.hwp']
    
    def __init__(self, api_key: str, timeout: int, endpoint: str, model: str):
        """
        Initialize Upstage parser.
        
        Args:
            api_key: Upstage API key
            timeout: Request timeout in seconds (default: 300)
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.timeout = timeout
        self.endpoint = endpoint
        self.model = model
    
    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions"""
        return self.SUPPORTED_EXTENSIONS
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse document using Upstage Layout Analysis API.
        
        Args:
            file_path: Path to document file
            
        Returns:
            ParsedDocument with structured elements
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format unsupported
            Exception: If API call fails
        """
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.supports_file(str(file_path)):
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        # Call API
        response_data = self._call_api(file_path)
        
        # Parse response
        return self._parse_response(response_data, file_path)
    
    def _call_api(self, file_path: Path) -> dict:
        """
        Make API call to Upstage.
        
        Raises:
            Exception: If API returns error
        """
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    self.endpoint,  # Changed from self.API_ENDPOINT
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    files={'document': (file_path.name, f, self._get_mime_type(file_path))},
                    data={'model': self.model},  # Use instance variable
                    timeout=self.timeout
                )
            
            # Check response
            if response.status_code == 401:
                raise Exception("Invalid API key (401 Unauthorized)")
            elif response.status_code == 413:
                raise Exception("File too large (413 Payload Too Large)")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded (429 Too Many Requests)")
            elif response.status_code != 200:
                raise Exception(
                    f"Upstage API error: {response.status_code} - {response.text[:200]}"
                )
            
            return response.json()
        
        except requests.exceptions.Timeout:
            raise Exception(f"API request timed out after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
    
    def _parse_response(self, response: dict, file_path: Path) -> ParsedDocument:
        """
        Response structure:
        {
        "api": "2.0",
        "model": "document-parse-251217",
        "elements": [
            {
            "id": 0,
            "page": 1,
            "category": "heading1",
            "content": {
                "html": "<h1>...</h1>",
                "text": "",
                "markdown": ""
            },
            "coordinates": [...]
            }
        ]
        }
        """
        elements_raw = response.get('elements', [])
        
        if not elements_raw:
            raise ValueError("API returned no elements (empty document?)")
        
        # Process each element
        elements = []
        for elem in elements_raw:
            try:
                parsed_elem = self._process_element(elem)
                if parsed_elem:  # Skip if None (e.g., empty figures)
                    elements.append(parsed_elem)
            except Exception as e:
                # Log but don't fail entire document
                print(f"️ Warning: Failed to process element {elem.get('id')}: {e}")
                continue
        
        # Reconstruct full text
        full_text = self._reconstruct_full_text(elements)
        
        return ParsedDocument(
            elements=elements,
            full_text=full_text,
            metadata={
                'source_file': str(file_path),
                'file_name': file_path.name,
                'api_version': response.get('api', 'unknown'),
                'billed_pages': response.get('billed_pages', 0),
                'total_elements': len(elements)
            }
        )
    
    def _process_element(self, elem: dict) -> Optional[ParsedElement]:
        """
        Process a single element from API response.
        
        Handles special cases:
        - Tables: Convert HTML to markdown
        - Figures: Skip (no text content)
        - Text: Extract from HTML if text field empty
        """
        category = elem.get('category', 'paragraph')
        element_type = self._map_category(category)
        
        # Skip figures (images have no text)
        if element_type == 'figure':
            return None
        
        # Get content object
        content_obj = elem.get('content', {})
        
        # Special handling for tables
        if element_type == 'table':
            html = content_obj.get('html', '')
            if not html:
                # Fallback: extract text
                content = self._extract_text_from_content(content_obj)
            else:
                # Convert HTML table to markdown
                content = self._html_table_to_markdown(html)
        else:
            # Extract text from content object
            content = self._extract_text_from_content(content_obj)
        
        # Skip empty elements
        if not content.strip():
            return None
        
        return ParsedElement(
            element_type=element_type,
            content=content,
            metadata={
                'id': elem.get('id'),
                'page': elem.get('page', 0),
                'category': category,
                'coordinates': elem.get('coordinates', []),  # CHANGED from bounding_box
                'html': content_obj.get('html', '')  # Keep original HTML
            }
        )
        
    def _extract_text_from_content(self, content_obj: dict) -> str:
        """
        Extract text from new API's content object.
        
        Priority:
        1. Use 'text' field if available
        2. Extract from 'html' field using BeautifulSoup
        3. Fall back to empty string
        
        Args:
            content_obj: Content object from API response
            
        Returns:
            Extracted text string
        """
        # Try text field first
        text_content = content_obj.get('text', '').strip()
        if text_content:
            return text_content
        
        # Extract from HTML
        html_content = content_obj.get('html', '')
        if html_content:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text(strip=True)
            except Exception as e:
                print(f" Warning: Failed to extract text from HTML: {e}")
                return ""
        
        return ""
    
    def _html_table_to_markdown(self, html: str) -> str:
        """
        Convert HTML table to markdown format.
        
        Critical: Replaces internal newlines with spaces to prevent
        breaking markdown table structure.
        
        Example:
            Input:  <table><tr><td>A</td><td>B</td></tr></table>
            Output: | A | B |
                    |---|---|
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return ""
            
            rows = []
            for tr in table.find_all('tr'):
                # Robust: Replace internal newlines with spaces
                cells = [
                    cell.get_text(strip=True).replace('\n', ' ')
                    for cell in tr.find_all(['td', 'th'])
                ]
                
                # Skip empty rows
                if not any(cells):
                    continue
                
                rows.append('| ' + ' | '.join(cells) + ' |')
            
            if not rows:
                return ""
            
            # Add separator after first row (header)
            num_cols = rows[0].count('|') - 1
            separator = '|' + '|'.join(['---' for _ in range(num_cols)]) + '|'
            rows.insert(1, separator)
            
            return '\n'.join(rows)
        
        except Exception as e:
            print(f"️ Warning: Failed to convert HTML table to markdown: {e}")
            # Fallback: return empty (will be skipped)
            return ""
    
    def _map_category(self, category: str) -> str:
        """
        Map Upstage category to our element types.
        
        Known Upstage categories:
        - heading1, heading2, heading3, ...
        - paragraph
        - table
        - figure
        - list
        - header (page header)
        - footer (page footer)
        """
        category_lower = category.lower()
        
        if 'heading' in category_lower:
            return 'heading'
        elif category_lower == 'table':
            return 'table'
        elif category_lower == 'figure':
            return 'figure'
        elif category_lower == 'list':
            return 'list'
        elif category_lower in ['paragraph', 'header', 'footer']:
            return 'text'
        else:
            # Default: treat as text
            return 'text'
    
    def _reconstruct_full_text(self, elements: List[ParsedElement]) -> str:
        """
        Reconstruct full document text from elements.
        
        Adds spacing between elements for readability.
        """
        text_parts = []
        
        for elem in elements:
            if elem.content.strip():
                text_parts.append(elem.content)
        
        # Join with double newlines for spacing
        return '\n\n'.join(text_parts)
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for file upload"""
        mime_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.hwp': 'application/x-hwp'
        }
        return mime_types.get(file_path.suffix.lower(), 'application/octet-stream')