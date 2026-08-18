"""
Upstage Async Document Parser for large files.

Handles documents up to 1000 pages using Upstage's async API.
Files are processed in 10-page batches.
"""

import requests
import time
import logging
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup

from .base import BaseParser, ParsedDocument, ParsedElement
from src.config.models import UpstageConfig

logger = logging.getLogger(__name__)


class UpstageAsyncParser(BaseParser):
    """
    Upstage Async API integration for large documents.
    
    Features:
    - Handles up to 1000 pages
    - Processes in 10-page batches
    - Automatic polling and merging
    - Same output format as sync parser
    
    Usage:
        parser = UpstageAsyncParser(config)
        doc = parser.parse("large_document.pdf")
    """
    
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.xlsx', '.hwp']
    
    def __init__(
        self,
        api_key: str,
        async_endpoint: str,
        model: str = "document-parse",
        poll_interval: int = 10,
        max_wait_time: int = 3600
    ):
        """
        Initialize async parser.
        
        Args:
            api_key: Upstage API key
            async_endpoint: Async API endpoint URL
            model: Model to use (document-parse)
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait for completion
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.async_endpoint = async_endpoint
        self.model = model
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
    
    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions"""
        return self.SUPPORTED_EXTENSIONS
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse document using async API.
        
        Workflow:
        1. Submit document → get request_id
        2. Poll status until completed
        3. Download all batch results
        4. Merge batches into single ParsedDocument
        
        Args:
            file_path: Path to document file
            
        Returns:
            ParsedDocument with merged elements from all batches
            
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
        
        logger.info(f" Starting async parsing: {file_path.name}")
        
        # Step 1: Submit document
        request_id = self._submit_document(file_path)
        logger.info(f" Document submitted. Request ID: {request_id}")
        
        # Step 2: Poll until complete
        result = self._poll_until_complete(request_id)
        logger.info(f" Processing complete. Total pages: {result['total_pages']}")
        
        # Step 3: Download and merge batches
        all_elements = self._download_and_merge_batches(result)
        logger.info(f" Downloaded {len(all_elements)} elements from {len(result['batches'])} batches")
        
        # Step 4: Create ParsedDocument
        return self._create_parsed_document(all_elements, file_path, result)
    
    def _submit_document(self, file_path: Path) -> str:
        """
        Submit document for async processing.
        
        Returns:
            request_id for polling
        """
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    self.async_endpoint,
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    files={'document': (file_path.name, f, self._get_mime_type(file_path))},
                    data={'model': self.model}
                )
            
            if response.status_code != 200:
                error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                raise Exception(f"Async submission failed: {response.status_code} - {error_detail}")
            
            result = response.json()
            return result['request_id']
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error during submission: {str(e)}")
    
    def _poll_until_complete(self, request_id: str) -> dict:
        """
        Poll status endpoint until processing completes.
        
        Returns:
            Full result object with batches
        """
        poll_url = f"https://api.upstage.ai/v1/document-digitization/requests/{request_id}"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        start_time = time.time()
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.max_wait_time:
                raise TimeoutError(
                    f"Async processing timeout after {elapsed:.0f}s. "
                    f"Request ID: {request_id} (may still be processing on server)"
                )
            
            # Poll status
            response = requests.get(poll_url, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Status polling failed: {response.status_code}")
            
            result = response.json()
            status = result['status']
            
            if status == 'completed':
                return result
            
            elif status == 'failed':
                failure_msg = result.get('failure_message', 'Unknown error')
                raise Exception(f"Async processing failed: {failure_msg}")
            
            elif status in ['scheduled', 'started']:
                # Still processing
                completed = result.get('completed_pages', 0)
                total = result.get('total_pages', '?')
                logger.info(f" Processing... {completed}/{total} pages complete")
                time.sleep(self.poll_interval)
            
            else:
                # Unknown status
                logger.warning(f"Unknown status: {status}. Continuing to poll...")
                time.sleep(self.poll_interval)
    
    def _download_and_merge_batches(self, result: dict) -> List[dict]:
        """
        Download all batch results and merge elements.
        
        Args:
            result: Full result object from polling
            
        Returns:
            List of all elements from all batches, sorted by page
        """
        all_elements = []
        
        for batch in result['batches']:
            if batch['status'] != 'completed':
                logger.warning(
                    f" Batch {batch['id']} (pages {batch['start_page']}-{batch['end_page']}) "
                    f"not completed: {batch['status']}"
                )
                continue
            
            # Download batch
            download_url = batch['download_url']
            try:
                response = requests.get(download_url)
                response.raise_for_status()
                batch_data = response.json()
                
                # Extract elements
                batch_elements = batch_data.get('elements', [])
                all_elements.extend(batch_elements)
                
                logger.info(
                    f" Batch {batch['id']}: {len(batch_elements)} elements "
                    f"(pages {batch['start_page']}-{batch['end_page']})"
                )
            
            except Exception as e:
                logger.error(f" Failed to download batch {batch['id']}: {e}")
                raise
        
        # Sort by page number for proper order
        all_elements.sort(key=lambda e: (e.get('page', 0), e.get('id', 0)))
        
        return all_elements
    
    def _create_parsed_document(
        self,
        elements_raw: List[dict],
        file_path: Path,
        result: dict
    ) -> ParsedDocument:
        """
        Convert merged elements to ParsedDocument.
        
        Uses same logic as sync parser.
        """
        if not elements_raw:
            raise ValueError("No elements received from async API")
        
        # Process each element (same as sync parser)
        elements = []
        for elem in elements_raw:
            try:
                parsed_elem = self._process_element(elem)
                if parsed_elem:
                    elements.append(parsed_elem)
            except Exception as e:
                logger.warning(f" Failed to process element {elem.get('id')}: {e}")
                continue
        
        # Reconstruct full text
        full_text = self._reconstruct_full_text(elements)
        
        return ParsedDocument(
            elements=elements,
            full_text=full_text,
            metadata={
                'source_file': str(file_path),
                'file_name': file_path.name,
                'api_version': result.get('model', 'async'),
                'total_pages': result.get('total_pages', 0),
                'total_elements': len(elements),
                'request_id': result.get('id', ''),
                'processing_time': result.get('completed_at', '')
            }
        )
    
    def _process_element(self, elem: dict) -> Optional[ParsedElement]:
        """Process element (same logic as sync parser)"""
        category = elem.get('category', 'paragraph')
        element_type = self._map_category(category)
        
        if element_type == 'figure':
            return None
        
        content_obj = elem.get('content', {})
        
        if element_type == 'table':
            html = content_obj.get('html', '')
            if not html:
                content = self._extract_text_from_content(content_obj)
            else:
                content = self._html_table_to_markdown(html)
        else:
            content = self._extract_text_from_content(content_obj)
        
        if not content.strip():
            return None
        
        return ParsedElement(
            element_type=element_type,
            content=content,
            metadata={
                'id': elem.get('id'),
                'page': elem.get('page', 0),
                'category': category,
                'coordinates': elem.get('coordinates', []),
                'html': content_obj.get('html', '')
            }
        )
    
    def _extract_text_from_content(self, content_obj: dict) -> str:
        """Extract text from content object"""
        text_content = content_obj.get('text', '').strip()
        if text_content:
            return text_content
        
        html_content = content_obj.get('html', '')
        if html_content:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text(strip=True)
            except Exception as e:
                logger.warning(f" Failed to extract text from HTML: {e}")
                return ""
        
        return ""
    
    def _html_table_to_markdown(self, html: str) -> str:
        """Convert HTML table to markdown"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return ""
            
            rows = []
            for tr in table.find_all('tr'):
                cells = [
                    cell.get_text(strip=True).replace('\n', ' ')
                    for cell in tr.find_all(['td', 'th'])
                ]
                
                if not any(cells):
                    continue
                
                rows.append('| ' + ' | '.join(cells) + ' |')
            
            if not rows:
                return ""
            
            num_cols = rows[0].count('|') - 1
            separator = '|' + '|'.join(['---' for _ in range(num_cols)]) + '|'
            rows.insert(1, separator)
            
            return '\n'.join(rows)
        
        except Exception as e:
            logger.warning(f" Failed to convert table to markdown: {e}")
            return ""
    
    def _map_category(self, category: str) -> str:
        """Map category to element type"""
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
            return 'text'
    
    def _reconstruct_full_text(self, elements: List[ParsedElement]) -> str:
        """Reconstruct full document text"""
        text_parts = []
        
        for elem in elements:
            if elem.content.strip():
                text_parts.append(elem.content)
        
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