import pytest
from src.generation.citation_formatter import CitationFormatter


class TestCitationFormatter:
    """Test suite for CitationFormatter"""
    
    @pytest.fixture
    def formatter(self):
        """Create a fresh formatter for each test"""
        return CitationFormatter()
    
    @pytest.fixture
    def sample_sources(self):
        """Sample sources for testing"""
        return [
            {
                'child_id': 'doc1_chunk1',
                'parent_id': 'doc1',
                'source_file': 'data/inputs/sample.pdf',
                'page_number': 5,
                'parent_type': 'text'
            },
            {
                'child_id': 'doc1_chunk2',
                'parent_id': 'doc1',
                'source_file': 'data/inputs/sample.pdf',
                'page_number': 12,
                'parent_type': 'table'
            },
            {
                'child_id': 'doc2_chunk1',
                'parent_id': 'doc2',
                'source_file': 'data/inputs/catalog.pdf',
                'page_number': 3,
                'parent_type': 'text'
            }
        ]
    
    def test_add_single_source(self, formatter, sample_sources):
        """Test adding a single source"""
        citation_num = formatter.add_source(sample_sources[0])
        
        assert citation_num == 1
        assert len(formatter.sources) == 1
        assert formatter.next_citation_num == 2
    
    def test_add_multiple_sources(self, formatter, sample_sources):
        """Test adding multiple sources sequentially"""
        citation_nums = formatter.add_sources(sample_sources)
        
        assert citation_nums == [1, 2, 3]
        assert len(formatter.sources) == 3
        assert formatter.next_citation_num == 4
    
    def test_duplicate_source_same_number(self, formatter, sample_sources):
        """Test that duplicate sources get same citation number"""
        first_num = formatter.add_source(sample_sources[0])
        second_num = formatter.add_source(sample_sources[0])
        
        assert first_num == second_num
        assert first_num == 1
        assert len(formatter.sources) == 1  # Should not duplicate
    
    def test_format_citation_marker(self, formatter):
        """Test citation marker formatting"""
        assert formatter.format_citation_marker(1) == "[1]"
        assert formatter.format_citation_marker(5) == "[5]"
        assert formatter.format_citation_marker(99) == "[99]"
    
    def test_format_citation_list_korean(self, formatter, sample_sources):
        """Test formatting citation list in Korean"""
        formatter.add_sources(sample_sources)
        citation_list = formatter.format_citation_list(language="ko")
        
        assert "참고 문서:" in citation_list
        assert "[1] sample.pdf, 5페이지" in citation_list
        assert "[2] sample.pdf, 12페이지" in citation_list
        assert "[3] catalog.pdf, 3페이지" in citation_list
    
    def test_format_citation_list_english(self, formatter, sample_sources):
        """Test formatting citation list in English"""
        formatter.add_sources(sample_sources)
        citation_list = formatter.format_citation_list(language="en")
        
        assert "Sources:" in citation_list
        assert "[1] sample.pdf, page 5" in citation_list
        assert "[2] sample.pdf, page 12" in citation_list
        assert "[3] catalog.pdf, page 3" in citation_list
    
    def test_empty_citation_list(self, formatter):
        """Test citation list when no sources added"""
        citation_list = formatter.format_citation_list()
        
        assert citation_list == ""
    
    def test_source_without_page_number(self, formatter):
        """Test source that has no page number"""
        source = {
            'child_id': 'chunk1',
            'source_file': 'document.pdf',
            'parent_type': 'text'
        }
        
        formatter.add_source(source)
        citation_list = formatter.format_citation_list(language="en")
        
        assert "[1] document.pdf" in citation_list
        assert "page" not in citation_list
    
    def test_get_citation_metadata(self, formatter, sample_sources):
        """Test getting citation metadata for frontend"""
        formatter.add_sources(sample_sources)
        metadata = formatter.get_citation_metadata()
        
        assert len(metadata) == 3
        
        # Check first citation
        assert metadata[0]['id'] == '[1]'
        assert metadata[0]['file'] == 'data/inputs/sample.pdf'
        assert metadata[0]['page'] == 5
        assert metadata[0]['source_id'] == 'doc1_chunk1'
        assert metadata[0]['type'] == 'text'
    
    def test_validate_citations_all_valid(self, formatter, sample_sources):
        """Test validation when all citations are valid"""
        formatter.add_sources(sample_sources)
        text = "PVD 코팅 [1], CVD 코팅 [2], 그리고 DLC [3]."
        
        valid, invalid = formatter.validate_citations_in_text(text)
        
        assert valid is True
        assert invalid == []
    
    def test_validate_citations_with_invalid(self, formatter, sample_sources):
        """Test validation when some citations are invalid"""
        formatter.add_sources(sample_sources[:2])  # Only add first 2
        text = "Valid [1] and [2], but [99] is invalid."
        
        valid, invalid = formatter.validate_citations_in_text(text)
        
        assert valid is False
        assert 99 in invalid
        assert 1 not in invalid
        assert 2 not in invalid
    
    def test_validate_citations_no_citations_in_text(self, formatter):
        """Test validation when text has no citations"""
        text = "This text has no citations at all."
        
        valid, invalid = formatter.validate_citations_in_text(text)
        
        assert valid is True
        assert invalid == []
    
    def test_reset_clears_all_state(self, formatter, sample_sources):
        """Test that reset completely clears formatter state"""
        formatter.add_sources(sample_sources)
        
        assert len(formatter.sources) == 3
        assert formatter.next_citation_num == 4
        assert len(formatter.citation_map) == 3
        
        formatter.reset()
        
        assert len(formatter.sources) == 0
        assert formatter.next_citation_num == 1
        assert len(formatter.citation_map) == 0
    
    def test_source_with_only_parent_id(self, formatter):
        """Test source that only has parent_id (no child_id)"""
        source = {
            'parent_id': 'parent123',
            'source_file': 'document.pdf',
            'page_number': 1
        }
        
        citation_num = formatter.add_source(source)
        
        assert citation_num == 1
        assert 'parent123' in formatter.citation_map
    
    def test_source_missing_both_ids(self, formatter):
        """Test source missing both child_id and parent_id"""
        source = {
            'source_file': 'document.pdf',
            'page_number': 1
        }
        
        citation_num = formatter.add_source(source)
        
        assert citation_num == 0  # Should return 0 for invalid source
        assert len(formatter.sources) == 0
    
    def test_filename_extraction_from_path(self, formatter):
        """Test that long paths show only filename"""
        source = {
            'child_id': 'chunk1',
            'source_file': '/very/long/path/to/file/document.pdf',
            'page_number': 5
        }
        
        formatter.add_source(source)
        citation_list = formatter.format_citation_list()
        
        # Should only show filename, not full path
        assert "document.pdf" in citation_list
        assert "/very/long/path" not in citation_list
    
    def test_multiple_citations_same_page(self, formatter):
        """Test multiple chunks from same page"""
        sources = [
            {
                'child_id': 'chunk1',
                'source_file': 'doc.pdf',
                'page_number': 5
            },
            {
                'child_id': 'chunk2',
                'source_file': 'doc.pdf',
                'page_number': 5
            }
        ]
        
        nums = formatter.add_sources(sources)
        
        # Should get different numbers even though same page
        assert nums == [1, 2]
        assert len(formatter.sources) == 2