import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from src.ingestion.parsers.upstage import UpstageParser
from src.ingestion.parsers.base import ParsedDocument, ParsedElement


@pytest.fixture
def parser():
    """Create parser with dummy API key"""
    return UpstageParser(api_key="test_key")


@pytest.fixture
def mock_api_response():
    """Mock successful API response"""
    return {
        "api": "1.1",
        "billed_pages": 2,
        "elements": [
            {
                "id": 0,
                "page": 1,
                "category": "heading1",
                "text": "Test Heading",
                "html": "<h1>Test Heading</h1>",
                "bounding_box": []
            },
            {
                "id": 1,
                "page": 1,
                "category": "paragraph",
                "text": "Test paragraph content.",
                "html": "<p>Test paragraph content.</p>",
                "bounding_box": []
            },
            {
                "id": 2,
                "page": 1,
                "category": "table",
                "text": "Table data",
                "html": "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>",
                "bounding_box": []
            }
        ]
    }


def test_parser_initialization():
    """Test parser initializes correctly"""
    parser = UpstageParser(api_key="test_key")
    assert parser.api_key == "test_key"
    assert parser.timeout == 60


def test_parser_requires_api_key():
    """Test parser raises error without API key"""
    with pytest.raises(ValueError, match="API key is required"):
        UpstageParser(api_key="")


def test_supported_extensions(parser):
    """Test supported file extensions"""
    extensions = parser.get_supported_extensions()
    assert '.pdf' in extensions
    assert '.docx' in extensions
    assert '.hwp' in extensions


def test_supports_file(parser):
    """Test file support checking"""
    assert parser.supports_file("document.pdf") == True
    assert parser.supports_file("document.docx") == True
    assert parser.supports_file("document.txt") == False


def test_html_table_to_markdown(parser):
    """Test HTML to markdown conversion"""
    html = "<table><tr><td>Name</td><td>Value</td></tr><tr><td>Test</td><td>123</td></tr></table>"
    markdown = parser._html_table_to_markdown(html)
    
    expected = "| Name | Value |\n|---|---|\n| Test | 123 |"
    assert markdown == expected


def test_html_table_with_newlines(parser):
    """Test table conversion handles internal newlines"""
    html = "<table><tr><td>Multi\nLine</td><td>Value</td></tr></table>"
    markdown = parser._html_table_to_markdown(html)
    
    # Should replace newline with space
    assert "Multi Line" in markdown
    assert "\n" not in markdown.split('|')[1]  # Check cell content


def test_category_mapping(parser):
    """Test category to element type mapping"""
    assert parser._map_category("heading1") == "heading"
    assert parser._map_category("table") == "table"
    assert parser._map_category("paragraph") == "text"
    assert parser._map_category("figure") == "figure"
    assert parser._map_category("unknown") == "text"  # Default


@patch('src.ingestion.parsers.upstage.requests.post')
@patch('pathlib.Path.exists')
def test_parse_success(mock_exists, mock_post, parser, mock_api_response):
    """Test successful document parsing"""
    mock_exists.return_value = True
    mock_post.return_value = Mock(status_code=200, json=lambda: mock_api_response)
    
    with patch('builtins.open', mock_open(read_data=b'fake pdf')):
        doc = parser.parse("test.pdf")
    
    assert isinstance(doc, ParsedDocument)
    assert len(doc.elements) == 3  # heading, paragraph, table
    assert doc.metadata['billed_pages'] == 2


@patch('pathlib.Path.exists')
def test_parse_file_not_found(mock_exists, parser):
    """Test parsing non-existent file"""
    mock_exists.return_value = False
    
    with pytest.raises(FileNotFoundError):
        parser.parse("nonexistent.pdf")


@patch('pathlib.Path.exists')
def test_parse_unsupported_format(mock_exists, parser):
    """Test parsing unsupported file format"""
    mock_exists.return_value = True
    
    with pytest.raises(ValueError, match="Unsupported file type"):
        parser.parse("document.txt")


@patch('src.ingestion.parsers.upstage.requests.post')
@patch('pathlib.Path.exists')
def test_api_error_handling(mock_exists, mock_post, parser):
    """Test API error handling"""
    mock_exists.return_value = True
    mock_post.return_value = Mock(status_code=401, text="Unauthorized")
    
    with patch('builtins.open', mock_open(read_data=b'fake pdf')):
        with pytest.raises(Exception, match="Invalid API key"):
            parser.parse("test.pdf")


def test_skip_empty_elements(parser):
    """Test that empty elements are skipped"""
    elem_empty = {"id": 0, "category": "paragraph", "text": "   ", "html": ""}
    result = parser._process_element(elem_empty)
    assert result is None


def test_skip_figures(parser):
    """Test that figures are skipped"""
    elem_figure = {"id": 0, "category": "figure", "text": "", "html": "<img/>"}
    result = parser._process_element(elem_figure)
    assert result is None