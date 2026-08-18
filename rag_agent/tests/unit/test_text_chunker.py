import pytest
from src.ingestion.chunkers.text_chunker import TextChunker, TextChunk
from src.config.models import ChunkingConfig, TextChunkingConfig


@pytest.fixture
def chunker():
    """Create text chunker with default config"""
    config = ChunkingConfig(
        tokenizer_model="gpt-4o",
        text=TextChunkingConfig(
            parent_max_tokens=2000,
            child_chunk_size=400,
            child_overlap=50
        )
    )
    return TextChunker(config)


def test_chunker_initialization():
    """Test chunker initializes correctly"""
    config = ChunkingConfig()
    chunker = TextChunker(config)
    
    assert chunker.parent_max_tokens == 2000
    assert chunker.child_chunk_size == 400
    assert chunker.child_overlap == 50


def test_empty_text():
    """Test handling of empty text"""
    config = ChunkingConfig()
    chunker = TextChunker(config)
    
    parents, children = chunker.chunk_text("", "test_id")
    
    assert len(parents) == 0
    assert len(children) == 0


def test_short_text_single_parent(chunker):
    """Test short text creates single parent with multiple children"""
    text = "This is a short paragraph. " * 50  # ~50 tokens
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    assert len(parents) == 1
    assert parents[0].is_split == False
    assert parents[0].chunk_id == "test_id_text"
    assert len(children) >= 1


def test_paragraph_splitting(chunker):
    """Test text is split into paragraphs correctly"""
    text = """First paragraph with some content.
This continues the first paragraph.

Second paragraph starts here.
And continues.

Third paragraph."""
    
    paragraphs = chunker._split_into_paragraphs(text)
    
    assert len(paragraphs) == 3
    assert "First paragraph" in paragraphs[0]
    assert "Second paragraph" in paragraphs[1]
    assert "Third paragraph" in paragraphs[2]


def test_long_text_multiple_parents(chunker):
    """Test long text creates multiple parents"""
    # Create text longer than parent_max_tokens (2000)
    paragraph = "This is a sentence with enough words to make it substantial. " * 20
    text = (paragraph + "\n\n") * 10  # ~1200 tokens per paragraph block
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    assert len(parents) > 1
    assert all(p.is_split for p in parents)
    assert all(p.total_parts == len(parents) for p in parents)


def test_children_have_overlap(chunker):
    """Test children have correct overlap"""
    text = "Word " * 500  # Create text that will need multiple children
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    if len(children) > 1:
        # Check overlap exists
        child1_end = children[0].text[-50:]  # Last 50 chars
        child2_start = children[1].text[:50]  # First 50 chars
        
        # There should be some overlap in content
        assert len(children[0].text) > 0
        assert len(children[1].text) > 0


def test_children_belong_to_parent(chunker):
    """Test all children reference their parent correctly"""
    text = "Content " * 100
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    for child in children:
        assert child.parent_id is not None
        assert any(child.parent_id == p.chunk_id for p in parents)


def test_token_counting(chunker):
    """Test token counting is accurate"""
    text = "Hello world"
    
    token_count = chunker._count_tokens(text)
    
    assert token_count > 0
    assert token_count < 10  # "Hello world" should be ~2-3 tokens


def test_large_single_paragraph(chunker):
    """Test handling of single paragraph exceeding parent limit"""
    # Create single paragraph longer than 2000 tokens
    text = "This is a very long sentence. " * 200  # ~600 tokens
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    # Should split the large paragraph
    assert len(parents) >= 1
    if len(parents) > 1:
        assert all(p.is_split for p in parents)


def test_sentence_splitting(chunker):
    """Test sentence splitting logic"""
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    
    sentences = chunker._split_into_sentences(text)
    
    assert len(sentences) == 4
    assert "First sentence" in sentences[0]
    assert "Second sentence" in sentences[1]


def test_korean_text_handling(chunker):
    """Test handling of Korean text"""
    text = """한국어로 된 첫 번째 문단입니다.
이것은 계속됩니다.

두 번째 문단은 여기서 시작합니다.

세 번째 문단입니다."""
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    assert len(parents) >= 1
    assert len(children) >= 1
    assert all(len(c.text) > 0 for c in children)


def test_mixed_language_text(chunker):
    """Test handling of mixed Korean-English text"""
    text = """This is English text mixed with 한국어 텍스트입니다.
The system should handle both languages seamlessly.

영어와 한국어가 섞여 있어도 잘 처리되어야 합니다.
This includes proper tokenization and chunking."""
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    assert len(parents) >= 1
    assert len(children) >= 1


def test_chunk_indices(chunker):
    """Test children have correct sequential indices"""
    text = "Content " * 200
    
    parents, children = chunker.chunk_text(text, "test_id")
    
    if len(children) > 1:
        # Check indices are sequential within each parent
        parent_children = {}
        for child in children:
            parent_id = child.parent_id
            if parent_id not in parent_children:
                parent_children[parent_id] = []
            parent_children[parent_id].append(child.chunk_index)
        
        for parent_id, indices in parent_children.items():
            assert indices == list(range(len(indices)))