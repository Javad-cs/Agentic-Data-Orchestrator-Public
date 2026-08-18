"""Indexing utilities for LSH and FAISS-based matching."""

from .shingling import create_shingles, normalize_text, stable_hash
from .lsh_matcher import LexicalLSHMatcher, MatchResult
from .schema_matcher import SchemaLiteralMatcher, SchemaMatch, NUMERIC_TYPES
from .sql_utils import escape_sql_identifier, format_sql_literal, build_where_clause
from .embeddings import EmbeddingModel, SentenceTransformerModel, create_embedding_model
from .field_index import FieldIndex, FieldIndexEntry, SemanticMatch

__all__ = [
    "create_shingles",
    "normalize_text",
    "stable_hash",
    "LexicalLSHMatcher",
    "MatchResult",
    "SchemaLiteralMatcher",
    "SchemaMatch",
    "NUMERIC_TYPES",
    "escape_sql_identifier",
    "format_sql_literal",
    "build_where_clause",
    "EmbeddingModel",
    "SentenceTransformerModel",
    "create_embedding_model",
    "FieldIndex",
    "FieldIndexEntry",
    "SemanticMatch",
]