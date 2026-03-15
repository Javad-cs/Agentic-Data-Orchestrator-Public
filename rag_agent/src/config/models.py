from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from urllib.parse import quote_plus  # For secure URL encoding

# ============= UPSTAGE (UNIFIED) =============
class UpstageConfig(BaseModel):
    """Unified Upstage API configuration"""
    api_key: str = Field(...)
    upstage_timeout: int = Field(default=300, ge=60, description="API timeout in seconds for large PDFs")
    
    # Document Parse (updated endpoint) [sync]
    parse_endpoint: str = Field(
        default="https://api.upstage.ai/v1/document-digitization",
        description="Upstage document parsing endpoint"
    )
    parse_model: str = Field(
        default="document-parse",
        description="Model to use for parsing (document-parse, ocr)"
    )
    
    # Async Document Parse (for large files)
    async_endpoint: str = Field(
        default="https://api.upstage.ai/v1/document-digitization/async",
        description="Async endpoint for large documents"
    )
    async_poll_interval: int = Field(
        default=10,
        ge=5,
        description="Seconds between status checks for async jobs"
    )
    async_max_wait_time: int = Field(
        default=3600,
        description="Maximum time to wait for async job completion (seconds)"
    )
    
    # Embeddings (FIXED)
    embedding_model_passage: str = Field(
        default="solar-embedding-1-large-passage",  # For indexing documents
        description="Embedding model for documents (passage)"
    )
    embedding_model_query: str = Field(
        default="solar-embedding-1-large-query",  # For search queries
        description="Embedding model for queries"
    )
    embedding_dimension: int = Field(
        default=4096,
        description="Embedding vector dimension"
    )
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        description="Batch size for embedding calls"
    )
    
    # Reranking
    reranking_model: str = Field(
        default="solar-rerank-1",
        description="Reranking model name"
    )
    reranking_enabled: bool = Field(
        default=False,
        description="Enable reranking"
    )


# ============= LLM (GENERATION) =============
class LLMConfig(BaseModel):
    """LLM configuration for generation (adapted from text_to_sql)"""
    
    # Provider
    provider: Literal["azure", "openai"] = Field(
        default="azure",
        description="LLM provider (azure or openai)"
    )
    
    # Azure OpenAI Configuration
    azure_endpoint: str = Field(
        ...,
        description="Azure OpenAI endpoint URL"
    )
    azure_api_key: str = Field(
        ...,
        description="Azure OpenAI API key"
    )
    azure_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version"
    )
    
    # Model Configuration
    default_model: str = Field(
        default="gpt-4-5-mini",
        description="Default model for Fast Lane (speed optimized)"
    )
    fallback_model: str = Field(
        default="gpt-4-1",
        description="Fallback model for Slow Lane (quality optimized)"
    )
    
    # Generation Parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, 2=very random)"
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4096,
        description="Maximum tokens in generated response"
    )
    streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    
    # Performance
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        description="Maximum concurrent LLM requests"
    )
    request_timeout: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds"
    )
    
    # Query Expansion (for future use)
    expansion_temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Temperature for query expansion (higher = more diverse)"
    )


# ============= SAFETY CHECK =============
class SafetyCheckConfig(BaseModel):
    """Safety check configuration for Fast Lane"""
    enabled: bool = Field(
        default=True,
        description="Enable safety checks before returning answer"
    )
    
    # Heuristic checks
    check_no_answer_phrases: bool = Field(
        default=False,
        description="Check for 'I don't know' type phrases"
    )
    check_citation_presence: bool = Field(
        default=True,
        description="Require at least one citation in answer"
    )
    min_answer_length: int = Field(
        default=10,
        ge=0,
        description="Minimum answer length in characters"
    )
    
    # NLI check
    use_nli: bool = Field(
        default=True,
        description="Use NLI model for entailment checking"
    )
    nli_model: str = Field(
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        description="HuggingFace NLI model (multilingual - supports Korean + English)"
    )
    nli_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum entailment score to pass (0-1)"
    )
    
    # Fallback behavior
    on_failure: Literal["return_partial", "route_to_slow", "return_error"] = Field(
        default="return_partial",
        description="What to do when safety check fails"
    )


# ============= BM25 =============
class BM25Config(BaseModel):
    """BM25 sparse retrieval configuration"""
    enabled: bool = Field(
        default=True,
        description="Enable BM25 for hybrid search"
    )
    k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 k1 parameter (term frequency saturation)"
    )
    b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter (length normalization)"
    )
    storage: Literal["postgresql", "memory"] = Field(
        default="postgresql",
        description="Where to store BM25 index"
    )


# ============= CHUNKING =============
class TableHandlingConfig(BaseModel):
    """Table-specific chunking parameters"""
    parent_max_tokens: int = Field(default=2000)
    child_target_tokens: int = Field(default=350)
    min_rows_per_child: int = Field(default=3, ge=1)
    max_rows_per_child: int = Field(default=7, ge=1)
    
    @field_validator('child_target_tokens')
    @classmethod
    def validate_child_size(cls, v, info):
        parent_max = info.data.get('parent_max_tokens', 2000)
        if v >= parent_max:
            raise ValueError(f"child_target_tokens ({v}) must be < parent_max_tokens ({parent_max})")
        return v


class TextChunkingConfig(BaseModel):
    """Text-specific chunking parameters"""
    parent_max_tokens: int = Field(default=2000)
    child_chunk_size: int = Field(default=400)
    child_overlap: int = Field(default=50, ge=0)


class ChunkingConfig(BaseModel):
    """Master chunking configuration"""
    tokenizer_model: str = Field(
        default="cl100k_base",
        description="Tokenizer for chunking (must match embedder)"
    )
    text: TextChunkingConfig = Field(default_factory=TextChunkingConfig)
    table: TableHandlingConfig = Field(default_factory=TableHandlingConfig)


# ============= INGESTION =============
class IngestionConfig(BaseModel):
    """Ingestion pipeline configuration"""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Documents to process in parallel"
    )


# ============= QUERY EXPANSION =============
class QueryExpansionConfig(BaseModel):
    """Query expansion configuration"""
    enabled: bool = Field(default=True)
    num_variants: int = Field(default=3, ge=1, le=5)
    parallel: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


# ============= RETRIEVAL =============
class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    dense_top_k: int = Field(default=12, ge=1)
    bm25_top_k: int = Field(default=12, ge=1)
    rrf_k: int = Field(default=60, ge=1)


# ============= RERANKING =============
class RerankerConfig(BaseModel):
    """Reranker configuration"""
    enabled: bool = Field(default=False, description="Enable reranking")
    provider: Literal["cohere", "none"] = Field(default="cohere")
    
    # Cohere via Azure AI Foundry
    cohere_api_key: str = Field(default="", description="Cohere API key")
    cohere_base_url: str = Field(
        default="",
        description="Azure endpoint base URL"
    )
    cohere_model: str = Field(
        default="Cohere-rerank-v4.0-fast",
        description="Model/deployment name"
    )
    
    top_n: int = Field(default=5, ge=1, le=20, description="Number of results to return after reranking")


# ============= FAST LANE =============
class FastLaneConfig(BaseModel):
    """Fast lane pipeline configuration"""
    query_expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    safety_check: SafetyCheckConfig = Field(default_factory=SafetyCheckConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    parent_context_limit: int = Field(default=5, ge=1)
    streaming: bool = Field(
        default=True,
        description="Stream responses"
    )


# ============= SLOW LANE =============
class SlowLaneConfig(BaseModel):
    """Slow lane agent configuration"""
    max_iterations: int = Field(default=5, ge=1)
    tool_timeout_seconds: int = Field(default=30, ge=1)
    enable_self_correction: bool = Field(default=True)
    streaming: bool = Field(
        default=True,
        description="Stream intermediate steps"
    )


# ============= ROUTER =============
class RouterConfig(BaseModel):
    """
    Router configuration.
    
    Added model_name for performance optimization.
    Router should use a fast, cheap model  rather than
    the main system model to avoid routing overhead.
    """
    enabled: bool = Field(default=True, description="Enable routing")
    
    model_name: str = Field(
        default="gpt-4o-mini",  # Fast model for routing
        description="LLM model for routing decisions (use fast/cheap model)"
    )
    
    default_lane: Literal["fast", "slow"] = Field(
        default="fast",
        description="Default lane when routing fails"
    )
    
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="LLM temperature for routing (lower = more consistent)"
    )
    
    class Config:
        frozen = True


# ============= DATABASE =============
class DatabaseConfig(BaseModel):
    """
    Database connection configuration.
    
    Improved password handling and URL encoding.
    """
    # Milvus
    milvus_uri: str = Field(
        default="http://localhost:19530",
        description="Milvus connection URI"
    )
    milvus_collection_name: str = Field(
        default="rag_children",
        description="Milvus collection name"
    )
    
    # PostgreSQL connection (loaded from env)
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_database: str = Field(default="rag_db")
    postgres_user: str = Field(default="rag_user")
    postgres_password: str = Field(
        default="",  # Empty default for dev, should be set in production
        description="PostgreSQL password (set via POSTGRES_PASSWORD env var)"
    )
    postgres_pool_min_size: int = Field(default=10, ge=1)
    postgres_pool_max_size: int = Field(default=50, ge=1)
    
    @property
    def postgres_dsn(self) -> str:
        """
        Construct PostgreSQL DSN with URL encoding.
        
        Uses quote_plus to handle special characters in password.
        Example: password "pass@word:123" is safely encoded.
        """
        # URL-encode username and password to handle special chars
        encoded_user = quote_plus(self.postgres_user)
        encoded_password = quote_plus(self.postgres_password)
        
        return (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )
    
    @model_validator(mode='after')
    def validate_password(self):
        """
        Warn if password is empty (common production mistake).
        
        In production, connecting with empty password is usually an error.
        This validator warns but doesn't fail (for dev flexibility).
        """
        import os
        
        # Only warn in production
        if os.getenv("ENVIRONMENT", "development") == "production":
            if not self.postgres_password:
                import warnings
                warnings.warn(
                    "PostgreSQL password is empty in production environment. "
                    "Set DATABASE__POSTGRES_PASSWORD environment variable.",
                    UserWarning
                )
        
        return self


# ============= SYSTEM (ROOT) =============
class SystemConfig(BaseSettings):
    """
    Main system configuration.
    
    Environment variable mapping:
    - UPSTAGE_API_KEY → upstage_api_key
    - LLM__AZURE_ENDPOINT → llm.azure_endpoint
    - LLM__AZURE_API_KEY → llm.azure_api_key
    - DATABASE__POSTGRES_HOST → database.postgres_host
    - ROUTER__MODEL_NAME → router.model_name 
    
    The double underscore (__) allows nested attribute access.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )
    
    # Upstage (unified, loaded from env)
    upstage_api_key: str = Field(
        ...,
        description="Upstage API key (shared across all services)"
    )
    upstage: Optional[UpstageConfig] = Field(
        default=None,
        description="Upstage configuration (auto-populated from upstage_api_key)"
    )
    
    # LLM (for generation)
    llm: LLMConfig = Field(
        ...,
        description="LLM configuration for generation"
    )
    
    # Sub-configs
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    fast_lane: FastLaneConfig = Field(default_factory=FastLaneConfig)
    slow_lane: SlowLaneConfig = Field(default_factory=SlowLaneConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    environment: Literal["development", "production", "test"] = Field(default="development")
    
    @model_validator(mode='after')
    def populate_upstage_config(self):
        """
        Auto-populate upstage config from upstage_api_key.
        
        This runs after all fields are validated.
        """
        if self.upstage is None:
            self.upstage = UpstageConfig(api_key=self.upstage_api_key)
        elif self.upstage.api_key != self.upstage_api_key:
            # If upstage was explicitly set, ensure api_key matches
            self.upstage.api_key = self.upstage_api_key
        
        return self