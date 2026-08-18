"""
Router configuration using Pydantic Settings.

Loads configuration from environment variables with validation.
"""
from pathlib import Path
from typing import Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterSettings(BaseSettings):
    """
    Router configuration with environment variable support.
    
    All settings can be overridden via environment variables.
    Prefix: None (direct mapping)
    
    """
    
    # Azure OpenAI Configuration
    llm__azure_endpoint: str = Field(
        ...,
        alias="LLM__AZURE_ENDPOINT",
        description="Azure OpenAI endpoint URL"
    )
    llm__azure_api_key: str = Field(
        ...,
        alias="LLM__AZURE_API_KEY", 
        description="Azure OpenAI API key"
    )
    llm__azure_api_version: str = Field(
        default="2024-10-01-preview",
        alias="LLM__AZURE_API_VERSION",
        description="Azure OpenAI API version"
    )
    
    # Router Model Configuration
    router_model: str = Field(
        default="gpt-4.1-mini",
        description="Model to use for routing decisions (fast, cheap recommended)"
    )
    router_temperature: float = Field(
        default=0.0,
        description="Temperature for routing (0.0 for deterministic)"
    )
    router_max_tokens: int = Field(
        default=500,
        description="Max tokens for routing response"
    )
    
    # Paths
    routing_rules_path: str = Field(
        default="config/routing_rules.yaml",
        description="Path to routing rules YAML file"
    )
    log_dir: str = Field(
        default="logs",
        description="Directory for log files"
    )
    
    # Database Configuration (for SQL Tool)
    # Oracle Database Settings (reads from global .env)
    PRIMARY_DB_NAME: str = Field(default="PRIMARY_DATABASE")
    PRIMARY_DSN: str = Field(...)
    PRIMARY_USER: str = Field(...)
    PRIMARY_PASSWORD: str = Field(...)
    
    LINKED_DB_NAME: str = Field(default="LINKED_DATABASE")
    LINKED_DSN: str = Field(...)
    LINKED_USER: str = Field(...)
    LINKED_PASSWORD: str = Field(...)
    LINKED_SUFFIX: str = Field(default="@LINKED_DATABASE")
    DB_TYPE: str = Field(default="oracle", description="Database type (oracle, postgres, mysql)")
    
    USE_DOCKER: bool = Field(default=False)
    
    @computed_field
    @property
    def primary_connection(self) -> tuple[str, str, str]:
        """Get primary DB connection with Docker-aware DSN."""
        dsn = self.PRIMARY_DSN
        if self.USE_DOCKER and "127.0.0.1" in dsn:
            dsn = dsn.replace("127.0.0.1", "host.docker.internal")
        return (dsn, self.PRIMARY_USER, self.PRIMARY_PASSWORD)
    
    # Optional: Agent paths (for imports)
    text_to_sql_agent_path: Optional[str] = Field(
        default="../text_to_sql_agent",
        description="Path to text_to_sql_agent directory"
    )
    rag_agent_path: Optional[str] = Field(
        default="../rag_agent",
        description="Path to rag_agent directory"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env
    )


# Singleton pattern for settings
_settings: Optional[RouterSettings] = None


def get_settings() -> RouterSettings:
    """
    Get or create settings singleton.
    
    Returns:
        RouterSettings instance loaded from environment
        
    Raises:
        ValidationError: If required settings are missing
    """
    global _settings
    if _settings is None:
        _settings = RouterSettings()
    return _settings


def reload_settings() -> RouterSettings:
    """
    Force reload settings from environment.
    
    Useful for testing or when .env changes.
    
    Returns:
        Fresh RouterSettings instance
    """
    global _settings
    _settings = None
    return get_settings()