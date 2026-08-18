"""
Configuration management using Pydantic Settings.
Loads from .env file and provides type-safe config access.

IMPORTANT: This file contains NO secrets and is safe to commit to Git.
All sensitive values must be provided via .env file (see .env.example).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field, field_validator
from typing import Optional, Literal
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==================== Azure OpenAI Configuration ====================
    azure_openai_endpoint: str = Field(
        ...,  # Required - no default
        description="The Azure OpenAI endpoint URL"
    )
    azure_openai_key: str = Field(
        ...,  # Required - no default
        description="Azure OpenAI API Key"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version"
    )
    
    # ==================== Model Configuration ====================
    default_model: str = Field(
        default="gpt-4-5-mini",
        description="Default model for most operations"
    )
    fallback_model: str = Field(
        default="gpt-4-1",
        description="Fallback model for complex operations"
    )
    
    # ==================== Database Type Configuration ====================
    db_type: Literal["oracle", "postgres", "mysql", "sqlite"] = Field(
        default="oracle",
        description="Database type (oracle, postgres, mysql, sqlite)"
    )
    db_dialect: str = Field(
        default="oracle",
        description="SQL dialect for parser (sqlglot dialect name)"
    )
    
    # ==================== Primary Database Configuration ====================
    primary_db_name: str = Field(
        default="PRIMARY_DATABASE",
        description="Primary database name (non-sensitive)"
    )
    primary_dsn: str = Field(
        ...,  # Required - no default
        description="Primary database DSN/connection string"
    )
    primary_user: str = Field(
        ...,  # Required - no default
        description="Primary database username"
    )
    primary_password: str = Field(
        ...,  # Required - no default
        description="Primary database password"
    )
    
    # ==================== Linked Database Configuration ====================
    linked_db_name: str = Field(
        default="LINKED_DATABASE",
        description="Linked database name (non-sensitive)"
    )
    linked_dsn: str = Field(
        ...,  # Required - no default
        description="Linked database DSN/connection string"
    )
    linked_user: str = Field(
        ...,  # Required - no default
        description="Linked database username"
    )
    linked_password: str = Field(
        ...,  # Required - no default
        description="Linked database password"
    )
    linked_suffix: str = Field(
        default="@LINKED_DATABASE",
        description="Suffix for database link references (e.g., @LINKED_DATABASE)"
    )
    
    # ==================== Runtime Configuration ====================
    use_docker: bool = Field(
        default=False,
        description="Whether running inside Docker (affects host resolution)"
    )
    
    # ==================== BIRD Dataset Configuration ====================
    bird_data_path: Path = Field(
        default=Path("./bird_data/dev/dev_databases"),
        description="Path to BIRD dev databases"
    )
    bird_root_path: Path = Field(
        default=Path("../bird_dataset/dev_20240627"),
        description="Path to BIRD dev_20240627 root"
    )
    
    # ==================== Profiling Configuration ====================
    profile_sample_size: int = Field(
        default=10000,
        description="Number of distinct values to sample for profiling"
    )
    profile_top_k: int = Field(
        default=10,
        description="Number of top values to collect"
    )
    
    # ==================== Performance ====================
    max_concurrent_requests: int = Field(
        default=5,
        description="Maximum concurrent async requests"
    )
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    
    # ==================== Computed Properties ====================
    
    @computed_field
    @property
    def primary_connection(self) -> tuple[str, str, str]:
        """
        Get primary database connection tuple (DSN, user, password).
        
        Automatically adjusts DSN for Docker environment.
        
        Returns:
            Tuple of (dsn, username, password)
        """
        dsn = self.primary_dsn
        if self.use_docker and "127.0.0.1" in dsn:
            dsn = dsn.replace("127.0.0.1", "host.docker.internal")
        return (dsn, self.primary_user, self.primary_password)
    
    @computed_field
    @property
    def linked_connection(self) -> tuple[str, str, str]:
        """
        Get linked database connection tuple (DSN, user, password).
        
        Automatically adjusts DSN for Docker environment.
        
        Returns:
            Tuple of (dsn, username, password)
        """
        dsn = self.linked_dsn
        if self.use_docker and "127.0.0.1" in dsn:
            dsn = dsn.replace("127.0.0.1", "host.docker.internal")
        return (dsn, self.linked_user, self.linked_password)
    
    @computed_field
    @property
    def db_specific_instructions(self) -> str:
        """
        Get database-specific SQL generation instructions for LLM prompts.
        
        Returns:
            String with syntax rules for the configured database type
        """
        if self.db_type == "oracle":
            return f"""
Database-specific syntax ({self.db_type.upper()}):
- Use database links: TableName{self.linked_suffix} for tables in {self.linked_db_name}
- Date format: TO_DATE(field, 'YYYYMMDD') or TO_CHAR(date_field, 'YYYYMMDD')
- String concatenation: field1 || field2
- Current date: SYSDATE
- Null handling: NVL(field, default_value)
- String comparison: Use LIKE for pattern matching
"""
        elif self.db_type == "postgres":
            return """
Database-specific syntax (POSTGRESQL):
- Use schemas: schema_name.table_name
- Date format: field::date or TO_DATE(field, 'YYYYMMDD')
- String concatenation: field1 || field2
- Current date: CURRENT_DATE
- Null handling: COALESCE(field, default_value)
"""
        elif self.db_type == "mysql":
            return """
Database-specific syntax (MYSQL):
- Date format: STR_TO_DATE(field, '%Y%m%d')
- String concatenation: CONCAT(field1, field2)
- Current date: CURDATE()
- Null handling: IFNULL(field, default_value)
"""
        elif self.db_type == "sqlite":
            return """
Database-specific syntax (SQLITE):
- Date format: date(field) or strftime('%Y%m%d', field)
- String concatenation: field1 || field2
- Current date: date('now')
- Null handling: COALESCE(field, default_value)
"""
        return ""
    
    # ==================== Validation ====================
    
    @field_validator('primary_dsn', 'linked_dsn')
    @classmethod
    def validate_dsn_not_empty(cls, v: str) -> str:
        """Ensure DSN is not empty string"""
        if not v or not v.strip():
            raise ValueError("DSN cannot be empty")
        return v.strip()
    
    @field_validator('primary_password', 'linked_password')
    @classmethod
    def validate_password_not_empty(cls, v: str) -> str:
        """Ensure password is not empty string"""
        if not v or not v.strip():
            raise ValueError("Password cannot be empty")
        return v
    
    def bird_db_path(self) -> Path:
        """Get BIRD database path (legacy method)"""
        return self.bird_data_path
    
    def validate_bird_paths(self) -> None:
        """Validate BIRD dataset paths exist"""
        if not (self.bird_root_path / "dev_databases").exists():
            raise ValueError(
                f"Invalid bird_root_path: {self.bird_root_path}"
            )
        if not self.bird_data_path.exists():
            raise ValueError(
                f"Invalid bird_data_path: {self.bird_data_path}"
            )
    
    def __repr__(self) -> str:
        """Safe repr that doesn't expose secrets."""
        return (
            f"Settings(db_type={self.db_type}, "
            f"primary_db={self.primary_db_name}, "
            f"linked_db={self.linked_db_name}, "
            f"model={self.default_model})"
        )


# Global settings instance
settings = Settings()


# Convenience function
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings