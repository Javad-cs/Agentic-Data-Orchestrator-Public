from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class QueryRequest(BaseModel):
    """Request schema for query endpoint"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )
    language: str = Field(
        default="ko",
        pattern="^(ko|en)$",
        description="Response language (ko or en)"
    )
    streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "스테인레스강 고속 가공에 적합한 코팅은?",
                "top_k": 5,
                "language": "ko",
                "streaming": True
            }
        }


class CitationData(BaseModel):
    """Citation metadata"""
    id: str = Field(..., description="Citation ID like [1]")
    file: str = Field(..., description="Source file name")
    page: Optional[int] = Field(None, description="Page number")
    source_id: str = Field(..., description="Internal source ID")


class EventType(str, Enum):
    """SSE event types"""
    STATUS = "status"
    CITATION = "citation"
    CHUNK = "chunk"
    CITATION_MARKER = "citation_marker"
    DONE = "done"
    ERROR = "error"


class SSEEvent(BaseModel):
    """SSE event schema"""
    type: EventType
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Non-streaming query response"""
    answer: str = Field(..., description="Generated answer")
    citations: List[CitationData] = Field(..., description="Source citations")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "스테인레스강 고속 가공에는 PC8110 PVD 코팅이 적합합니다 [1].",
                "citations": [
                    {
                        "id": "[1]",
                        "file": "sample.pdf",
                        "page": 5,
                        "source_id": "chunk_123"
                    }
                ],
                "metadata": {
                    "latency_ms": 2500,
                    "citation_count": 1,
                    "safety_passed": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "No relevant documents found",
                "type": "no_results",
                "details": {"query": "..."}
            }
        }