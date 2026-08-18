import logging
import json
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncIterator

from .schemas import QueryRequest, QueryResponse, ErrorResponse, SSEEvent

logger = logging.getLogger(__name__)

router = APIRouter()


async def sse_generator(query_request: QueryRequest, request: Request) -> AsyncIterator[str]:
    """
    SSE event generator.
    
    Yields Server-Sent Events in the format:
    data: {"type": "...", "content": "..."}
    
    """
    # Access Fast Lane from app.state
    fast_lane = getattr(request.app.state, 'fast_lane', None)
    
    if not fast_lane:
        error_event = {
            "type": "error",
            "data": {
                "message": "Fast Lane not initialized",
                "type": "initialization_error"
            }
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return
    
    try:
        async for event in fast_lane.query(
            query=query_request.query,
            top_k=query_request.top_k,
            language=query_request.language,
            streaming=True
        ):
            # Format as SSE event
            yield f"data: {json.dumps(event)}\n\n"
    
    except asyncio.CancelledError:
        logger.info("SSE stream cancelled (client disconnected)")
        raise  # Re-raise to properly cleanup
    
    except Exception as e:
        logger.error(f"SSE streaming error: {e}", exc_info=True)
        error_event = {
            "type": "error",
            "data": {
                "message": str(e),
                "type": type(e).__name__
            }
        }
        yield f"data: {json.dumps(error_event)}\n\n"


@router.post("/query")
async def query(request_body: QueryRequest, request: Request):
    """
    Query endpoint with SSE streaming.
    
    Returns Server-Sent Events stream with:
    - status: Progress updates
    - citation: Source metadata
    - chunk: Answer chunks
    - done: Completion metadata
    - error: Error events
    """
    if not request_body.streaming:
        # Non-streaming mode: collect all events and return JSON
        return await query_non_streaming(request_body, request)
    
    # Streaming mode: return SSE stream
    return StreamingResponse(
        sse_generator(request_body, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


async def query_non_streaming(request_body: QueryRequest, request: Request) -> QueryResponse:
    """
    Non-streaming query (collects all events and returns JSON).
    
    For clients that don't support SSE.
    """
    # Access Fast Lane from app.state
    fast_lane = getattr(request.app.state, 'fast_lane', None)
    
    if not fast_lane:
        raise HTTPException(
            status_code=503,
            detail="Fast Lane not initialized"
        )
    
    # Collect all events
    answer_chunks = []
    citations = []
    metadata = {}
    error_occurred = False
    
    try:
        async for event in fast_lane.query(
            query=request_body.query,
            top_k=request_body.top_k,
            language=request_body.language,
            streaming=True
        ):
            event_type = event.get('type')
            
            if event_type == 'chunk':
                answer_chunks.append(event.get('content', ''))
            
            elif event_type == 'citation':
                citations.append(event.get('data', {}))
            
            elif event_type == 'done':
                metadata = event.get('metadata', {})
            
            elif event_type == 'error':
                # Error event received - raise immediately
                error_data = event.get('data', {})
                error_msg = error_data.get('message', 'Unknown error')
                logger.error(f"Non-streaming query error: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
            
            # Ignore status events in non-streaming mode
        
        # Validate we got a complete response
        if not answer_chunks:
            logger.warning("No answer chunks collected - response may be empty")
            # This can happen if only error/status events were sent
            # Return empty response rather than crash
        
        # Build response
        answer = ''.join(answer_chunks)
        return QueryResponse(
            answer=answer,
            citations=citations,
            metadata=metadata if metadata else {"warning": "No completion metadata received"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/test")
async def test_query(request: Request):
    """Test endpoint to verify Fast Lane is working"""
    # Access Fast Lane from app.state
    fast_lane = getattr(request.app.state, 'fast_lane', None)
    
    if not fast_lane:
        raise HTTPException(
            status_code=503,
            detail="Fast Lane not initialized"
        )
    
    # Simple test query
    test_request_body = QueryRequest(
        query="PVD 코팅이란?",
        top_k=3,
        language="ko",
        streaming=False
    )
    
    return await query_non_streaming(test_request_body, request)