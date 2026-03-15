"""Test logger handles both objects and dicts"""
import asyncio
from dataclasses import dataclass
from src.utils.logger import RouterLogger


@dataclass
class MockToolResponse:
    """Mock object (like ToolResponse)"""
    answer: str
    success: bool
    error: str = ""


async def test_logger_with_object():
    """Test logger with dataclass object"""
    logger = RouterLogger(log_dir="logs/test")
    
    result = MockToolResponse(
        answer="Test answer",
        success=True
    )
    
    await logger.log_result(
        query="Test query",
        path="test_path",
        result=result
    )
    
    print(" Logger handles objects")


async def test_logger_with_dict():
    """Test logger with dictionary"""
    logger = RouterLogger(log_dir="logs/test")
    
    result = {
        "answer": "Test answer",
        "success": True,
        "error": None
    }
    
    await logger.log_result(
        query="Test query",
        path="test_path",
        result=result
    )
    
    print(" Logger handles dicts")


async def test_logger_with_none():
    """Test logger with None/missing fields"""
    logger = RouterLogger(log_dir="logs/test")
    
    result = {"answer": "Only answer"}  # Missing 'success', 'error'
    
    await logger.log_result(
        query="Test query",
        path="test_path",
        result=result
    )
    
    print(" Logger handles missing fields")


async def test_all():
    await test_logger_with_object()
    await test_logger_with_dict()
    await test_logger_with_none()
    print("\n All logger tests passed!")


if __name__ == "__main__":
    asyncio.run(test_all())