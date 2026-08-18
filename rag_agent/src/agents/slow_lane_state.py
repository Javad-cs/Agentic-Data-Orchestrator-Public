from typing import TypedDict, List, Dict, Any, Optional


class SlowLaneState(TypedDict):
    """
    State for Slow Lane LangGraph agent.
    
    Tracks the reasoning process through multiple steps.
    """
    # Input
    original_query: str
    language: str
    
    # Planning (STRUCTURED - changed from List[str])
    current_plan: List[Dict[str, Any]]  # List of step dicts with tool selection
    # Each step: {"step_id": int, "rationale": str, "tool": str, "query": str}
    
    # Memory
    scratchpad: List[str]  # Facts found: "Q: ... A: ..."
    context_bag: List[Dict[str, Any]]  # Accumulated context chunks
    
    # Loop control
    current_step_count: int
    max_iterations: int
    
    # Tool management
    available_tools: List[str]  # ["rag_tool", "sql_tool"]
    tool_descriptions: Dict[str, str]  # Tool name -> description
    table_context: str  # Formatted table summaries for SQL queries
    
    # Output
    final_answer: str
    error: str
    
    # Internal state
    _executor_result: Optional[Dict[str, Any]]
    _current_step: Optional[Dict[str, Any]]  # Changed from _current_subquery
    _validation: Optional[str]