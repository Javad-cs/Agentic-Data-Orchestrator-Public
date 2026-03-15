import logging
import json
import hashlib
from typing import Dict, Any, AsyncIterator, Optional
from langgraph.graph import StateGraph, END

from src.config.models import SystemConfig
from src.agents.slow_lane_state import SlowLaneState
from src.generation import create_llm_client

logger = logging.getLogger(__name__)


class SlowLane:
    """
    Slow Lane agent for complex multi-hop queries.
    
    NOW SUPPORTS MULTIPLE TOOLS with intelligent routing:
    - RAG tool for document search
    - SQL tool for database queries
    - Planner decides which tool for each sub-query
    
    Architecture:
    1. Planner: Decompose query + assign tools (JSON plan)
    2. Executor: Call assigned tool
    3. Critic: Validate evidence (self-correction)
    4. Rewriter: Refine sub-query if evidence invalid
    5. Synthesizer: Generate final answer with citations
    """
    
    def __init__(
        self,
        config: SystemConfig,
        tools: Optional[Any] = None  # LangGraphToolWrapper instance
    ):
        """
        Initialize Slow Lane.
        
        Args:
            config: System configuration
            tools: LangGraphToolWrapper with RAG/SQL tools (REQUIRED)
        """
        if not tools:
            raise ValueError("Slow Lane requires tools wrapper (LangGraphToolWrapper)")
        
        self.config = config
        self.tools = tools
        self.llm_client = create_llm_client(config.llm)
        
        # Build LangGraph
        self.graph = self._build_graph()
        
        available = self.tools.get_available_tools()
        logger.info(f"SlowLane initialized with tools: {available}")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        workflow = StateGraph(SlowLaneState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("rewriter", self._rewriter_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "critic")
        
        # Critic decides: pass → executor/synthesizer, fail → rewriter
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "continue": "executor",  # More sub-queries remain
                "synthesize": "synthesizer",  # All done
                "rewrite": "rewriter"  # Evidence invalid
            }
        )
        
        # Rewriter goes back to executor
        workflow.add_edge("rewriter", "executor")
        
        # Synthesizer ends
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    async def query(
        self,
        query: str,
        language: str = "ko",
        streaming: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute Slow Lane query.
        
        Args:
            query: Complex user query
            language: Query language
            streaming: Stream intermediate steps
            
        Yields:
            SSE events with status updates and final answer
        """
        # Get available tools and descriptions
        available_tools = self.tools.get_available_tools()
        tool_descriptions = self.tools.get_tool_descriptions()
        
        # Get table context for SQL queries (if SQL tool available)
        table_context = ""
        if "sql_tool" in available_tools:
            try:
                table_context = await self.tools.get_table_context(query, max_tables=5)
            except Exception as e:
                logger.warning(f"Failed to get table context: {e}")
        
        # Initialize state
        initial_state: SlowLaneState = {
            "original_query": query,
            "language": language,
            "current_plan": [],
            "scratchpad": [],
            "context_bag": [],
            "current_step_count": 0,
            "max_iterations": self.config.slow_lane.max_iterations,
            "available_tools": available_tools,
            "tool_descriptions": tool_descriptions,
            "table_context": table_context,
            "final_answer": "",
            "error": "",
            "_executor_result": None,
            "_current_step": None,
            "_validation": None
        }
        
        if streaming:
            yield {"type": "status", "content": "Planning multi-step approach..."}
        
        # Run the graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            # Stream final answer
            if streaming and final_state["final_answer"]:
                answer = final_state["final_answer"]
                for i in range(0, len(answer), 10):
                    chunk = answer[i:i+10]
                    yield {"type": "chunk", "content": chunk}
                
                yield {
                    "type": "done",
                    "metadata": {
                        "steps_taken": final_state["current_step_count"],
                        "facts_found": len(final_state["scratchpad"]),
                        "sources_used": len(final_state["context_bag"])
                    }
                }
            elif final_state["error"]:
                yield {
                    "type": "error",
                    "data": {"message": final_state["error"]}
                }
        
        except Exception as e:
            logger.error(f"Slow Lane error: {e}", exc_info=True)
            yield {
                "type": "error",
                "data": {"message": str(e)}
            }
    
    async def _planner_node(self, state: SlowLaneState) -> Dict[str, Any]:
        """
        Decompose complex query into sub-queries with tool assignments.
        
        Generates structured JSON plan with tool routing decisions.
        """
        logger.info("Planner: Decomposing query with tool assignment")
        
        prompt = self._build_planner_prompt(
            state["original_query"],
            state["available_tools"],
            state["tool_descriptions"],
            state["table_context"],
            state["language"]
        )
        
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=800
        )
        
        # Parse JSON plan with retry
        plan = await self._parse_json_plan_with_retry(
            response.content,
            state["original_query"],
            state["language"]
        )
        
        # Safety: Detect empty plan early
        if not plan:
            logger.error("Planner generated empty plan!")
            return {
                "current_plan": [],
                "error": "Failed to generate valid execution plan"
            }
        
        logger.info(f"Planner: Generated {len(plan)} steps")
        for step in plan:
            logger.info(f"  Step {step['step_id']}: {step['tool']} - {step['query'][:50]}...")
        
        return {"current_plan": plan}
    
    async def _executor_node(self, state: SlowLaneState) -> Dict[str, Any]:
        """Execute next step using assigned tool."""
        
        # Safety: Stop if error already set
        if state.get("error"):
            logger.warning("Executor: Skipping due to existing error")
            return {}
        
        if not state["current_plan"]:
            logger.info("Executor: No more steps")
            return {}
        
        # Pop next step
        current_plan = state["current_plan"].copy()
        step = current_plan.pop(0)
        
        tool = step["tool"]
        query = step["query"]
        
        logger.info(f"Executor: Step {step['step_id']} - {tool.upper()}: {query[:50]}...")
        
        try:
            # Call appropriate tool
            if tool == "sql_tool":
                result = await self.tools.sql_query(query)
            elif tool == "rag_tool":
                result = await self.tools.rag_search(
                    query=query,
                    top_k=5,
                    language=state["language"]
                )
            else:
                result = {"success": False, "error": f"Unknown tool: {tool}"}
        
        except Exception as e:
            logger.error(f"Executor exception: {e}", exc_info=True)
            result = {"success": False, "error": str(e)}
        
        return {
            "current_plan": current_plan,
            "current_step_count": state["current_step_count"] + 1,
            "_executor_result": result,
            "_current_step": step
        }
    
    async def _critic_node(self, state: SlowLaneState) -> Dict[str, Any]:
        """Validate evidence from executor."""
        result = state.get("_executor_result", {})
        step = state.get("_current_step", {})
        sub_query = step.get("query", "")
        
        # Safety: Check success flag
        if not result.get("success"):
            logger.warning(f"Critic: Executor failed - {result.get('error')}")
            return {"_validation": "rewrite"}
        
        # Safety: Use .get() to avoid KeyError
        answer = result.get("answer", "")
        
        if not answer or len(answer) < 20:
            logger.warning("Critic: Answer too short or missing")
            return {"_validation": "rewrite"}
        
        no_answer_phrases = ["모르겠습니다", "알 수 없습니다", "정보가 없습니다",
                            "cannot", "don't know", "no information"]
        
        if any(phrase in answer.lower() for phrase in no_answer_phrases):
            logger.warning("Critic: Answer contains 'don't know'")
            return {"_validation": "rewrite"}
        
        logger.info("Critic: Evidence validated")
        
        # Update scratchpad
        scratchpad = state["scratchpad"].copy()
        scratchpad.append(f"Q: {sub_query}\nA: {answer}")
        
        # Update context bag with dedup (hash dedup across iterations)
        context_bag = state["context_bag"].copy()
        
        # Build existing IDs and hashes from current context_bag
        existing_ids = set()
        existing_hashes = set()
        
        for existing_chunk in context_bag:
            chunk_id = existing_chunk.get("child_id")
            if chunk_id:
                existing_ids.add(chunk_id)
            else:
                # Chunk without ID: compute its hash for dedup
                chunk_text = existing_chunk.get("parent_text") or existing_chunk.get("child_text") or ""
                if chunk_text:
                    chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                    existing_hashes.add(chunk_hash)
        
        # Now dedup new chunks against existing sets
        for chunk in result.get("context_chunks", []):
            chunk_id = chunk.get("child_id")
            
            if chunk_id:
                # Has ID: use it for dedup
                if chunk_id not in existing_ids:
                    context_bag.append(chunk)
                    existing_ids.add(chunk_id)
            else:
                # No ID: generate hash for dedup
                chunk_text = chunk.get("parent_text") or chunk.get("child_text") or ""
                if chunk_text:
                    chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                    
                    if chunk_hash not in existing_hashes:
                        context_bag.append(chunk)
                        existing_hashes.add(chunk_hash)
        
        return {
            "scratchpad": scratchpad,
            "context_bag": context_bag,
            "_validation": "pass"
        }
    
    async def _rewriter_node(self, state: SlowLaneState) -> Dict[str, Any]:
        """Rewrite sub-query if evidence was invalid."""
        step = state.get("_current_step", {})
        sub_query = step.get("query", "")
        tool = step.get("tool", "rag_tool")
        
        logger.info(f"Rewriter: Refining sub-query: {sub_query}")
        
        prompt = self._build_rewriter_prompt(sub_query, state["language"])
        
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=200
        )
        
        refined_query = response.content.strip()
        
        # Create new step with refined query (keep same tool)
        refined_step = {
            "step_id": step.get("step_id", 0),
            "rationale": f"Refined: {step.get('rationale', '')}",
            "tool": tool,
            "query": refined_query
        }
        
        # Add refined step back to plan (at front)
        current_plan = [refined_step] + state["current_plan"]
        
        logger.info(f"Rewriter: Refined to: {refined_query}")
        
        return {"current_plan": current_plan}
    
    async def _synthesizer_node(self, state: SlowLaneState) -> Dict[str, Any]:
        """Generate final answer from accumulated facts."""
        
        # Check if we have any evidence
        if not state["scratchpad"]:
            logger.warning("Synthesizer: No evidence collected")
            
            # If error exists, return it
            if state.get("error"):
                return {
                    "final_answer": "",
                    "error": state["error"]
                }
            
            return {
                "final_answer": "",
                "error": "Insufficient evidence to answer query"
            }
        
        logger.info("Synthesizer: Generating final answer")
        
        prompt = self._build_synthesizer_prompt(
            state["original_query"],
            state["scratchpad"],
            state["context_bag"],
            state["language"]
        )
        
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        final_answer = response.content
        
        if not final_answer or len(final_answer) < 10:
            logger.warning("Synthesizer: Generated answer too short")
            return {
                "final_answer": "",
                "error": "Failed to generate adequate answer"
            }
        
        logger.info(f"Synthesizer: Generated answer ({len(final_answer)} chars)")
        
        return {"final_answer": final_answer}
    
    def _should_continue(self, state: SlowLaneState) -> str:
        """Decide next step after critic."""
        
        # Safety: If error set, stop immediately
        if state.get("error"):
            logger.error(f"Stopping due to error: {state['error']}")
            return "synthesize"
        
        if state["current_step_count"] >= state["max_iterations"]:
            logger.warning("Max iterations reached")
            return "synthesize"
        
        validation = state.get("_validation", "pass")
        
        if validation == "rewrite":
            return "rewrite"
        
        if state["current_plan"]:
            return "continue"
        
        return "synthesize"
    
    async def _parse_json_plan_with_retry(
        self,
        response: str,
        original_query: str,
        language: str,
        retry_count: int = 0
    ) -> list:
        """
        Parse JSON plan with retry on failure.
        
        Handle parse failures gracefully with retry.
        """
        plan = self._parse_json_plan(response)
        
        if plan:
            return plan
        
        # Retry with repair prompt (max 2 retries)
        if retry_count < 2:
            logger.warning(f"JSON parse failed (attempt {retry_count + 1}), retrying with repair prompt...")
            
            repair_prompt = self._build_repair_prompt(response, original_query, language)
            
            repair_response = await self.llm_client.generate(
                prompt=repair_prompt,
                temperature=0.1,
                max_tokens=800
            )
            
            # Recursive retry
            return await self._parse_json_plan_with_retry(
                repair_response.content,
                original_query,
                language,
                retry_count + 1
            )
        else:
            logger.error("JSON parsing failed after retries")
            return []
    
    def _parse_json_plan(self, response: str) -> list:
        """
        Parse JSON plan from LLM response.
        
        Uses brace-counting extraction for robustness.
        """
        try:
            # Extract JSON using brace matching
            json_str = self._extract_json(response)
            
            # Parse JSON
            data = json.loads(json_str)
            
            if "plan" in data and isinstance(data["plan"], list):
                return data["plan"]
            else:
                logger.warning("Invalid plan structure (missing 'plan' key or not a list)")
                return []
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.debug(f"Response preview: {response[:300]}...")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing plan: {e}")
            return []
    
    def _extract_json(self, response: str) -> str:
        """
        Extract first valid JSON object using brace matching.
        
        Most robust approach - handles nested braces,
        multiple objects, code fences, etc.
        """
        # Find first '{'
        start = response.find('{')
        if start == -1:
            logger.warning("No opening brace found in response")
            return response.strip()
        
        # Count braces to find matching '}'
        brace_count = 0
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    return response[start:i + 1]
        
        # No matching brace found
        logger.warning("No matching closing brace found")
        return response.strip()
    
    def _build_planner_prompt(
        self,
        query: str,
        available_tools: list,
        tool_descriptions: Dict[str, str],
        table_context: str,
        language: str
    ) -> str:
        """Build prompt for planner node with tool assignment."""
        
        # Format tool descriptions
        tools_text = "\n".join([
            f"- {tool}: {desc}" 
            for tool, desc in tool_descriptions.items()
        ])
        
        # Include table context if SQL available
        table_section = ""
        if table_context and "sql_tool" in available_tools:
            table_section = f"\n\n{table_context}"
        
        if language == "ko":
            return f"""복잡한 질문을 여러 단계로 분해하고 각 단계에 적절한 도구를 할당하세요.

질문: {query}

사용 가능한 도구:
{tools_text}
{table_section}

JSON 형식으로 계획을 작성하세요 (코드 펜스 없이 JSON만):
{{
  "plan": [
    {{
      "step_id": 1,
      "rationale": "이 단계가 필요한 이유",
      "tool": "rag_tool" 또는 "sql_tool",
      "query": "도구에 전달할 하위 질문"
    }}
  ]
}}

규칙:
1. 숫자/계산/집계가 필요하면 sql_tool 사용
2. 정의/설명/정책이 필요하면 rag_tool 사용
3. 각 단계는 독립적으로 실행 가능해야 함
4. 반드시 유효한 JSON 형식으로 응답 (설명 없이 JSON만)

계획:"""
        else:
            return f"""Decompose this complex question into steps and assign the right tool for each step.

Question: {query}

Available Tools:
{tools_text}
{table_section}

Generate a plan in JSON format (JSON only, no code fences):
{{
  "plan": [
    {{
      "step_id": 1,
      "rationale": "Why this step is needed",
      "tool": "rag_tool" or "sql_tool",
      "query": "Sub-question to ask the tool"
    }}
  ]
}}

Rules:
1. Use sql_tool for: numbers, counts, calculations, aggregations
2. Use rag_tool for: definitions, explanations, policies, context
3. Each step should be independently executable
4. Respond with valid JSON only (no explanation, no code fences)

Plan:"""
    
    def _build_repair_prompt(self, failed_response: str, original_query: str, language: str) -> str:
        """Build prompt to repair malformed JSON."""
        if language == "ko":
            return f"""이전 응답이 유효한 JSON이 아닙니다. 올바른 JSON 형식으로 다시 작성하세요.

원래 질문: {original_query}

실패한 응답:
{failed_response[:500]}

다음 형식으로 정확히 작성하세요 (설명 없이 JSON만):
{{
  "plan": [
    {{"step_id": 1, "rationale": "...", "tool": "rag_tool" or "sql_tool", "query": "..."}}
  ]
}}

수정된 JSON:"""
        else:
            return f"""The previous response was not valid JSON. Please rewrite it in correct JSON format.

Original question: {original_query}

Failed response:
{failed_response[:500]}

Write in this exact format (JSON only, no explanation):
{{
  "plan": [
    {{"step_id": 1, "rationale": "...", "tool": "rag_tool" or "sql_tool", "query": "..."}}
  ]
}}

Corrected JSON:"""
    
    def _build_rewriter_prompt(self, sub_query: str, language: str) -> str:
        """Build prompt for rewriter node"""
        if language == "ko":
            return f"""다음 질문에 대한 답변을 찾지 못했습니다. 질문을 더 구체적으로 다시 작성하세요.

원래 질문: {sub_query}

개선된 질문:"""
        else:
            return f"""We couldn't find an answer to this question. Rewrite it to be more specific.

Original: {sub_query}

Refined:"""
    
    def _build_synthesizer_prompt(
        self,
        original_query: str,
        scratchpad: list,
        context_bag: list,
        language: str
    ) -> str:
        """Build prompt for synthesizer node"""
        facts_text = "\n\n".join(scratchpad)
        
        context_text = ""
        for i, chunk in enumerate(context_bag, 1):
            parent_text = chunk.get("parent_text", chunk.get("child_text", ""))
            context_text += f"[{i}] {parent_text}\n\n"
        
        if language == "ko":
            return f"""다음 정보를 바탕으로 원래 질문에 대한 완전한 답변을 작성하세요.

원래 질문: {original_query}

수집한 정보:
{facts_text}

컨텍스트 (출처):
{context_text}

최종 답변 (반드시 [1], [2] 등으로 출처를 표시하세요):"""
        else:
            return f"""Based on the following information, write a complete answer to the original question.

Original Question: {original_query}

Gathered Information:
{facts_text}

Context (Sources):
{context_text}

Final Answer (cite sources using [1], [2], etc.):"""
    
    async def close(self):
        """Cleanup resources"""
        if self.llm_client:
            await self.llm_client.close()
        logger.info("SlowLane closed")


async def create_slow_lane(config: SystemConfig, tools) -> SlowLane:
    """
    Create Slow Lane agent.
    
    Args:
        config: System configuration
        tools: LangGraphToolWrapper instance (REQUIRED)
        
    Returns:
        SlowLane instance
    """
    return SlowLane(config=config, tools=tools)