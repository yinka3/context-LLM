from datetime import datetime, timezone
import json
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import uuid

from loguru import logger
from agent.orchestrate import ContextState, StateOrchestrator
from agent.tools import Tools
from redisclient import AsyncRedisClient, SyncRedisClient
from main.service import LLMService
from main.system_prompt import get_stella_prompt
from schema.dtypes import (
    ClarificationRequest, 
    ClarificationResult, 
    CompleteResult, 
    FinalResponse,
    QueryTrace, 
    RunResult, 
    StellaResponse, 
    ToolCall,
    TraceEntry
)
from schema.tool_schema import TOOL_SCHEMAS

if TYPE_CHECKING:
    from db.memgraph import MemGraphStore
    from main.entity_resolve import EntityResolver

def build_user_message(
    ctx: ContextState,
    conversation_history: List[Dict],
    last_result: Optional[Dict] = None,
    error: Optional[str] = None
) -> str:
    msg = ""
    
    if conversation_history:
        recent = conversation_history[-4:]
        msg += "**Recent conversation:**\n"
        for turn in recent:
            role = "User" if turn["role"] == "user" else "STELLA"
            msg += f"{role}: {turn['content']}\n"
        msg += "\n"
    
    msg += f"**Query:** {ctx.user_query}\n"
    msg += f"**State:** {ctx.current_state}\n"
    msg += f"**Calls remaining:** {ctx.max_calls - ctx.call_count}\n"
    
    if error:
        msg += f"\n**Error from last action:** {error}\n"
    
    if last_result:
        msg += f"\n**Last tool result:**\n```json\n{json.dumps(last_result, indent=2, default=str)}\n```\n"
    
    if ctx.hot_topic_context:
        msg += f"\n**Hot topic context (pre-fetched):**\n```json\n{json.dumps(ctx.hot_topic_context, indent=2, default=str)}\n```\n"
    
    if ctx.entity_profiles:
        msg += f"\n**Accumulated profiles ({len(ctx.entity_profiles)}):**\n```json\n{json.dumps(ctx.entity_profiles, indent=2, default=str)}\n```\n"
    
    if ctx.graph_results:
        msg += f"\n**Accumulated graph results ({len(ctx.graph_results)}):**\n```json\n{json.dumps(ctx.graph_results, indent=2, default=str)}\n```\n"
    
    if ctx.retrieved_messages:
        msg += f"\n**Accumulated messages ({len(ctx.retrieved_messages)}):**\n```json\n{json.dumps(ctx.retrieved_messages, indent=2, default=str)}\n```\n"
    
    if ctx.web_results:
        msg += f"\n**Web results ({len(ctx.web_results)}):**\n```json\n{json.dumps(ctx.web_results, indent=2, default=str)}\n```\n"
    
    return msg


def call_the_doctor(
    llm: LLMService,
    ctx: ContextState,
    user_name: str,
    last_result: Optional[Dict] = None,
    error: Optional[str] = None,
    persona: str = ""
) -> StellaResponse:
    
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    system_prompt = get_stella_prompt(user_name, current_time, persona)
    user_message = build_user_message(ctx, ctx.history, last_result, error)
    
    response = llm.call_with_tools_sync(
        system=system_prompt,
        user=user_message,
        tools=TOOL_SCHEMAS
    )
    
    if not response or not response.get("tool_calls"):
        return FinalResponse(content="I couldn't determine how to help.")
    
    tool_call = response["tool_calls"][0]
    name = tool_call["name"]
    args = json.loads(tool_call["arguments"])
    
    if name == "finish":
        return FinalResponse(content=args.get("response", ""))
    
    if name == "request_clarification":
        return ClarificationRequest(question=args.get("question", ""))
    
    return ToolCall(name=name, args=args)   
    

def execute_tool(tools: Tools, name: str, args: Dict) -> Optional[Dict]:
    dispatch = {
        "search_messages": lambda: tools.search_messages(args.get("query", "")),
        "search_entities": lambda: tools.search_entities(args.get("query", "")),
        "get_profile": lambda: tools.get_profile(args.get("entity_name", "")),
        "get_connections": lambda: tools.get_connections(args.get("entity_name", "")),
        "get_activity": lambda: tools.get_recent_activity(args.get("entity_name", ""), args.get("hours", 24)),
        "find_path": lambda: tools.find_path(args.get("entity_a", ""), args.get("entity_b", "")),
        "web_search": lambda: tools.web_search(args.get("query", ""))
    }
    
    if name not in dispatch:
        return {"error": f"Unknown tool: {name}"}
    
    try:
        result = dispatch[name]()
        return {"data": result}
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return {"error": str(e)}


def update_accumulators(ctx: ContextState, tool_name: str, result):
    if not result:
        return
    
    if tool_name == "search_messages":
        ctx.retrieved_messages.extend(result if isinstance(result, list) else [])
    
    elif tool_name == "search_entities":
        ctx.entity_profiles.extend(result if isinstance(result, list) else [])
    
    elif tool_name == "get_profile":
        ctx.entity_profiles.append(result)
        if result.get("id"):
            ctx.inspected_entity_ids.add(result["id"])
    
    elif tool_name in ("get_connections", "get_activity", "find_path"):
        ctx.graph_results.extend(result if isinstance(result, list) else [])
    
    elif tool_name == "web_search":
        ctx.web_results.extend(result if isinstance(result, list) else [])


def summarize_result(tool_name: str, result: Dict) -> Tuple[str, int]:
    """Summarize tool result for trace."""
    if "error" in result:
        return f"Error: {result['error']}", 0
    
    data = result.get("data")
    if data is None:
        return "No results", 0
    
    if tool_name == "get_profile":
        if data:
            return f"Found: {data.get('name', 'unknown')} ({data.get('type', 'unknown')})", 1
        return "Not found", 0
    
    if tool_name in ("get_connections", "get_activity", "search_messages", "search_entities"):
        count = len(data) if isinstance(data, list) else 0
        return f"Found {count} results", count
    
    if tool_name == "find_path":
        if data:
            return f"Path found: {len(data)} hops", len(data)
        return "No path", 0
    
    return "Completed", 1



async def run(user_query: str,
        user_name: str,
        conversation_history: List[Dict],
        hot_topics: List[str], 
        active_topics: List[str],
        llm: LLMService,
        store: 'MemGraphStore',
        ent_resolver: 'EntityResolver',
        redis_client: AsyncRedisClient
        ) -> RunResult:
    
    system_warning = ""
    try:
        raw_warning = await redis_client.get_client().get("system:active_job_warning")
        if raw_warning:
            system_warning = f"{raw_warning.decode()}\n\n---\n\n"
    except Exception as e:
        logger.error(f"Failed to check system warning: {e}")
    
    trace = QueryTrace(
        trace_id=str(uuid.uuid4()),
        user_query=user_query,
        started_at=datetime.now(timezone.utc)
    )
    
    context = ContextState(
        user_query=user_query,
        hot_topics=hot_topics,
        active_topics=active_topics,
        trace_id=trace.trace_id,
        history=conversation_history
    )
    machine = StateOrchestrator(context)
    sync_redis = SyncRedisClient().get_client()
    tools = Tools(user_name, store, ent_resolver, sync_redis)

    if hot_topics:
        context.hot_topic_context = tools.get_hot_topic_context(hot_topics)
    
    last_result = None
    error = None
    terminal_states = {machine.complete, machine.clarify}
    while machine.current_state not in terminal_states:
        
        context.current_state = machine.current_state.id
        context.current_step += 1
        if context.call_count >= context.max_calls:
            if machine.current_state not in terminal_states:
                return ClarificationResult(
                    status="clarification_needed",
                    question="I wasn't able to find enough information to answer that. Could you be more specific or rephrase?",
                    tools_used=context.tools_used,
                    state=machine.current_state.id
                )
        step_start = time.perf_counter()
        response = call_the_doctor(llm, context, user_name, last_result, error)
        
        if isinstance(response, ClarificationRequest):
            valid, reason = machine.validate("request_clarification", {})
            if valid:
                machine.request_clarification()
                final_q = response.question
                if system_warning:
                    final_q = system_warning + final_q

                return ClarificationResult(
                    status="clarification_needed",
                    question=final_q,
                    tools_used=context.tools_used,
                    state=machine.current_state.id
                )
            else:
                error = reason
                last_result = None
                continue
        
        if isinstance(response, FinalResponse):
            valid, reason = machine.validate("finish", {})
            if valid:
                machine.record_call("finish", {})
                machine.finish()
                final_response_text = response.content
                if system_warning:
                    final_response_text = system_warning + final_response_text

                return CompleteResult(
                    status="complete",
                    response=final_response_text,
                    tools_used=context.tools_used,
                    state=machine.current_state.id,
                    messages=context.retrieved_messages,
                    profiles=context.entity_profiles,
                    graph=context.graph_results,
                    web=context.web_results
                )
            else:
                error = reason
                last_result = None
                continue
        
        tool_name = response.name
        args = response.args
        
        valid, reason = machine.validate(tool_name, args)
        
        if not valid:
            trace.entries.append(TraceEntry(
                step=context.current_step,
                state=context.current_state,
                tool=tool_name,
                args=args,
                resolved_args={},
                result_summary="",
                result_count=0,
                duration_ms=(time.perf_counter() - step_start) * 1000,
                error=reason
            ))
            error = reason
            last_result = None
            continue
        

        last_result = execute_tool(tools, tool_name, args)
        result_summary, result_count = summarize_result(tool_name, last_result)
        
        trace.entries.append(TraceEntry(
            step=context.current_step,
            state=context.current_state,
            tool=tool_name,
            args=args,
            resolved_args=args,
            result_summary=result_summary,
            result_count=result_count,
            duration_ms=(time.perf_counter() - step_start) * 1000,
            error=last_result.get("error") if isinstance(last_result, dict) else None
        ))


        machine.record_call(tool_name, args)
        getattr(machine, tool_name)()

        update_accumulators(context, tool_name, last_result)
        machine.try_advance()
        error = None
    
    logger.info(f"[STELLA] Trace {trace.trace_id} completed: {len(trace.entries)} steps")
    for entry in trace.entries:
        logger.debug(f"[STELLA] Step {entry.step}: {entry.tool} -> {entry.result_summary} ({entry.duration_ms:.0f}ms)")

    return CompleteResult(
        status="complete",
        response=system_warning + "I encountered a state error and could not finish.",
        tools_used=context.tools_used,
        state=machine.current_state.id,
        messages=context.retrieved_messages,
        profiles=context.entity_profiles,
        graph=context.graph_results,
        web=context.web_results
    )

