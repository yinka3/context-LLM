from datetime import datetime, timezone
import json
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import uuid

from loguru import logger
import redis
from agent.orchestrate import ContextState, StateOrchestrator
from agent.tools import Tools
from redisclient import AsyncRedisClient
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
    
    if last_result:
        msg += "\n**Last tool result(s):**\n"
        
        results = last_result if isinstance(last_result, list) else [last_result]
        
        for r in results:
            tool = r.get("tool", "unknown")
            if "error" in r:
                msg += f"- `{tool}`: Error - {r['error']}\n"
            else:
                data = r.get("result", {}).get("data")
                if data is None or data == [] or data == {}:
                    msg += f"- `{tool}`: No results found\n"
                else:
                    msg += f"- `{tool}`: {json.dumps(data, indent=2, default=str)[:500]}\n"
    
    if ctx.hot_topic_context:
        msg += f"\n**Hot topic context (pre-fetched):**\n```json\n{json.dumps(ctx.hot_topic_context, indent=2, default=str)}\n```\n"
    
    if ctx.entity_profiles:
        msg += f"\n**Accumulated profiles ({len(ctx.entity_profiles)}):**\n```json\n{json.dumps(ctx.entity_profiles, indent=2, default=str)}\n```\n"
    
    if ctx.graph_results:
        msg += f"\n**Accumulated graph results ({len(ctx.graph_results)}):**\n```json\n{json.dumps(ctx.graph_results, indent=2, default=str)}\n```\n"
    
    if ctx.retrieved_messages:
        msg += f"\n**Accumulated messages ({len(ctx.retrieved_messages)}):**\n```json\n{json.dumps(ctx.retrieved_messages, indent=2, default=str)}\n```\n"
    
    
    # if ctx.web_results:
    #     msg += f"\n**Web results ({len(ctx.web_results)}):**\n```json\n{json.dumps(ctx.web_results, indent=2, default=str)}\n```\n"
    
    return msg


async def call_the_doctor(
    llm: LLMService,
    ctx: ContextState,
    user_name: str,
    last_result: Optional[Dict] = None,
    persona: str = ""
) -> StellaResponse:
    
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    system_prompt = get_stella_prompt(user_name, current_time, persona)
    user_message = build_user_message(ctx, ctx.history, last_result)
    
    response = await llm.call_with_tools(
        system=system_prompt,
        user=user_message,
        tools=TOOL_SCHEMAS
    )
    
    if not response or not response.get("tool_calls"):
        return FinalResponse(content="I couldn't determine how to help.")
    
    tool_calls = response["tool_calls"]
    if len(tool_calls) == 1:
        tc = tool_calls[0]
        name = tc["name"]
        args = json.loads(tc["arguments"])
        
        if name == "finish":
            return FinalResponse(content=args.get("response", ""))
        if name == "request_clarification":
            return ClarificationRequest(question=args.get("question", ""))
    
        return ToolCall(name=name, args=args)

    return [ToolCall(name=tc["name"], args=json.loads(tc["arguments"])) for tc in tool_calls] 
    

async def execute_tool(tools: Tools, name: str, args: Dict) -> Optional[Dict]:
    dispatch = {
        "search_messages": lambda: tools.search_messages(args.get("query", ""), args.get("limit", 5)),
        "search_entities": lambda: tools.search_entities(args.get("query", "")),
        "get_profile": lambda: tools.get_profile(args.get("entity_name", "")),
        "get_connections": lambda: tools.get_connections(args.get("entity_name", "")),
        "get_activity": lambda: tools.get_recent_activity(args.get("entity_name", ""), args.get("hours", 24)),
        "find_path": lambda: tools.find_path(args.get("entity_a", ""), args.get("entity_b", ""))
    }
    
    if name not in dispatch:
        return {"error": f"Unknown tool: {name}"}
    
    try:
        result = await dispatch[name]()
        return {"data": result}
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return {"error": str(e)}


def update_accumulators(ctx: ContextState, tool_name: str, result):
    if not result or "error" in result:
        return
    
    data = result.get("data")
    if not data:
        return
    
    if tool_name == "search_messages":
        ctx.retrieved_messages.extend(data if isinstance(data, list) else [])
    
    elif tool_name == "search_entities":
        ctx.entity_profiles.extend(data if isinstance(data, list) else [])
    
    elif tool_name == "get_profile":
        if data:
            ctx.entity_profiles.append(data)
    
    elif tool_name in ("get_connections", "get_activity", "find_path"):
        ctx.graph_results.extend(data if isinstance(data, list) else [])
    
    # elif tool_name == "web_search":
    #     ctx.web_results.extend(result if isinstance(data, list) else [])


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
        redis_client: redis.Redis
    ) -> RunResult:
    
    system_warning = ""
    try:
        raw_warning = await redis_client.get("system:active_job_warning")
        if raw_warning:
            system_warning = f"{raw_warning}\n\n---\n\n"
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
    tools = Tools(user_name, store, ent_resolver, redis_client, active_topics)

    if not ent_resolver.entity_profiles or len(ent_resolver.entity_profiles) <= 1:
        return CompleteResult(
            status="complete",
            response="I don't know much about your world yet. Tell me about the people, places, and things in your life and I'll start remembering.",
            tools_used=[],
            state="start",
            messages=[],
            profiles=[],
            graph=[]
            # web=[]
        )

    if hot_topics:
        context.hot_topic_context = tools.get_hot_topic_context(hot_topics)
    
    last_result = None
    terminal_states = {machine.complete, machine.clarify}
    while machine.current_state not in terminal_states:
        context.attempt_count += 1

        if context.attempt_count >= context.max_attempts:
            return ClarificationResult(
                status="clarification_needed",
                question="I'm having trouble processing this. Could you rephrase your question?",
                tools_used=context.tools_used,
                state=machine.current_state.id)
    
        if context.call_count >= context.max_calls:
            if machine.can_finish():
                partial_response = "Here's what I found, though I couldn't fully answer your question:\n"
                
                if context.entity_profiles:
                    names = [p.get("canonical_name", "unknown") for p in context.entity_profiles if isinstance(p, dict)]
                    if names:
                        partial_response += f"- Found profiles: {', '.join(names)}\n"
                
                if context.retrieved_messages:
                    partial_response += f"- Found {len(context.retrieved_messages)} related messages\n"
                
                return CompleteResult(
                    status="complete",
                    response=partial_response,
                    tools_used=context.tools_used,
                    state=machine.current_state.id,
                    messages=context.retrieved_messages,
                    profiles=context.entity_profiles,
                    graph=context.graph_results)
            
            return ClarificationResult(
                status="clarification_needed",
                question="I couldn't find relevant information. Could you rephrase or be more specific?",
                tools_used=context.tools_used,
                state=machine.current_state.id)
    
        
        context.current_state = machine.current_state.id
        context.current_step += 1
        step_start = time.perf_counter()
        response = await call_the_doctor(llm, context, user_name, last_result)
        
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
                last_result = [{"tool": "request_clarification", "error": reason}]
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
                    graph=context.graph_results
                )
            else:
                last_result = [{"tool": "finish", "error": reason}]
                continue
        

        tool_calls = [response] if isinstance(response, ToolCall) else response
        
        all_results = []
        for tc in tool_calls:
            tool_name = tc.name
            args = tc.args
            
            valid, reason = machine.validate(tool_name, args)
            
            if not valid:
                trace.entries.append(TraceEntry(
                    step=context.current_step,
                    state=context.current_state,
                    tool=tool_name,
                    args=args,
                    resolved_args={},
                    result_summary=f"Validation failed: {reason}",
                    result_count=0,
                    duration_ms=(time.perf_counter() - step_start) * 1000,
                    error=reason
                ))
                
                all_results.append({"tool": tool_name, "error": reason})
                context.consecutive_rejections += 1
                if context.consecutive_rejections >= 3:
                    break
                continue
            
            result = await execute_tool(tools, tool_name, args)
            result_summary, result_count = summarize_result(tool_name, result)
        
            trace.entries.append(TraceEntry(
                step=context.current_step,
                state=context.current_state,
                tool=tool_name,
                args=args,
                resolved_args=args,
                result_summary=result_summary,
                result_count=result_count,
                duration_ms=(time.perf_counter() - step_start) * 1000,
                error=result.get("error") if isinstance(result, dict) else None
            ))
            
            machine.record_call(tool_name, args)
            context.consecutive_rejections = 0
            
            if hasattr(machine, tool_name):
                getattr(machine, tool_name)()
            
            update_accumulators(context, tool_name, result)
            all_results.append({"tool": tool_name, "result": result})

        if context.consecutive_rejections >= 3 and not any("result" in r for r in all_results):
            if context.entity_profiles or context.retrieved_messages or context.graph_results:
                return CompleteResult(
                    status="complete",
                    response="I found some information but had trouble completing the search.",
                    tools_used=context.tools_used,
                    state=machine.current_state.id,
                    messages=context.retrieved_messages,
                    profiles=context.entity_profiles,
                    graph=context.graph_results
                )
            return ClarificationResult(
                status="clarification_needed",
                question="I'm having trouble with that search. Could you rephrase or be more specific?",
                tools_used=context.tools_used,
                state=machine.current_state.id
            )
        last_result = all_results
        machine.try_advance()
    
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
        graph=context.graph_results
    )

