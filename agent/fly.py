from datetime import datetime, timezone
import json
from typing import Dict, List, Optional

from agent.orchestrate import ContextState, StateOrchestrator
from agent.tools import Tools
from main.service import LLMService
from main.system_prompt import get_stella_prompt
from schema.dtypes import (
    ClarificationRequest, 
    ClarificationResult, 
    CompleteResult, 
    FinalResponse, 
    RunResult, 
    StellaResponse, 
    ToolCall
)
from schema.tool_schema import TOOL_SCHEMAS

def build_user_message(
    ctx: ContextState,
    last_result: Optional[Dict] = None,
    error: Optional[str] = None
) -> str:
    msg = f"**Query:** {ctx.user_query}\n"
    msg += f"**State:** {ctx.current_state}\n"
    msg += f"**Calls remaining:** {ctx.max_calls - ctx.call_count}\n"
    
    if error:
        msg += f"\n**Error from last action:** {error}\n"
    
    if last_result:
        msg += f"\n**Last tool result:**\n```json\n{json.dumps(last_result, indent=2, default=str)[:2000]}\n```\n"
    
    if ctx.hot_topic_context:
        msg += f"\n**Hot topic context (pre-fetched):**\n```json\n{json.dumps(ctx.hot_topic_context, indent=2, default=str)[:1500]}\n```\n"
    
    if ctx.entity_profiles:
        msg += f"\n**Accumulated profiles ({len(ctx.entity_profiles)}):**\n```json\n{json.dumps(ctx.entity_profiles, indent=2, default=str)[:2000]}\n```\n"
    
    if ctx.graph_results:
        msg += f"\n**Accumulated graph results ({len(ctx.graph_results)}):**\n```json\n{json.dumps(ctx.graph_results, indent=2, default=str)[:1500]}\n```\n"
    
    if ctx.retrieved_messages:
        msg += f"\n**Accumulated messages ({len(ctx.retrieved_messages)}):**\n```json\n{json.dumps(ctx.retrieved_messages, indent=2, default=str)[:1500]}\n```\n"
    
    if ctx.web_results:
        msg += f"\n**Web results ({len(ctx.web_results)}):**\n```json\n{json.dumps(ctx.web_results, indent=2, default=str)[:1500]}\n```\n"
    
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
    user_message = build_user_message(ctx, last_result, error)
    
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
    
    if name in dispatch:
        return dispatch[name]()
    return None


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

def run(user_query: str,
        user_name: str,
        hot_topics: List[str], 
        active_topics: List[str],
        llm: Optional[LLMService] = None) -> RunResult:
    
    if llm is None:
        llm = LLMService()
    
    context = ContextState(
        user_query=user_query,
        hot_topics=hot_topics,
        active_topics=active_topics
    )
    machine = StateOrchestrator(context)
    tools = Tools()

    if hot_topics:
        context.hot_topic_context = tools.get_hot_topic_context(hot_topics)
    
    last_result = None
    error = None
    terminal_states = {machine.complete, machine.clarify}
    while machine.current_state not in terminal_states:
        
        context.current_state = machine.current_state.id
        response = call_the_doctor(llm, context, user_name, last_result, error)
        
        if isinstance(response, ClarificationRequest):
            valid, reason = machine.validate("request_clarification", {})
            if valid:
                machine.request_clarification()
                return ClarificationResult(
                    status="clarification_needed",
                    question=response.question,
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
                return CompleteResult(
                    status="complete",
                    response=response.content,
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
            error = reason
            last_result = None
            continue
        

        last_result = execute_tool(tools, tool_name, args)
        machine.record_call(tool_name, args)
        getattr(machine, tool_name)()

        update_accumulators(context, tool_name, last_result)
        machine.try_advance()
        error = None
    
    return CompleteResult(
        status="complete",
        response="",
        tools_used=context.tools_used,
        state=machine.current_state.id,
        messages=context.retrieved_messages,
        profiles=context.entity_profiles,
        graph=context.graph_results,
        web=context.web_results
    )

