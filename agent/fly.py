from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from agent.orchastrate import ContextState, StateOrchestrator
from agent.tools import Tools



class BaseResult(TypedDict):
    status: str
    state: str
    tools_used: List[str]


class CompleteResult(BaseResult):
    response: str
    messages: List[Dict]
    profiles: List[Dict]
    graph: List[Dict]
    web: List[Dict]


class ClarificationResult(BaseResult):
    question: str


RunResult = Union[CompleteResult, ClarificationResult]

@dataclass
class ToolCall:
    name: str
    args: Dict = field(default_factory=dict)


@dataclass 
class FinalResponse:
    content: str


@dataclass
class ClarificationRequest:
    question: str


StellaResponse = Union[ToolCall, FinalResponse, ClarificationRequest]

def build_context_for_stella(ctx: ContextState) -> Dict:
    return {
        "query": ctx.user_query,
        "hot_topic_context": ctx.hot_topic_context,
        "messages": ctx.retrieved_messages,
        "profiles": ctx.entity_profiles,
        "graph": ctx.graph_results,
        "web": ctx.web_results,
        "calls_remaining": ctx.max_calls - ctx.call_count,
        "current_state": ctx.current_state
    }


def call_the_doctor(
    ctx: ContextState, 
    last_result: Optional[Dict] = None, 
    error: Optional[str] = None
) -> StellaResponse:
    # TODO: Build prompt, call LLM with tools, parse response
    return FinalResponse(content="")


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

def run(user_query: str, hot_topics: List[str], active_topics: List[str]) -> RunResult:
    
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
        response = call_the_doctor(context, last_result, error)
        
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

