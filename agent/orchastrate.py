from dataclasses import dataclass, field
from statemachine import StateMachine, State, Event
from typing import Optional, Set, List, Dict
from agent.tools import Tools


@dataclass
class ContextState:
    call_count: int = 0
    max_calls: int = 5
    user_query: str = ""
    target_entity: str = ""
    operation: str = ""
    second_entity: Optional[str] = None
    inspected_entity_ids: Set[int] = field(default_factory=set)
    hot_topics: List[str] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)
    hot_topic_context: Dict[str, List[Dict]] = field(default_factory=dict)
    graph_returned_empty: bool = False
    needs_clarification: bool = False
    time_window_hours: int = 24
    evidence: List[Dict] = field(default_factory=list)
    retrieved_messages: List[Dict] = field(default_factory=list)
    entity_profiles: List[Dict] = field(default_factory=list)
    graph_results: List[Dict] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

class StateOrchestrator(StateMachine):

    ready = State("ready", initial=True)

    search_messages = State("search_message")
    inspect_profile = State("inspect_profile")
    query_graph = State("query_graph")
    web_search = State("web_search")

    clarify = State("clarify")
    summarize = State("summarize", final=True)
    failed = State("failed", final=True)

    start_message_search = ready.to(search_messages, cond="can_continue")
    start_profile_inspect = ready.to(inspect_profile, cond="can_continue")
    start_graph_query = ready.to(query_graph, cond="can_continue")
    start_web_search = ready.to(web_search)

    search_to_profile = search_messages.to(inspect_profile, cond="can_continue")
    search_to_summarize = search_messages.to(summarize, cond="can_continue")
    search_to_clarify = search_messages.to(clarify, cond="can_continue")

    inspect_another = inspect_profile.to.itself(cond="can_continue")
    profile_to_graph = inspect_profile.to(query_graph, cond="can_continue")
    profile_to_summarize = inspect_profile.to(summarize, cond="can_continue")

    query_another = query_graph.to.itself(cond="can_continue")
    graph_to_profile = query_graph.to(inspect_profile, cond="can_continue")
    graph_to_summarize = query_graph.to(summarize, cond="can_continue")
    profile_to_clarify = inspect_profile.to(clarify)
    graph_to_clarify = query_graph.to(clarify)
    web_to_summarize = web_search.to(summarize)

    clarify_restart = clarify.to(ready)

    give_up = Event(
        search_messages.to(failed, cond="should_abort") |
        inspect_profile.to(failed, cond="should_abort") |
        query_graph.to(failed, cond="should_abort")
    )

    complete = Event(
        search_messages.to(summarize) |
        inspect_profile.to(summarize) |
        query_graph.to(summarize)
    )


    def __init__(self, state: ContextState):
        self.ctx_state = state
        self.tools = Tools() 
        super().__init__()
    
    def can_continue(self):
        return self.ctx_state.call_count < self.ctx_state.max_calls
    
    def should_abort(self):
        return self.ctx_state.call_count == self.ctx_state.max_calls

    def after_transition(self, event, state):
        self.ctx_state.call_count += 1
        self.ctx_state.tools_used.append(event)

    async def on_enter_search_messages(self):
        results = await self.tools.search_messages(self.ctx_state.user_query)
        self.ctx_state.retrieved_messages.extend(results)
    
    async def on_enter_inspect_profile(self):
        op = self.ctx_state.operation
        
        if op == "search":
            results = await self.tools.search_entities(self.ctx_state.user_query)
            self.ctx_state.entity_profiles.extend(results)
        else:  # "get" is default
            result = await self.tools.get_profile(self.ctx_state.target_entity)
            if result:
                self.ctx_state.entity_profiles.append(result)
                self.ctx_state.inspected_entity_ids.add(result.get("id"))

    async def on_enter_query_graph(self):
        op = self.ctx_state.operation
        target = self.ctx_state.target_entity
        
        if op == "path":
            results = await self.tools.find_path(target, self.ctx_state.second_entity)
        elif op == "activity":
            results = await self.tools.get_recent_activity(target, self.ctx_state.time_window_hours)
        else:  # "connections" is default
            results = await self.tools.get_connections(target)
        
        if results:
            self.ctx_state.graph_results.extend(results)
        else:
            self.ctx_state.graph_returned_empty = True
    
    async def on_enter_summarize(self):
        # LLM synthesis happens here — stub for now
        pass

    async def on_enter_failed(self):
        self.ctx_state.evidence.append({
            "type": "system",
            "message": "I couldn't find information to answer that."
        })

    async def on_enter_clarify(self):
        self.ctx_state.call_count = 0 