from dataclasses import dataclass, field
from statemachine import StateMachine, State
from typing import Set, List, Dict, Tuple


@dataclass
class ContextState:
    call_count: int = 0
    max_calls: int = 5
    user_query: str = ""
    target_entity: str = ""
    inspected_entity_ids: Set[int] = field(default_factory=set)
    hot_topics: List[str] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)
    hot_topic_context: Dict[str, List[Dict]] = field(default_factory=dict)
    evidence: List[Dict] = field(default_factory=list)
    retrieved_messages: List[Dict] = field(default_factory=list)
    entity_profiles: List[Dict] = field(default_factory=list)
    graph_results: List[Dict] = field(default_factory=list)
    web_results: List[Dict] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

class StateOrchestrator(StateMachine):

    start = State(initial=True)
    exploring = State()
    grounded = State()
    web_only = State()
    clarify = State(final=True)
    complete = State(final=True)
    
    search_messages = start.to(exploring) | exploring.to.itself() | grounded.to.itself()
    search_entities = start.to(exploring) | exploring.to.itself() | grounded.to.itself()
    
    get_profile = exploring.to.itself() | grounded.to.itself()
    get_connections = exploring.to.itself() | grounded.to.itself()
    get_activity = exploring.to.itself() | grounded.to.itself()
    
    find_path = grounded.to.itself()
    
    web_search = start.to(web_only)
    
    finish = grounded.to(complete) | web_only.to(complete)
    request_clarification = start.to(clarify) | exploring.to(clarify) | grounded.to(clarify)
    advance = exploring.to(grounded)


    def __init__(self, ctx: "ContextState"):
        self.ctx = ctx
        self._previous_calls: Set[Tuple[str, str]] = set()
        super().__init__()
    
    def validate(self, tool_name: str, args: Dict) -> Tuple[bool, str]:
        if tool_name == "request_clarification":
            allowed = [e.name for e in self.allowed_events]
            if tool_name not in allowed:
                return False, f"cannot clarify from {self.current_state.id}"
            return True, ""
        
        if self.ctx.call_count >= self.ctx.max_calls:
            return False, "call limit reached"
        
        call_sig = (tool_name, str(sorted(args.items())))
        if call_sig in self._previous_calls:
            return False, "duplicate call"
        
        if tool_name == "find_path" and len(self.ctx.inspected_entity_ids) < 2:
            return False, "find_path requires two known entities"
        
        if tool_name == "finish" and not self.can_finish():
            return False, "no evidence gathered"
        
        allowed = [e.name for e in self.allowed_events]
        if tool_name not in allowed:
            return False, f"cannot {tool_name} from {self.current_state.id}"
        
        return True, ""
    
    def record_call(self, tool_name: str, args: Dict):
        call_sig = (tool_name, str(sorted(args.items())))
        self._previous_calls.add(call_sig)
        self.ctx.call_count += 1
        self.ctx.tools_used.append(tool_name)
    
    def try_advance(self):
        if self.current_state == self.exploring:
            has_profiles = len(self.ctx.entity_profiles) > 0
            has_evidence = self.ctx.graph_results or self.ctx.retrieved_messages
            if has_profiles and has_evidence:
                self.advance()
    
    def can_finish(self) -> bool:
        if self.current_state == self.web_only:
            return bool(self.ctx.web_results)
        
        return bool(
            self.ctx.entity_profiles or 
            self.ctx.retrieved_messages or 
            self.ctx.graph_results
        )