from statemachine import StateMachine, State
from graph.memgraph import MemGraphStore

class StateOrchestrator(StateMachine):

    ready = State("ready", initial=True)

    search_messages = State("search_message")
    inspect_profile = State("inspect_profile")
    query_graph = State("query_graph")
    web_search = State("web_search")

    clarify = State("clarify")
    summarize = State("summarize", final=True)
    failed = State("failed", final=True)

    start_message_search = ready.to(search_messages)
    start_profile_inspect = ready.to(inspect_profile)
    start_graph_query = ready.to(query_graph)

    search_to_profile = search_messages.to(inspect_profile)
    search_to_summarize = search_messages.to(summarize)
    search_to_clarify = search_messages.to(clarify)

    inspect_another = inspect_profile.to.itself()
    profile_to_graph = inspect_profile.to(query_graph)
    profile_to_summarize = inspect_profile.to(summarize)

    query_another = query_graph.to.itself()
    graph_to_profile = query_graph.to(inspect_profile)
    graph_to_web = query_graph.to(web_search)
    graph_to_summarize = query_graph.to(summarize)

    web_to_summarize = web_search.to(summarize)

    clarify_restart = clarify.to(ready)

    give_up = (
        search_messages.to(failed) |
        inspect_profile.to(failed) |
        query_graph.to(failed)
    )