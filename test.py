from context import Context
from dtypes import MessageData

CONTEXT = Context()
MESSAGE_ID_COUNTER = 0
def process_test_message(msg: str):
    global MESSAGE_ID_COUNTER
    
    msg_data = MessageData(
        id=MESSAGE_ID_COUNTER,
        role="user",
        message=msg,
        sentiment="unknown"
    )
    CONTEXT.add(msg_data)
    MESSAGE_ID_COUNTER += 1


def test_conversation_flow():
    process_test_message("Ashlynn works on the Phoenix project.")

    # ASSERTION 1: Check that two entity nodes were created in the graph.
    # We expect 3 nodes total: 1 for the message, 2 for the entities.
    assert len(CONTEXT.graph.nodes) == 3 

    ashlynn_node = None
    phoenix_node = None
    for _, node_data in CONTEXT.graph.nodes(data=True):
        if node_data.get("type") == "entity":
            if node_data["data"].name == "Ashlynn":
                ashlynn_node = node_data["data"]
            elif node_data["data"].name == "the Phoenix project":
                phoenix_node = node_data["data"]

    # ASSERTION 2: Verify the relationship was created correctly.
    assert "works_on" in ashlynn_node.attributes
    print(ashlynn_node.attributes)
    assert ashlynn_node.attributes["works_on"][0].value == phoenix_node

    process_test_message("Ashlynn lives in New York.")

    # ASSERTION 3: Check that only one new entity was created (for New York).
    # We now expect 5 nodes: 2 messages + 3 entities.
    assert len(CONTEXT.graph.nodes) == 5

    # ASSERTION 4: Verify the new "lives_in" attribute was added to Ashlynn.
    assert "lives_in" in ashlynn_node.attributes
    assert ashlynn_node.attributes["lives_in"][0].value.name == "New York"


test_conversation_flow()
