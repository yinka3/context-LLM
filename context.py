from typing import Dict, List, Set, Union
from networkx import DiGraph, Graph

from dtypes import EntityData, MessageData
from vectordb import ChromaClient

class Context:

    def __init__(self):
        self.graph: DiGraph = DiGraph()
        self.history: List[Dict[MessageData, EntityData]] = []
        self.short_context: Set[EntityData] = {}
        self.chroma: ChromaClient = ChromaClient()
        self.user_message_cnt: int = 0

    def add(item: Union[MessageData, EntityData]):
        pass
        
