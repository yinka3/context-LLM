import time
from typing import Any, Dict, TYPE_CHECKING
import graph_tool.all as gt
import threading

from shared.dtypes import EdgeData

if TYPE_CHECKING:
    from shared.dtypes import EntityData


class ThreadSafeGraph:
    _instance = None
    _lock = threading.RLock()


    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        
        return cls._instance


    def __init__(self):
        self.graph = gt.Graph()

        self.v_property = {
            'entity_id': self.graph.new_vertex_property("int"),
            'entity_name': self.graph.new_vertex_property("string"),
            'entity_type': self.graph.new_vertex_property("string"),
            'data': self.graph.new_vertex_property("object")
        }


        self.e_property = {
            'edge_type': self.graph.new_edge_property("string"),
            'timestamp': self.graph.new_edge_property("int64_t"),
            'data': self.graph.new_edge_property("object")
        }

        self.ent_to_vertex: Dict[int, Any] = {}
    
    def add_entity(self, entity_id: int, entity_data: 'EntityData'):

        with self._lock:
            if entity_id in self.ent_to_vertex:
                return self.ent_to_vertex[entity_id]
            
            v = self.graph.add_vertex()
            self.v_property['entity_id'][v] = entity_id
            self.v_property['entity_name'][v] = entity_data.name
            self.v_property['entity_type'][v] = entity_data.type
            self.v_property['data'][v] = entity_data
            
            self.ent_to_vertex[entity_id] = v
            return v
    

    def add_relationship(self, entity1_id: int, entity2_id: int, 
                        edge_data: 'EdgeData'):
        """Thread-safe edge addition"""
        with self._lock:
            v1 = self.ent_to_vertex.get(entity1_id)
            v2 = self.ent_to_vertex.get(entity2_id)
            
            if v1 is None or v2 is None:
                return None
                
            e = self.graph.add_edge(v1, v2)
            self.e_property['edge_type'][e] = edge_data.bridge.type
            self.e_property['timestamp'][e] = int(time.time())
            self.e_property['data'][e] = edge_data
            return e
    

    def get_neighbors(self, entity_id: int, max_depth: int = 2):

        with self._lock:
            if entity_id not in self.ent_to_vertex:
                return []
            
            start_vertex = self.ent_to_vertex[entity_id]

            dist = gt.shortest_distance(
                self.g,
                source=start_vertex,
                max_dist=max_depth
            )

            neighbors = []
            for v in self.graph.vertices():
                if dist[v] <= max_depth and v != start_vertex:
                    neighbor_id = self.v_property['entity_id'][v]
                    neighbors.append(neighbor_id)
                    
            return neighbors
    
      
