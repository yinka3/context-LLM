import time
from typing import Any, Dict, TYPE_CHECKING, Optional
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
            'entity_id': self.graph.new_vertex_property("str"),
            'entity_name': self.graph.new_vertex_property("string"),
            'entity_type': self.graph.new_vertex_property("string"),
            'page_rank': self.graph.new_vertex_property("double"),
            'data': self.graph.new_vertex_property("object")
        }

        self.e_property = {
            'edge_type': self.graph.new_edge_property("string"),
            'timestamp': self.graph.new_edge_property("int64_t"),
            'confidence_score': self.graph.new_edge_property("double"),
            'data': self.graph.new_edge_property("object")
        }

        self.vertex_stats = {} 

        self.ent_to_vertex: Dict[str, Any] = {}
        self.graph_snapshots = Dict[int, 'gt.Graph'] = {}
    
    def add_entity(self, entity_id: str, entity_data: 'EntityData'):

        with self._lock:
            if entity_id in self.ent_to_vertex:
                return self.ent_to_vertex[entity_id]
            
            v = list(self.graph.add_vertex())
            self.v_property['entity_id'][v] = entity_id
            self.v_property['entity_name'][v] = entity_data.name
            self.v_property['entity_type'][v] = entity_data.type
            self.v_property['data'][v] = entity_data
            
            self.ent_to_vertex[entity_id] = v
            return v
    

    def add_relationship(self, entity1_id: str, entity2_id: str, 
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
            self.e_property['confidence_score'][e] = edge_data.confidence
            self.e_property['data'][e] = edge_data
            return e
    
    def get_vertex_stats(self, snapshot_graph: Optional['gt.Graph'] = None):


        if snapshot_graph:
            target_graph = snapshot_graph
        else:
            target_graph = self.graph

        stats = {
            "avg_in_degree": gt.vertex_average(g=target_graph, deg='in'),
            "avg_out_degree": gt.vertex_average(g=target_graph, deg='out'),
            "avg_total_degree": gt.vertex_average(g=target_graph, deg='total')
        }

        if not snapshot_graph:
            self.vertex_stats = stats
            
        return stats

    def get_neighbors(self, entity_id: str, max_depth: int = 2):

        with self._lock:
            if entity_id not in self.ent_to_vertex:
                return []
            
            start_vertex = self.ent_to_vertex[entity_id]

            dist = gt.shortest_distance(
                self.graph,
                source=start_vertex,
                max_dist=max_depth)

            neighbors = []
            for v in self.graph.vertices():
                if dist[v] <= max_depth and v != start_vertex:
                    neighbor_id = self.v_property['entity_id'][v]
                    neighbors.append(neighbor_id)
                    
            return neighbors
    
    def community_dectection(self, use_snapshot=True):

        if use_snapshot:
            target_graph = self.snapshot_graph()
        else:
            target_graph = self.graph
        
        state = gt.minimize_blockmodel_dl(target_graph,
                                          state=gt.OverlapBlockState, 
                                          state_args=dict(deg_corr=True))
        
        community_map = state.get_blocks()

        communities = {}
        entity_id_map = target_graph.vertex_properties['entity_id']
        for v in target_graph.vertices():
            entity_id = entity_id_map[v]
            community_id = community_map[v]
            communities[entity_id] = community_id
        
        return communities


    def snapshot_graph(self) -> 'gt.Graph':

        with self._lock:
            snapshot = gt.Graph(self.graph)
            snapshot_id = time.time()
            self.graph_snapshots[snapshot_id] = snapshot

            return snapshot
    

    def page_rank(self):


        snapshot = self.snapshot_graph()
        weight_map = snapshot.edge_properties['confidence_score']
        rank_map = snapshot.vertex_properties['page_rank']
        gt.pagerank(g=snapshot, weight=weight_map, prop=rank_map)
        
        ranks = {}
        entity_id_map = snapshot.vertex_properties['entity_id']

        for v in snapshot.vertices():
            entity_id = entity_id_map[v]
            rank_score = rank_map[v]
            ranks[entity_id] = rank_score
        
        return ranks
      
