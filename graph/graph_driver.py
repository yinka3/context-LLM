from collections import defaultdict
import time
from typing import Any, Dict, TYPE_CHECKING, Optional
import graph_tool.all as gt
import threading
import logging
from shared.dtypes import EdgeData

if TYPE_CHECKING:
    from shared.dtypes import EntityData

logger = logging.getLogger(__name__)

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
            'entity_id': self.graph.new_vertex_property("string"),
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
        self.snapshot_id = 0
        self.ent_to_vertex: Dict[str, Any] = {}
        self.graph_snapshots = Dict[int, 'gt.Graph'] = {}
    
    def add_entity(self, entity_id: str, entity_data: 'EntityData'):
        """Thread-safe edge addition"""
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
    
    def community_detection_within_topics(self,
                                          external_assignments: Dict[str, str], 
                                          use_snapshot=True,
                                          num_samples=1500,
                                          confidence_threshold=0.8):
        
        start_time = time.time()

        if use_snapshot:
            target_graph = self.snapshot_graph()
        else:
            target_graph = self.graph
        

        logger.info(f"starting community detection: {target_graph.num_vertices()} nodes and {target_graph.num_edges()} edges")

        if use_snapshot:
            logger.info(f"it took {time.time() - start_time:.1f}s for snapshot operation")
        

        
        entity_to_vertex = {}
        if use_snapshot:
            entity_id_map = target_graph.vertex_properties['entity_id'] # this should be a string
            for v in target_graph.vertices():
                entity_id = entity_id_map[v]
                entity_to_vertex[entity_id] = v
        else:
            entity_to_vertex = self.ent_to_vertex
        
        topics_to_entities = defaultdict(list)

        for entity_id, topic_name in external_assignments.items():
            if entity_id in entity_to_vertex:
                topics_to_entities[topic_name].append(entity_id)
        
        logger.info(f"there are {len(topics_to_entities)} topics:")
        for topic_name, ent_ids in topics_to_entities.items():
            logger.info(f"  {topic_name}: {len(ent_ids)} entities")
        
        logger.info(f"Topic grouping completed: {time.time() - start_time:.1f}s")
        
        results = {}

        for topic_name, ent_ids in topics_to_entities.items():

            if len(ent_ids) < 3:
                logger.warning(f"Skipping {topic_name}: only {len(ent_ids)} entities")

                results[topic_name] = {
                    "communities": {},
                    'confidence_scores': {},
                    'too_small': True,
                    'entity_count': len(ent_ids)
                }
                continue
            
            step1_time = time.time()
            logger.info(f"Looking into subgraph for {topic_name}")

            topic_vertices = [entity_to_vertex[ent_id] for ent_id in ent_ids]
            filter = target_graph.new_vertex_property("bool")
            for v in topic_vertices:
                filter[v] = True

            topic_subgraph = gt.GraphView(target_graph, vfilt=filter)
            topic_subgraph = gt.Graph(topic_subgraph)

            topic_entity_map = topic_subgraph.vertex_properties['entity_id']
            topic_confidence_map = topic_subgraph.edge_properties['confidence_score']

            logger.info(f"  Topic subgraph: {topic_subgraph.num_vertices()} nodes, {topic_subgraph.num_edges()} edges")

            print(f"Running community detection for {topic_name}...")

            if topic_subgraph.num_edges == 0:
                logger.warning(f" No edges in {topic_name} - skipping community detection")
                results[topic_name] = {
                    'communities': {},
                    'confidence_scores': {},
                    'no_edges': True,
                    'entity_count': len(ent_ids)
                }
                continue

            state = gt.minimize_blockmodel_dl(
                topic_subgraph,
                state=gt.OverlapBlockState,
                state_args=dict(deg_corr=True, recs=topic_confidence_map)
            )

            partition_samples = []
            def collect_topic_sample(state: 'gt.BlockState'):
                partition_copy = state.get_blocks().copy()
                partition_samples.append(partition_copy)
                
                if len(partition_samples) % 100 == 0:
                    logger.info(f"    {topic_name}: {len(partition_samples)}/{num_samples} samples")

            gt.mcmc_equilibrate(state=state,
                                wait=num_samples * 15,
                                mcmc_args=dict(niter=15),
                                callback=collect_topic_sample,
                                verbose=True) # i want to see progress information
            
            logger.info(f"  calculating confidence score for {topic_name}")

            if len(partition_samples) > 0:
                partition_mode = gt.PartitionModeState(partition_samples, converge=True)
        
        


    def snapshot_graph(self) -> 'gt.Graph':
        """Get Snapshot of current graph state"""
        with self._lock:
            snapshot = self.graph.copy()
            self.snapshot_id += 1
            self.graph_snapshots[self.snapshot_id] = (snapshot, time.time())
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
      
