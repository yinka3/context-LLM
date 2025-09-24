from collections import defaultdict
import time
from typing import Any, Dict, TYPE_CHECKING, Optional
import graph_tool.all as gt
from concurrent.futures import ThreadPoolExecutor
import logging

from redis import Redis
from shared.dtypes import EdgeData

if TYPE_CHECKING:
    from shared.dtypes import EntityData

logger = logging.getLogger(__name__)

#NOTE its not thread safe, i need to define the data coming in and out before making this thread safe
class KnowGraph:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        
        return cls._instance


    def __init__(self):
        self.graph = gt.Graph()

        self.v_property = {
            'entity_id': self.graph.new_vertex_property("string"),
            'entity_name': self.graph.new_vertex_property("string"),
            'entity_type': self.graph.new_vertex_property("string"),
            'aliasis': self.graph.new_vertex_property("object", vals=[]),
            'topic': self.graph.new_vertex_property("string"),
            'origin_msg': self.graph.new_vertex_property("string"),
            'community_id': self.graph.new_vertex_property("int"),
            'confidence_score': self.graph.new_edge_property("double"),
            'mentioned_in': self.graph.new_vertex_property("object", vals=[]),
            'context_mentions': self.graph.new_vertex_property("object", vals=[]),
            'page_rank_hist': self.graph.new_vertex_property("object", vals={}),
            'data': self.graph.new_vertex_property("object")
        }

        self.e_property = {
            'relation_type': self.graph.new_edge_property("string"),
            'timestamp': self.graph.new_edge_property("int64_t"),
            'messages_connected': self.graph.new_edge_property("object", []),
            'confidence_score': self.graph.new_edge_property("double"),
            'data': self.graph.new_edge_property("object")
        }

        self.vertex_stats = {} 
        self.current_snapshot_id = 0
        self.ent_to_vertex: Dict[str, Any] = {}
        self.graph_snapshots = Dict[int, 'gt.Graph'] = {}
    
    def add_entity(self, entity_data: dict):
        """Thread-safe edge addition"""
        
        if entity_data["id"] in self.ent_to_vertex:
            return
        
        v = list(self.graph.add_vertex())
        self.v_property['entity_id'][v] = entity_data["id"]
        self.v_property['entity_name'][v] = entity_data["name"]
        self.v_property['entity_type'][v] = entity_data["type"]
        self.v_property['topic'][v] = entity_data["id"]
        self.v_property['confidence_score'] = entity_data["confidence"]
        self.v_property['data'][v] = entity_data
        
        self.ent_to_vertex[entity_data["id"]] = v
        return v
    

    def add_relationship(self, entity1_id: str, entity2_id: str, 
                        edge_data: 'EdgeData'):
        """Thread-safe edge addition"""
        v1 = self.ent_to_vertex.get(entity1_id)
        v2 = self.ent_to_vertex.get(entity2_id)
        
        if v1 is None or v2 is None:
            return None
            
        e = self.graph.add_edge(v1, v2)
        self.e_property['relation_type'][e] = edge_data.bridge.type
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
    
    def snapshot_graph(self) -> 'gt.Graph':
        """Get Snapshot of current graph state"""
        snapshot = self.graph.copy()
        self.current_snapshot_id += 1
        self.graph_snapshots[self.current_snapshot_id] = (snapshot, time.time())
        return snapshot


    #NOTE look into graph_tool.topology, see if its possible to use similarity to check how far off or on page_rank was? when time permits
    # link: https://graph-tool.skewed.de/static/docs/stable/topology.html
    # using the similarity funtion in this subsection
    # look at dominator_tree - find the main vertex aka node in a community or in fildered GraphView
    # topo-sort can be used but propably mainly fo return vertices in order based on edges but edges can change over time, no?

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

            logger.info(f"Running community detection for {topic_name}...")

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
                                multiflip=True,
                                callback=collect_topic_sample,
                                verbose=True) # i want to see progress information
            
            logger.info(f"  calculating confidence score for {topic_name}")

            if len(partition_samples) > 0:
                partition_mode = gt.PartitionModeState(partition_samples, converge=True)
                aligned_samples = partition_mode.get_partitions()

                topic_confidences = {}
                for v in topic_subgraph.vertices():
                    entity_id = topic_entity_map[v]
                    community_counts = defaultdict(int)

                    for partition in aligned_samples:
                        community_id = partition[v]
                        community_counts[community_id] += 1
                    
                    total_samples = len(aligned_samples)
                    community_confidences = {}
                    for community_id, count in community_counts.items():
                        confidence_score = count / total_samples
                        community_confidences[community_id] = confidence_score
                    
                    topic_confidences[entity_id] = community_confidences
                
                topic_results = {
                    'communities': {},
                    'confidence_scores': {},
                    'entity_count': len(ent_ids),
                    'processing_time': time.time() - step1_time
                }

                for entity_id, confidences in topic_confidences.items():
                    high_confidence_communities = [
                        (comm_id, conf) for comm_id, conf in confidences.items()
                        if conf >= confidence_threshold
                    ]
                    
                    if high_confidence_communities:
                        high_confidence_communities.sort(key=lambda x: x[1], reverse=True)
                        
                        topic_results['confidence_scores'][entity_id] = {
                            'topic': topic_name,
                            'community_ids': [sub_id for sub_id, _ in high_confidence_communities],
                            'community_confidences': dict(high_confidence_communities),
                            'max_confidence': max(confidences.values()),
                            'all_community_confidences': confidences
                        }
                    else:
                        # Low confidence - potential topic misassignment or bridge entity
                        topic_results['confidence_scores'][entity_id] = {
                            'topic': topic_name,
                            'community_ids': [],
                            'community_confidences': {},
                            'max_confidence': max(confidences.values()),
                            'potential_misassignment': True,  # Flag for your re-assignment system
                            'all_community_confidences': confidences
                        }
                
                community_groups = defaultdict(list)
                for entity_id, conf_info in topic_results['confidence_scores'].items():
                    for subcomm_id in conf_info.get('community_ids', []):
                        community_groups[subcomm_id].append(entity_id)
                
                topic_results['communities'] = dict(community_groups)
                
                logger.info(f"  {topic_name} completed: {len(community_groups)} communities found")
                
            else:
                logger.info(f"  No samples collected for {topic_name}")
                topic_results = {'error': 'No samples collected'}
            
            results[topic_name] = topic_results
        
        final_results = {
            'per_topic_communities': results,
            'potential_reassignments': [],
            'summary': {
                'total_topics': len(topics_to_entities),
                'total_entities_processed': sum(len(entities) for entities in topics_to_entities.values()),
                'total_time_minutes': (time.time() - start_time) / 60
            }
        }

        for topic_name, topic_results in results.items():
            if 'confidence_scores' in topic_results:
                for entity_id, conf_info in topic_results['confidence_scores'].items():
                    if conf_info.get('potential_misassignment', False):
                        final_results['potential_reassignments'].append({
                            'entity_id': entity_id,
                            'current_topic': topic_name,
                            'max_confidence': conf_info['max_confidence'],
                            'suggested_action': 'review_assignment'
                        })

        logger.info(f"\nCommunity detection completed! Total time: {(time.time() - start_time)/60:.1f}m")
        logger.info(f"Found {len(final_results['potential_reassignments'])} potential topic reassignments")
        
        return final_results