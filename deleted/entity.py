import logging
from metaphone import doublemetaphone
from typing import Callable, Dict, TYPE_CHECKING, List, Optional, Tuple
from shared.dtypes import EntityData
import networkx as nx
from main.redisclient import RedisClient

if TYPE_CHECKING:
    from networkx import DiGraph
    from main.vectordb import ChromaClient


logger = logging.getLogger(__name__)

ENTITY_MAP = Dict[Tuple[int, int], EntityData]

class EntityResolver:

    def __init__(self, graph: 'DiGraph', chroma: 'ChromaClient',
                 user_entity: EntityData, next_id: Callable, alias_index: Dict[str, EntityData], redis_client: RedisClient):
        self.graph = graph
        self.chroma = chroma
        self.user_entity = user_entity
        self.next_id_func = next_id
        self.alias_index = alias_index
        self.redis_client = redis_client
        self.blocking_index = {}

    
    def _request_disambiguation_clarification(self, entity_data: Dict, candidates: List[EntityData]):
        candidate_names = [c.name for c in candidates]
        logger.info(
            f"CLARIFICATION NEEDED: For '{entity_data['text']}', which did you mean? {candidate_names}"
        )
        pass

    def _check_candidate_ambiguity(self, top_candidates: List[EntityData], entity_data: Dict) -> bool:
        """
        Hook for subclasses to implement ambiguity detection.
        The base class has no ambiguity detection, so it always returns False.
        """
        return False
        
    def _request_alias_clarification(self, entity: EntityData, potential_alias_text: str):
        """
        STUB: This function will eventually trigger a clarification question to the user.
        """
        logger.info(
            f"CLARIFICATION NEEDED: Is '{potential_alias_text}' a new name for '{entity.name}'?"
        )
        # In a real system, this would flag the agent to ask the user on its next turn.
        pass

    def _handle_alias_creation(self, new_mention_data: Dict, matched_entity: EntityData,
                               coref_clusters: List, appositive_map: Dict):
        
        new_text = new_mention_data["text"]
        new_span = new_mention_data["span"]
        
        known_names = {alias['text'].lower() for alias in matched_entity.aliases}
        known_names.add(matched_entity.name.lower())

        if new_text.lower() in known_names:
            return

        for head_span, appos_span in appositive_map.items():
            if appos_span == new_span:
                # The new mention is an appositive. Now check if its head
                # has been resolved to our matched_entity.
                # (This part requires a more complex mapping of spans to resolved entities,
                # for now we'll assume a direct text match is a strong signal).
                logger.info(f"High-confidence alias found via appositive: '{new_text}' for '{matched_entity.name}'")
                matched_entity.aliases.append({"text": new_text, "type": new_mention_data["type"]})
                return

        for cluster in coref_clusters:
            cluster_texts = {mention.text.lower() for mention in cluster.mentions}
            if new_text.lower() in cluster_texts and any(name in cluster_texts for name in known_names):
                logger.info(f"High-confidence alias found via coreference: '{new_text}' for '{matched_entity.name}'")
                matched_entity.aliases.append({"text": new_text, "type": new_mention_data["type"]})
                return
        
        self._request_alias_clarification(matched_entity, new_text)

    def _build_coref_span_map(self, coref_clusters: List) -> ENTITY_MAP:
        """Maps mention spans to their main mention span"""
        span_map = {}

        for cluster in coref_clusters:
            if len(cluster.mentions) < 2:
                continue

            main_span = (cluster.main.start_char, cluster.main.end_char)

            for mention in cluster.mentions:
                mention_span = (mention.start_char, mention.end_char)

                if mention_span != main_span:
                    span_map[mention_span] = main_span
        
        return span_map
    
    def _get_match_score(self, search_text: str, candidate: EntityData):
        """Multi-level scoring with strict thresholds"""
        
        # Exact match on name or aliases
        if search_text == candidate.name.lower():
            return 1.0
            
        for alias in candidate.aliases:
            if search_text == alias["text"].lower():
                return 0.95
        
        from difflib import SequenceMatcher
        
        # Check name similarity
        name_sim = SequenceMatcher(None, search_text.lower(), candidate.name.lower()).ratio()
        if name_sim >= 0.9:
            return name_sim * 0.9
        
        for alias in candidate.aliases:
            alias_sim = SequenceMatcher(None, search_text.lower(), alias["text"].lower()).ratio()
            if alias_sim >= 0.9:
                return alias_sim * 0.85
        

        tokens1 = set(search_text.lower().split())
        tokens2 = set(candidate.name.lower().split())
        token_sim = len(tokens1 & tokens2) / max(len(tokens1), len(tokens2))
        
        #NOTE This is just a random threshold, did not think about it for real
        if name_sim > 0.8 and token_sim > 0.8:
            return (0.5 * name_sim + 0.2 * token_sim)
        
        return 0.0
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if entity types are compatible for merging"""
        
        type_groups = {
            "person": {"person", "people", "human", "possessive_entity"},
            "organization": {"organization", "company", "corporation", "org", "group_of_entities"},
            "location": {"location", "place", "city", "country", "region"},
            "work": {"work_product_or_project", "project", "product"},
            "concept": {"academic_concept", "technology", "topic"},
            "temporal": {"date", "time", "event"},
        }
        
        type1_lower = type1.lower() if type1 else ""
        type2_lower = type2.lower() if type2 else ""
        
        # Same type is always compatible
        if type1_lower == type2_lower:
            return True
        
        # Check if in same type group
        for group in type_groups.values():
            if type1_lower in group and type2_lower in group:
                return True
        
        return False
    
    def _resolve_pronouns(self, coref_clusters: List, 
                          resolved_entities: ENTITY_MAP, 
                          max_distance: int = 250) -> ENTITY_MAP:

        pronoun_map = {}

        for cluster in coref_clusters:
            anchor_entity = None
            anchor_span = None

            for mention in cluster.mentions:
                mention_span = (mention.start_char, mention.end_char)
                if mention_span in resolved_entities:
                    anchor_entity = resolved_entities[mention_span]
                    anchor_span = mention_span
                    break
            
            if not anchor_entity:
                continue

            for mention in cluster.mentions:
                mention_span = (mention.start_char, mention.end_char)

                #skip if the mentioned span has already been resolved
                if mention_span in resolved_entities or mention_span in pronoun_map:
                    continue

                if anchor_span:
                    distance = abs(mention.start_char - anchor_span[0])
                    if distance > max_distance:
                        logger.warning(f"Pronoun '{mention.text}' too far from anchor '{anchor_entity.name}' ({distance} chars)")
                        continue
                
                pronoun_map[mention_span] = anchor_entity
                logger.info(f"Coreference: '{mention.text}' -> '{anchor_entity.name}'")
        
        return pronoun_map
    
    def find_exact_match(self, entity_data: Dict, alias_index: Dict[str, EntityData]):

        user_pronouns = {'i', 'me', 'my', 'myself'}
        search_text = entity_data["text"].lower()
        if search_text in user_pronouns:
            return self.user_entity
        
        return alias_index.get(search_text)
    
    def resolve_entities_with_coreference(self, entities_from_nlp: List[Dict],
                                          coref_clusters: List,
                                          msg_id: str,
                                          conversational_context,
                                          appositive_map: Dict = {}) -> ENTITY_MAP:
        """Main entry point for entity resolution with coreference awareness."""
        coref_span_map = self._build_coref_span_map(coref_clusters)

        resolved_entities = {}
        unresolved = []
        
        for ent_data in entities_from_nlp:
            entity = self.find_exact_match(ent_data, self.alias_index)
            if entity:
                resolved_entities[ent_data["span"]] = entity
            else:
                unresolved.append(ent_data)
        
        for ent_data in unresolved[:]:
            main_span = coref_span_map.get(ent_data["span"])
            if main_span and main_span in resolved_entities:
                resolved_entities[ent_data["span"]] = resolved_entities[main_span]
                unresolved.remove(ent_data)

        anchors = list(resolved_entities.values())
        for ent_data in unresolved:
            entity = self.process_entity(
                ent_data, 
                msg_id,
                context_entities=anchors,
                conversational_context=conversational_context,
                entity_type_filter=ent_data.get("type")
            )
            if entity:
                resolved_entities[ent_data["span"]] = entity
                self._handle_alias_creation(ent_data, entity, coref_clusters, appositive_map)
        
        pronoun_map = self._resolve_pronouns(coref_clusters, resolved_entities)
        resolved_entities.update(pronoun_map)

        return resolved_entities
    
    def get_graph_candidates(self, context_entities: List[EntityData], 
                           max_depth: int = 2) -> List[EntityData]:
        """
        Get candidate entities from graph based on context entities.
        Uses BFS to find related entities within max_depth.
        """
        candidate_set = set()
        
        for anchor_entity in context_entities:
            anchor_node_id = f"ent_{anchor_entity.id}"
            
            if not self.graph.has_node(anchor_node_id):
                continue
            
            # BFS from this anchor with depth limit
            try:
                bfs_tree = nx.bfs_tree(self.graph, source=anchor_node_id, depth_limit=max_depth)
                
                for node_id in bfs_tree.nodes():
                    # Only consider entity nodes (not message nodes)
                    if node_id.startswith("ent_") and node_id != anchor_node_id:
                        if 'data' in self.graph.nodes[node_id]:
                            candidate_set.add(self.graph.nodes[node_id]['data'])
            except nx.NetworkXError as e:
                logger.warning(f"Graph traversal error for {anchor_node_id}: {e}")
                continue

        return list(candidate_set)

    def create_entity(self, entity_data: Dict, msg_id: str):

        new_ent = EntityData(
            id=self.next_id_func(),
            name=entity_data["text"],
            type=entity_data["type"],
            aliases=[{
                "text": entity_data["text"],
                "type": entity_data["type"]
            }],
            confidence=entity_data.get("confidence", 0.7)
        )

        if "contextual_mention" in entity_data:
            new_ent.contextual_mentions.append(entity_data["contextual_mention"])

        new_ent.mentioned_in.append(msg_id)
        ent_id = f"ent_{new_ent.id}"
        self.graph.add_node(ent_id, data=new_ent, type="entity")

        # if self.chroma:
        #     self.chroma.add_item(new_ent)

        self.graph.add_edge(
            msg_id, 
            ent_id, 
            type="mentions",
            confidence=new_ent.confidence)
        
        logger.info(f"Created new entity: {new_ent.name} (type: {new_ent.type})")
        self.update_blocking_index(new_ent)
        return new_ent
            
    def merge_entity_data(self, existing_id: str, new_entity_data: Dict, msg_id: str):

        existing_ent: EntityData = self.graph.nodes[existing_id]['data']
        new_text: str = new_entity_data["text"]
        
        new_confidence = new_entity_data.get("confidence", 0.7)
        existing_ent.confidence = max(existing_ent.confidence, new_confidence)

        for alias in new_entity_data.get("aliases_to_add", []):
            existing_ent.aliases.append(
                {
                    "text": alias,
                    "type": new_entity_data["type"]
                }
            )
        
        self.graph.add_edge(
            msg_id, 
            existing_id, 
            type="mentions",
            confidence=existing_ent.confidence
        )

        if msg_id not in existing_ent.mentioned_in:
            existing_ent.mentioned_in.append(msg_id)
        logger.info(f"Merged entity: {existing_ent.name} with alias: {new_text}")
        self.update_blocking_index(existing_ent)
        return existing_ent
    
    def process_entity(self, entity_data: Dict, msg_id: str, 
                      context_entities: List[EntityData],
                      conversational_context: str = "",
                      entity_type_filter: Optional[str] = None):
        
        search_text = entity_data["text"].lower()
        match_threshold = 0.85

        if context_entities:
            candidates = self.get_graph_candidates(context_entities, max_depth=2)

            if entity_type_filter:
                candidates = [c for c in candidates 
                             if self._are_types_compatible(c.type, entity_type_filter)]
            
            scored_candidates = []
            for candidate in candidates:
                score = self._get_match_score(search_text, candidate)

                if conversational_context and candidate.name.lower() in conversational_context.lower():
                    score *= 1.2

                if score >= match_threshold:
                    scored_candidates.append((candidate, score))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [c for c, _ in scored_candidates]
            
            if len(top_candidates) > 1:
                if self._check_candidate_ambiguity(top_candidates, entity_data):
                    self._request_disambiguation_clarification(entity_data, top_candidates)
                    return None
                
            if top_candidates:
                best_match = top_candidates[0]
                best_score = scored_candidates[0][1]
                logger.info(f"GRAPH MATCH: '{entity_data['text']}' -> '{best_match.name}' (score: {best_score:.2f})")
                return self.merge_entity_data(f"ent_{best_match.id}", entity_data, msg_id)
            

        logger.info(f"Creating new entity for '{entity_data['text']}'")
        return self.create_entity(entity_data, msg_id)
    

    def create_blocking_keys(self, entity_text: str):

        keys = set()
        text_lower = entity_text.lower()
        
        if len(text_lower) >= 3:
            keys.add(f"prefix_{text_lower[:3]}")
        
        tokens = text_lower.split()
        if tokens:
            keys.add(f"token_{tokens[0]}")
        
        significant_word = tokens[-1] if tokens else ""
        if significant_word:
            primary_key, secondary_key = doublemetaphone(significant_word)
            if primary_key:
                keys.add(f"ph_{primary_key}")
            if secondary_key:
                keys.add(f"ph_{secondary_key}")
        
        return keys
    
    def update_blocking_index(self, entity: EntityData):
        keys = self.create_blocking_keys(entity.name)
        for key in keys:
            self.blocking_index.setdefault(key, set()).add(entity.id)

    def get_candidates_with_blocking(self, entity_data: Dict):

        blocking_keys = self.create_blocking_keys(entity_data["text"])
        candidates_ids = set()


        for key in blocking_keys:
            if key in self.blocking_index:
                candidates_ids.update(self.blocking_index[key])
    

        return [self.graph.nodes[f"ent_{id}"]['data'] for id in candidates_ids if f"ent_{id}" in self.graph.nodes]





# if self.chroma:
        #     #NOTE have to change the where condition, the type is only ent and meessage
        #     semantic_matches = self.chroma.query(
        #         text=entity_data["text"],
        #         n_results=5,
        #         where={"type": entity_type_filter} if entity_type_filter else None
        #     )
            
        #     if semantic_matches and semantic_matches['distances'][0][0] < 0.3:
        #         # Very high confidence threshold for auto-merge
        #         entity_id = semantic_matches['ids'][0][0]
        #         logger.info(f"SEMANTIC MATCH: '{entity_data['text']}' -> entity {entity_id}")
        #         return self.graph.nodes[entity_id]['data']

    # def check_chroma(self, text: str, entity_type: Optional[str], 
    #                  auto_merge_threshold: float = 0.2,
    #                  flag_threshold: float = 0.55,
    #                  n_result: int = 5) -> Tuple[Optional[str], Optional[Dict]]:
        
    #     results = self.chroma.query(text=text, n_results=n_result, node_type="entity")
    #     if not results or not results.get('ids') or not results['ids'][0]:
    #         return None, None

    #     best_candidate_id = results['ids'][0][0]
    #     best_distance = abs(results['distances'][0][0])
    #     best_metadata = results['metadatas'][0][0]

    #     if entity_type:
    #         existing_type = best_metadata.get('entity_type')
    #         if not self.are_types_compatible(entity_type, existing_type):
    #             return None, None
        
    #     if best_distance <= auto_merge_threshold:
    #         return best_candidate_id, None
    #     elif best_distance <= flag_threshold and best_distance > auto_merge_threshold:
    #         flag_candidates = []
    #         for i, entity_id in enumerate(results['ids'][0]):
    #             distance = abs(results['distances'][0][i])
    #             if distance <= flag_threshold and distance > auto_merge_threshold:
    #                 flag_candidates.append({
    #                     "entity_id": entity_id,
    #                     "distance": distance
    #                 })
            
    #         return None, {"candidates": flag_candidates}
    #     return None, None