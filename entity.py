import logging
from typing import Callable, Dict, TYPE_CHECKING, List, Optional, Tuple

from dtypes import EntityData

if TYPE_CHECKING:
    from networkx import DiGraph
    from vectordb import ChromaClient

class EntityResolver:

    def __init__(self, graph: 'DiGraph', chroma: 'ChromaClient',
                 user_entity: EntityData, next_id: Callable):
        self.graph = graph
        self.chroma = chroma
        self.user_entity = user_entity
        self.next_id_func = next_id
    


    def is_more_specific_type(self, new_type: str, existing_type: str):
        specificity_hierarchy = {
            'organization': ['company', 'team'],
            'project': ['initiative', 'product', 'feature'],
        }
        
        for generic, specific_types in specificity_hierarchy.items():
            if existing_type == generic and new_type in specific_types:
                return True
        
        return False
    
    def are_types_compatible(self, type1: str, type2: str):
        compatible_groups = [
            {'person', 'team'},
            {'project', 'product', 'initiative'},
            {'organization', 'company', 'team'},
        ]
        
        if type1 == type2:
            return True
            
        for group in compatible_groups:
            if type1 in group and type2 in group:
                return True
        
        return False
    
    def span_to_entity(self, phrase_span: Tuple[int, int], resolved_entities_by_span: Dict[Tuple[int, int], EntityData]):

        start, end = phrase_span
        for key, ent_obj in resolved_entities_by_span.items():
            ent_start, ent_end = key

            if start < ent_end and end > ent_start:
                return ent_obj
        
        return None

    def find_entity_for_phrase(self, phrase_data: Dict, resolved_entities: Dict[Tuple[int, int], EntityData]):

        entity = self.span_to_entity(phrase_data["span"], resolved_entities)

        if entity:
            return entity
        
        logging.info(f"Span lookup failed for '{phrase_data['text']}, switching to semantic search...")
        found_id_str = self.check_chroma(
            text=phrase_data['text'],
            entity_type=None,
            threshold=0.5
        )

        if found_id_str and self.graph.has_node(found_id_str):
            return self.graph[found_id_str]["data"]

        return None
    
    def check_chroma(self, text: str, entity_type: Optional[str], threshold: float = 0.3,
                     n_result: int = 5):
        
        results = self.chroma.query(text=text, n_results=n_result, node_type="entity")
        if not results['ids'][0]:
            return None

        for i, metadata in enumerate(results['metadatas'][0]):
            if entity_type:
                existing_type = metadata.get('entity_type')
                if not self.are_types_compatible(entity_type, existing_type):
                    continue
                
            if results['distances'][0][i] < threshold:
                return results['ids'][0][i]
        
        return None

    def process_entity(self, entity_data: Dict, msg_id: str) -> EntityData:
        if entity_data["text"].lower() in ['i', 'me', 'my', 'myself']:
            return self.user_entity
        
        confidence_level = entity_data.get("confidence_level")

        if confidence_level == "high":
            threshold = 0.2
        else:
            threshold = 0.4
        
        found_id = self.check_chroma(text=entity_data["text"], 
                                     threshold=threshold,
                                     entity_type=entity_data["type"])
        
        if not found_id:
            return self.create_entity(entity_data, msg_id)
        else:
            return self.merge_entity_data(found_id, entity_data, msg_id)
    
    def create_entity(self, entity_data: Dict, msg_id: str):
        new_ent = EntityData(
            id=self.next_id_func(),
            name=entity_data["text"],
            type=entity_data["type"],
            aliases=[{
                "text": entity_data["text"],
                "type": entity_data["type"]
            }],
            confidence=entity_data.get("confidence", 0.5)
        )

        for alias in entity_data.get("aliases_to_add", []):
            new_ent.aliases.append(
                {
                    "text": alias,
                    "type": entity_data["type"]
                }
            )


        new_ent.mentioned_in.append(msg_id)
        ent_id = f"ent_{new_ent.id}"
        self.graph.add_node(ent_id, data=new_ent, type="entity")
        self.chroma.add_item(new_ent)
        self.graph.add_edge(
            msg_id, 
            ent_id, 
            type="mentions",
            confidence=new_ent.confidence)
        
        logging.info(f"Created new entity: {new_ent.name} (type: {new_ent.type})")
        return new_ent
            
    def merge_entity_data(self, existing_id: str, new_entity_data: Dict, msg_id: str):
        existing_ent: EntityData = self.graph.nodes[existing_id]['data']

        new_text = new_entity_data["text"]
        if not any(alias["text"] == new_text for alias in existing_ent.aliases):
            existing_ent.aliases.append({
                "text": new_text,
                "type": new_entity_data["type"]
            })
        
        new_confidence = new_entity_data.get("confidence", 0.5)
        existing_ent.confidence = max(existing_ent.confidence, new_confidence)

        if self.is_more_specific_type(new_entity_data["type"], existing_ent.type):
            existing_ent.type = new_entity_data["type"]
        

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

        logging.info(f"Merged entity: {existing_ent.name} with alias: {new_text}")
        return existing_ent