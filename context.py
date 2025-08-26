from datetime import datetime
import logging
import json
import os
from typing import Dict, List, Optional, Set, Tuple, Union
from networkx import DiGraph, Graph
from entity import EntityResolver
from factcheck import FactExtractor
from nlp_pipe import NLP_PIPE
from dtypes import AttributeData, EntityData, MessageData
from vectordb import ChromaClient

class Context:

    def __init__(self, user_name: str = "Yinka"):
        self.next_id = 1
        self.graph: DiGraph = DiGraph()
        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        self.history: List[MessageData] = []
        self.entities: Dict[int, EntityData] = {}
        self.chroma: ChromaClient = ChromaClient()
        self.user_message_cnt: int = 0
        self.vocab = {}
        self.resolution_queue = []

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            vocab_path = os.path.join(script_dir, "vocab.json")
            with open(vocab_path) as file:
                self.vocab = json.load(file)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

        self.user_entity = self._create_user_entity(user_name)

        self.ENTITY_RESOLVER = EntityResolver(
            graph=self.graph,
            chroma=self.chroma,
            user_entity=self.user_entity,
            next_id=self.get_new_id
        )

        self.FACT_EXTRACTOR = FactExtractor(vocab=self.vocab)

    def _create_user_entity(self, user_name: str) -> EntityData:
        user_entity = EntityData(
            id=0, 
            name="USER", 
            type="person",
            aliases=[{"text": user_name, "type": "person"}]
        )
        ent_id = f"ent_{user_entity.id}"
        self.graph.add_node(ent_id, data=user_entity, type="entity")
        self.entities[user_entity.id] = user_entity
        self.chroma.add_item(user_entity)
        return user_entity
    
    def get_new_id(self):
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    def add_to_resolution_queue(self, entity_data, context):
        self.resolution_queue.append({
            'entity': entity_data,
            'message_context': context,
            'surrounding_entities': list(self.entities.keys()),
            'timestamp': datetime.now()
        })
    
    def add(self, item: MessageData):
        self.history.append(item)
        if item.role == "user":
            self.user_message_cnt += 1
        self.process_live_messages(item)
    

    def _apply_actions(self, actions: List[Dict]):
        """Applies the state changes returned by the FactExtractor."""
        for action in actions:
            if not action:
                continue

            subject_id_str = action.get("subject_id")
            if not subject_id_str or not self.graph.has_node(subject_id_str):
                continue
            
            subject_entity: EntityData = self.graph.nodes[subject_id_str]['data']

            if action["action"] == "add_fact":
                relation = action["relation"]
                new_fact = action["new_fact"]
                if relation not in subject_entity.attributes:
                    subject_entity.attributes[relation] = []
                subject_entity.attributes[relation].append(new_fact)
                logging.info(f"Applied Fact: ({subject_entity.name}) -> [{relation}] -> ({new_fact.value.name if isinstance(new_fact.value, EntityData) else new_fact.value})")

            elif action["action"] == "add_attribute":
                key = action["attribute_key"]
                new_attribute = action["new_attribute"]
                if key not in subject_entity.attributes:
                    subject_entity.attributes[key] = []
                subject_entity.attributes[key].append(new_attribute)
                logging.info(f"Applied Attribute: ({subject_entity.name}) -> [{key}] -> ({new_attribute.value})")

    
    def process_live_messages(self, msg: MessageData):
        if self.nlp_pipe.type == "long":
            logging.info("This is only for live stream messages, switching to short type.")
            self.nlp_pipe.type = "short"
        
        msg_id = f"msg_{msg.id}"
        self.graph.add_node(msg_id, data=msg, type="message")

        results = self.nlp_pipe.start_process(msg=msg)
        msg.sentiment = results["sentiment"]

        entities = results["found_entities"]
        noun_chunks = results["noun_chunks"]
        for entity in entities:
            ent_start, ent_end = entity["span"]
            
            for chunk in noun_chunks:
                chunk_start, chunk_end = chunk["span"]
                if ent_start >= chunk_start and ent_end <= chunk_end and chunk["text"] != entity["text"]:
                    entity.setdefault("aliases_to_add", []).append(chunk["text"])
                    break
        
        resolved_entities = {}
        for ents in entities:
            entity = self.ENTITY_RESOLVER.process_entity(ents, msg_id)
            if entity:
                resolved_entities[ents["span"]] = entity
                if entity.id not in self.entities:
                    self.entities[entity.id] = entity

        actions_to_perform = []
        for svo in results["SVO"]:
            subject_entity = self.ENTITY_RESOLVER.find_entity_for_phrase(svo.get("subject"), resolved_entities)
            if not subject_entity:
                continue
            
            object_entity = None
            if "object" in svo:
                object_entity = self.ENTITY_RESOLVER.find_entity_for_phrase(svo.get("object"), resolved_entities)
                if object_entity:
                    action = self.FACT_EXTRACTOR.process_svo(svo, subject_entity, object_entity, msg)
                    actions_to_perform.append(action)
            
            if "prepositional_phrases" in svo:
                for prep_phrase in svo["prepositional_phrases"]:
                    prep_object_entity: EntityData = self.ENTITY_RESOLVER.find_entity_for_phrase(prep_phrase.get("object"), resolved_entities)

                    if prep_object_entity:
                        temp_svo = {"verb": svo["verb"], "prepositional_phrases": [prep_phrase]}
                        action = self.FACT_EXTRACTOR.process_svo(temp_svo, subject_entity, prep_object_entity, msg)
                        actions_to_perform.append(action) 

        for attr in results["attributes"]:
            subject_entity = self.ENTITY_RESOLVER.span_to_entity(attr.get("subject", {}).get("span"), resolved_entities)
            if subject_entity:
                action = self.FACT_EXTRACTOR.process_attribute(attr, subject_entity, msg)
                actions_to_perform.append(action)

        self._apply_actions(actions_to_perform)   
        


        
