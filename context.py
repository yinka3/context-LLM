from datetime import datetime
import logging
import logging_setup
import json
import os
from typing import Dict, List, Optional, Set, Tuple, Union
from networkx import DiGraph, Graph
from entity import EntityResolver
from factcheck import FactExtractor
from nlp_pipe import NLP_PIPE
from dtypes import AttributeData, EntityData, MessageData
from vectordb import ChromaClient
from spacy.tokens import Token

logging_setup.setup_logging()

logger = logging.getLogger(__name__)

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

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            vocab_path = os.path.join(script_dir, "vocab.json")
            with open(vocab_path) as file:
                self.vocab = json.load(file)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

        self.user_entity = self._create_user_entity(user_name)

        self.ENTITY_RESOLVER = EntityResolver(
            graph=self.graph,
            chroma=self.chroma,
            user_entity=self.user_entity,
            next_id=self.get_new_id
        )

        self.FACT_EXTRACTOR = FactExtractor()

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
    
    
    def add(self, item: MessageData):
        self.history.append(item)
        if item.role == "user":
            self.user_message_cnt += 1
        self.process_live_messages(item)
    
    def flag_and_apply(self, actions: List[Dict], msg: MessageData, threshold: 0.75):
        """
        Applies high-confidence actions directly and flags low-confidence ones for Tier 2.
        """
        for action in actions:
            if not action:
                continue

            confidence = action["new_fact" if "new_fact" in action else "new_attribute"].confidence_score

            if confidence >= threshold:
                self._apply_state_action(action)
            else:
                subject_id = action.get("subject_id")
                subject_entity: EntityData = self.graph.nodes[subject_id]["data"]

                payload = {
                    "text": msg.message,
                    "action_details": action,
                    "entity_name": subject_entity.name
                }

                # a function that sends the payload dict off to tier 2, in that function we will create a Tier2Task data obj

                logger.warning(f"""Flagged for Tier 2: 
                                Low confidence fact ({confidence:.2f}) for entity '{subject_entity.name}'. 
                                Action not applied for message: {msg.message}. | Payload Info: {payload}""")

    def _apply_state_action(self, action: Dict):
        """Applies a state change returned by the FactExtractor."""
        subject_id_str = action.get("subject_id")
        if not subject_id_str or not self.graph.has_node(subject_id_str):
            return
        
        subject_entity: EntityData = self.graph.nodes[subject_id_str]['data']

        if action["action"] == "add_fact":
            new_fact: AttributeData = action["new_fact"]
            target_entity: EntityData = new_fact.value
            target_id_str = f"ent_{target_entity.id}"

            if not self.graph.has_node(target_id_str): return

            edge_attributes = {
                "relation": action["relation"],
                "confidence": new_fact.confidence_score,
                "message_id": f"msg_{new_fact.message.id}"
            }

            if "source_entity_id" in action:
                edge_attributes["source_id"] = action["source_entity_id"]
            
            self.graph.add_edge(subject_id_str, target_id_str, **edge_attributes)
            logger.info(f"Applied Fact Edge: ({subject_entity.name}) -> [{action['relation']}] -> ({target_entity.name})")

        elif action["action"] == "add_attribute":
            key = action["attribute_key"]
            new_attribute = action["new_attribute"]
            if key not in subject_entity.attributes:
                subject_entity.attributes[key] = []
            subject_entity.attributes[key].append(new_attribute)
            logger.info(f"Applied Attribute: ({subject_entity.name}) -> [{key}] -> ({new_attribute.value})")

    
    def process_live_messages(self, msg: MessageData):
        
        msg_id = f"msg_{msg.id}"
        self.graph.add_node(msg_id, data=msg, type="message")

        results = self.nlp_pipe.start_process(msg=msg, entity_threshold=0.7)

        if results.get("tier2_flags"):
            for flag in results["tier2_flags"]:
                logger.warning(f"Received Tier 2 flag from NLP_PIPE. Reason: {flag['reason']}")
        
        primary_emotion = max(results["emotion"], key=lambda x: x['score'])
        msg.sentiment = primary_emotion['label']

        entities = results["high_confidence_entities"]
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
            entity, tier2_payload = self.ENTITY_RESOLVER.process_entity(ents, msg_id)
            
            if tier2_payload:
                logger.warning(f"""Flagged for Tier 2: Potential merge conflict between new entity 
                                '{tier2_payload['new_entity_name']}' and existing entity '{tier2_payload['conflicting_entity_name']}'.
                                Message: {msg.message} | Payload Info: {payload} \n""")
            if entity:
                resolved_entities[ents["span"]] = entity
                if entity.id not in self.entities:
                    self.entities[entity.id] = entity
        
        coref_mention_map = {}
        if results.get("coref_clusters"):
            for cluster in results["coref_clusters"]:
                
                main_mention_span = cluster.main.start_char, cluster.main.end_char
                
                main_entity = resolved_entities.get(main_mention_span)
                if not main_entity:
                    for span, entity in resolved_entities.items():
                        if main_mention_span[0] >= span[0] and main_mention_span[1] <= span[1]:
                            main_entity = entity
                            break
                
                # if there still isnt a main entity then continue
                if not main_entity:
                    continue

                entities_cluser_set = {main_entity.id}
                for mention in cluster.mentions:
                    mention_span = (mention.start_char, mention.end_char)
                    for r_span, r_entity in resolved_entities.items():
                        if mention_span[0] >= r_span[0] and mention_span[1] <= r_span[1]:
                            entities_cluser_set.add(r_entity.id)
                
                if len(entities_cluser_set) > 1:

                    payload = {
                        "text": msg.message,
                        "cluster_contamination": True,
                        "main_mention": cluster.main.text,
                        "conflicting_entity_ids": list(entities_cluser_set),
                        "cluster_mentions": [m.text for m in cluster.mentions]
                    }

                    logger.warning(f"""Flagged for Tier 2: Coref cluster for 
                                    '{cluster.main.text}' is contaminated with multiple entities.
                                    Message: {msg.message} | Payload Info: {payload} \n""")
                    continue

                for mention in cluster.mentions:
                    if mention is cluster.main:
                        continue
                        
                    distance = mention.start_char - cluster.main.end_char

                    if distance > 250:
                        payload = {
                            "text": msg.message,
                            "pronoun_distance": distance,
                            "main_mention": cluster.main.text,
                            "distant_mention": mention.text,
                            "main_entity_id": main_entity.id
                        }

                        logger.warning(f"""Flagged for Tier 2: 
                                        Coref mention '{mention.text}' is very far ({distance} chars) from '{cluster.main.text}'.
                                        Message: {msg.message} | Payload Info: {payload} \n""")
                        continue
                    
                    mention_span = (mention.start_char, mention.end_char)
                    coref_mention_map[mention_span] = main_entity
                    logger.info(f"Coreference: Mapped '{mention.text}' to entity '{main_entity.name}'")

        source_entity = None
        actions_to_perform = []
        
        for match in results["dependency_matches"]:
            if match.get("pattern_name") == "EVIDENCE_SOURCE":
                source_token: Token = match.get("tokens", {}).get("source")
                if source_token:
                    source_entity = self.ENTITY_RESOLVER.find_entity_for_phrase(
                        {"text": source_token.text, "span": (source_token.idx, source_token.idx + len(source_token.text))},
                        resolved_entities
                    )
                break 

        for match in results["dependency_matches"]:
            if match.get("pattern_name") == "EVIDENCE_SOURCE":
                continue

            action = self.FACT_EXTRACTOR.process_dependency_match(
                match,
                resolved_entities,
                coref_mention_map,
                msg,
                source_entity=source_entity
            )
            if action:
                actions_to_perform.append(action)

        self.flag_and_apply(actions_to_perform, msg, threshold=0.75)   
        