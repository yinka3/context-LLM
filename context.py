from datetime import datetime
import logging
import logging_setup
import json
import os
from redisclient import RedisClient
from typing import Any, Dict, List, Optional
from networkx import DiGraph
from entity import EntityResolver
from nlp_pipe import NLP_PIPE
from dtypes import AttributeData, BridgeData, EdgeData, EntityData, MessageData
from vectordb import ChromaClient
from spacy.tokens import Token

logging_setup.setup_logging()

logger = logging.getLogger(__name__)

class Context:

    def __init__(self, user_name: str = "Yinka"):
        self.next_id = 1
        self.user_name = user_name
        self.graph: DiGraph = DiGraph()
        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        self.history: List[MessageData] = []
        self.entities: Dict[int, EntityData] = {}
        self.chroma: ChromaClient = ChromaClient()
        self.redis_client = RedisClient()
        self.user_message_cnt: int = 0
        self.bridge_map: Dict[str, Dict[Any, List[int]]] = {}

        self.user_entity = self._create_user_entity(user_name)

        self.ENTITY_RESOLVER = EntityResolver(
            graph=self.graph,
            chroma=self.chroma,
            user_entity=self.user_entity,
            next_id=self.get_new_id
        )

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

    def process_verbs(self, msg: MessageData, message_verbs: List, resolved_entites: Dict):
        pass

    def process_adjectives(self, msg: MessageData, message_adjectives: List):
        adj_only_index = self.bridge_map.setdefault("shared_adjective", {})
        adj_noun_index = self.bridge_map.setdefault("shared_adjective_concept", {})

        for adj_data in message_adjectives:
            adj_lemma = adj_data["lemma"]
            noun_lemma = adj_data.get("describes_lemma")

            if adj_lemma in adj_only_index:
                for prev_msg_id in adj_only_index[adj_lemma]:
                    if msg.id == prev_msg_id: continue
                    
                    bridge = BridgeData(type="shared_adjective", value=adj_lemma)
                    edge = EdgeData(messages=(msg.id, prev_msg_id), bridge=bridge)
                    self.graph.add_edge(f"msg_{msg.id}", f"msg_{prev_msg_id}", data=edge)
                    logger.info(f"Bridged msg {msg.id} and msg {prev_msg_id} via adjective: '{adj_lemma}'")
        
            if msg.id not in adj_only_index.setdefault(adj_lemma, []):
                adj_only_index[adj_lemma].append(msg.id)

            if noun_lemma:
                key = (adj_lemma, noun_lemma)
                
                if key in adj_noun_index:
                    for prev_msg_id in adj_noun_index[key]:
                        if msg.id == prev_msg_id: continue

                        bridge_value = f"{adj_lemma} {noun_lemma}"
                        bridge = BridgeData(type="shared_adjective_concept", value=bridge_value)
                        edge = EdgeData(messages=(msg.id, prev_msg_id), bridge=bridge)
                        
                        self.graph.add_edge(f"msg_{msg.id}", f"msg_{prev_msg_id}", data=edge)
                        logger.info(f"Bridged msg {msg.id} and msg {prev_msg_id} via concept: '{bridge_value}'")

                if msg.id not in adj_noun_index.setdefault(key, []):
                    adj_noun_index[key].append(msg.id)

    def process_coref_clusters(self, msg: MessageData, message_corefs: List, coref_mention_map: Dict, resolved_entities: Dict):
        for cluster in message_corefs:
                
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
                    if mention is cluster.main or mention.text == '':
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

                    if mention.text in main_entity.name:
                        continue
                    
                    mention_span = (mention.start_char, mention.end_char)
                    coref_mention_map[mention_span] = main_entity
                    logger.info(f"Coreference: Mapped '{mention.text}' to entity '{main_entity.name}'")    


    def process_live_messages(self, msg: MessageData):
        
        msg_id = f"msg_{msg.id}"
        self.graph.add_node(msg_id, data=msg, type="message")

        results = self.nlp_pipe.start_process(msg=msg, entity_threshold=0.7)

        self.redis_client.client.lpush(f'conversation:{self.user_name}', json.dumps({
            'message': msg.message,
            'timestamp': msg.timestamp.isoformat(),
            'role': msg.role
        }))


        self.redis_client.client.ltrim(f'conversation:{self.user_name}', 0, 49) # might removed this but enforce through UI to keep a limit for a conversation session.
        self.redis_client.client.expire(f'conversation:{self.user_name}', 86400)


        if results.get("tier2_flags") != []:
            for flag in results["tier2_flags"]:
                logger.warning(f"Received Tier 2 flag from NLP_PIPE. Reason: {flag['reason']}")
            return
        
        if results.get("high_confidence_entites") == []:
            logging.info("Need entities")
            return
        
        primary_emotion = max(results["emotion"], key=lambda x: x['score'])
        msg.sentiment = primary_emotion['label']

        entities = results["high_confidence_entities"]        
        resolved_entities = {}
        for ents in entities:
            entity, tier2_payload = self.ENTITY_RESOLVER.process_entity(ents, msg_id)
            
            if tier2_payload:
                conflicting_entity_name = tier2_payload.get('candidates', [{}])[0].get('name', 'N/A')

                logger.warning(f"""Flagged for Tier 2: Potential merge conflict between new entity 
                                '{tier2_payload['new_entity']['name']}' and existing entity '{conflicting_entity_name}'.
                                Message: {msg.message} | Payload Info: {tier2_payload} \n""")

            if entity:
                resolved_entities[ents["span"]] = entity
                if entity.id not in self.entities:
                    self.entities[entity.id] = entity
        
        logger.info("Starting noun chunk candidate analysis...")

        for ent_span, entity in resolved_entities.items():
            ent_start, ent_end = ent_span

            for chunk in results["noun_chunks"]:
                chunk_start, chunk_end = chunk["span"]

                if ent_start >= chunk_start and ent_end <= chunk_end:
                    new_alias_text = chunk["text"]
                    is_main_name = entity.name.lower() == new_alias_text.lower()
                    is_existing_alias = any(alias["text"].lower() == new_alias_text.lower() for alias in entity.aliases)

                    if not is_main_name and not is_existing_alias:
                        entity.aliases.append({
                            "text": new_alias_text,
                            "type": "noun_chunk_alias"
                        })
                        logger.info(f"Enriched entity '{entity.name}' with new alias from noun chunk: '{new_alias_text}'")
                        break 

        
        if results.get("coref_clusters") != []:
            self.process_coref_clusters(msg=msg, message_corefs=results["coref_clusters"])
        
        if results.get("verbs") != []:
            self.process_verbs(msg=msg, message_verbs=results["verbs"])
        
        if results.get("adjectives") != []:
            self.process_adjectives(msg=msg, message_adjectives=results["adjectives"])






    # def flag_and_apply(self, actions: List[Dict], msg: MessageData, threshold: 0.75):
    #     """
    #     Applies high-confidence actions directly and flags low-confidence ones for Tier 2.
    #     """
    #     for action in actions:
    #         if not action:
    #             continue

    #         confidence = action["new_fact" if "new_fact" in action else "new_attribute"].confidence_score

    #         if confidence >= threshold:
    #             self._apply_state_action(action)
    #         else:
    #             subject_id = action.get("subject_id")
    #             subject_entity: EntityData = self.graph.nodes[subject_id]["data"]

    #             payload = {
    #                 "text": msg.message,
    #                 "action_details": action,
    #                 "entity_name": subject_entity.name
    #             }

    #             # a function that sends the payload dict off to tier 2, in that function we will create a Tier2Task data obj
    #             logger.warning(f"""Flagged for Tier 2: 
    #                             Low confidence fact ({confidence:.2f}) for entity '{subject_entity.name}'. 
    #                             Action not applied for message: {msg.message}. | Payload Info: {payload}""")

    # def _apply_state_action(self, action: Dict):
    #     """Applies a state change returned by the FactExtractor."""
    #     subject_id_str = action.get("subject_id")
    #     if not subject_id_str or not self.graph.has_node(subject_id_str):
    #         return
        
    #     subject_entity: EntityData = self.graph.nodes[subject_id_str]['data']

    #     if action["action"] == "add_fact":
    #         new_fact: AttributeData = action["new_fact"]
    #         target_entity: EntityData = new_fact.value
    #         target_id_str = f"ent_{target_entity.id}"

    #         if not self.graph.has_node(target_id_str):
    #             logger.error(f"APPLY FACT FAILED: Target entity '{target_entity.name}' (ID: {target_id_str}) was not found in the graph before creating edge.")
    #             return

    #         if not self.graph.has_node(target_id_str): return

    #         edge_attributes = {
    #             "relation": action["relation"],
    #             "confidence": new_fact.confidence_score,
    #             "message_id": f"msg_{new_fact.message.id}"
    #         }

    #         if "source_entity_id" in action:
    #             edge_attributes["source_id"] = action["source_entity_id"]
            
    #         self.graph.add_edge(subject_id_str, target_id_str, **edge_attributes)
    #         logger.info(f"Applied Fact Edge: ({subject_entity.name}) -> [{action['relation']}] -> ({target_entity.name})")

    #     elif action["action"] == "add_attribute":
    #         key = action["attribute_key"]
    #         new_attribute = action["new_attribute"]
    #         if key not in subject_entity.attributes:
    #             subject_entity.attributes[key] = []
    #         subject_entity.attributes[key].append(new_attribute)

    #         if new_attribute.value.lower() in subject_entity.name.lower():
    #             logger.info(f"Attribute is within subject entity")
    #             return

    #         logger.info(f"Applied Attribute: ({subject_entity.name}) -> [{key}] -> ({new_attribute.value})")

    