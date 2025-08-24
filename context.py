import logging
import json
import os
from typing import Dict, List, Set, Union
from networkx import DiGraph, Graph
from nlp_pipe import NLP_PIPE
from dtypes import AttributeData, EntityData, MessageData
from vectordb import ChromaClient

class Context:

    def __init__(self):
        self.next_id = 0
        self.graph: DiGraph = DiGraph()
        self.nlp_pipe: NLP_PIPE = NLP_PIPE()
        self.history: List[Dict[MessageData, EntityData]] = []
        self.short_context: Set[EntityData] = {}
        self.chroma: ChromaClient = ChromaClient()
        self.user_message_cnt: int = 0

        self.vocab = {}
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            vocab_path = os.path.join(script_dir, "vocab.json")
            with open(vocab_path) as file:
                self.vocab = json.load(file)
        except FileNotFoundError:
            logging.error("Error: The file was not found.")
            raise
        except json.JSONDecodeError:
            logging.error("Error: The file is not a valid JSON.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def get_new_id(self):
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def check_chroma(self, text: str, threshold: float = 0.3):
        results = self.chroma.collection.query(
                    query_texts=[text],
                    n_results=1,
                    where={"node_type": "entity"})

        if results['ids'][0] and results['distances'][0][0] < threshold:
            return int(results['ids'][0][0])
        
        return None
    
    def extract_svo(self, svo: Dict):

        verb = svo["verb"]

        rules = self.vocab.get(verb)
        preposition_phrase = svo.get("prepositional_phrases")

        if verb not in self.vocab:
            return f"{verb}_expresses_about"
        
        if preposition_phrase:
            first = preposition_phrase[0].split(' ', 1)[0]

            if first in rules["prepositions"]:
                return rules["prepositions"][first]
        
        return rules["default_relation"]


    def process_live_messages(self, msg: MessageData):
        if self.nlp_pipe.type == "long":
            logging.info("This is only for live stream messages, switching to short type.")
            self.nlp_pipe.type = "short"
        
        msg_id = f"msg_{msg.id}"
        results = self.nlp_pipe.start_process(msg=msg)
        msg.sentiment = results["sentiment"]
        self.graph.add_node(msg_id, data=msg, type="message")
        resolved_entities = {}
        for ents in results["found_entities"]:
            word = ents["text"]
            found_id = self.check_chroma(text=word)
            if found_id is not None:
                ent_id = f"ent_{found_id}"
                if ent_id in self.graph.nodes and self.graph.nodes[ent_id]["type"] == "entity":
                    existing_ent: EntityData = self.graph.nodes[ent_id]['data']
                    if msg.id not in existing_ent.mentioned_in:
                        existing_ent.mentioned_in.append(msg.id)
                    resolved_entities[word] = existing_ent

                    self.graph.add_edge(msg_id, ent_id, type="mentions")
            else:
                new_ent = EntityData(
                    id=self.get_new_id(),
                    name=word,
                    type=ents["type"])
                
                ent_id = f"ent_{new_ent.id}"
                self.graph.add_node(ent_id, data=new_ent, type="entity")
                self.chroma.add_item(new_ent)
                resolved_entities[word] = new_ent
                self.graph.add_edge(msg_id, ent_id, type="mentions")


        for svo in results["SVO"]:
            subject_text = svo["subject"]

            if subject_text in resolved_entities:
                subject_entity = resolved_entities[subject_text]
                object_text  = svo.get("object")

                if object_text and object_text  in resolved_entities:

                    relations = self.extract_svo(svo)
                    object_entity = resolved_entities[object_text ]

                    new_fact = AttributeData(
                        value=object_entity,
                        message=msg,
                        confidence_score=0.95)

                    if relations not in subject_entity.attributes:
                        subject_entity.attributes[relations] = []
                    
                    subject_entity.attributes[relations].append(new_fact)
                    logging.info(f"Created Link: ({subject_entity.name}) -> [{relations}] -> ({object_entity.name})")

                if "prepositional_phrases" in svo:
                    for preposition in svo["prepositional_phrases"]:

                        try:
                            _, prep_object_text = preposition.split(' ', 1)
                        except ValueError:
                            continue

                        found_object: EntityData = None

                        for entity_name, entity_object in resolved_entities.items():
                            if entity_name in prep_object_text:
                                found_object = entity_object
                                break

                        if found_object:
                            temp_svo = {"verb": svo["verb"], "prepositional_phrases": [preposition]}
                            relations = self.extract_svo(temp_svo)

                            new_fact = AttributeData(
                                value=found_object,
                                message=msg,
                                confidence_score=0.95)

                            if relations not in subject_entity.attributes:
                                subject_entity.attributes[relations] = []
                            subject_entity.attributes[relations].append(new_fact)
                            logging.info(f"Created Link (Prep): ({subject_entity.name}) -> [{relations}] -> ({found_object.name})")
                        

    def add(self, item: Union[str, EntityData]):
        
        if isinstance(item, MessageData):
            self.process_live_messages(item)
        


        
