


import logging
from typing import Dict, List

from dtypes import AttributeData, EntityData, MessageData


class FactExtractor:

    def __init__(self, vocab: Dict):
        self.vocab = vocab

    def extract_svo(self, svo: Dict):

        verb = svo["verb"]

        rules: Dict = self.vocab.get(verb)
        preposition_phrases: List[Dict[str, str]] = svo.get("prepositional_phrases")
        if rules:
            if preposition_phrases:
                preposition = preposition_phrases[0]['text'].split(' ', 1)[0]
                if preposition in rules.get("prepositions", {}):
                    return rules["prepositions"][preposition]
            
            return rules.get("default_relation", verb)
        else:
            if preposition_phrases:
                preposition = preposition_phrases[0]['text'].split(' ', 1)[0]

                logging.info(f"Could not find '{verb}' in vocab, defaulting to heuristic: {verb}_{preposition}")
                return f"{verb}_{preposition}"
            else:
                logging.info(f"Could not find '{verb}' in vocab, defaulting to verb name.")
                return verb 
        

    def calculate_relationship_confidence(self, subject_conf: float, object_conf: float, svo_clarity: float):
            if subject_conf > 0.7 and object_conf > 0.7:
                return 0.9
            elif subject_conf < 0.5 or object_conf < 0.5:
                return 0.5
            else:
                return (subject_conf + object_conf) / 2 * svo_clarity
    
    
    
    def process_svo(self, svo, subject_entity: EntityData, object_entity: EntityData, msg):
        
        if subject_entity and object_entity:
            relationship_confidence = self.calculate_relationship_confidence(
                subject_entity.confidence,
                object_entity.confidence,
                svo_clarity=1.0
            )
            
            relation = self.extract_svo(svo)
            
            if relationship_confidence > 0.7:
                new_fact = AttributeData(
                    value=object_entity,
                    message=msg,
                    confidence_score=relationship_confidence
                )
                return {
                    "action": "add_fact",
                    "subject_id": f"ent_{subject_entity.id}",
                    "relation": relation,
                    "new_fact": new_fact
                }
        return None
    
    
    def process_attribute(self, attribute_data: Dict, subject_entity: EntityData, msg: MessageData):

        relation = attribute_data['relation']
        value = attribute_data['value']['text']
        
        new_fact = AttributeData(
            value=value,
            message=msg,
            confidence_score=subject_entity.confidence
        )

        attribute_key = f"is_{relation}"

        return {
            "action": "add_attribute",
            "subject_id": f"ent_{subject_entity.id}",
            "attribute_key": attribute_key,
            "new_attribute": new_fact
        }