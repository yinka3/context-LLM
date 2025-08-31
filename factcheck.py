import logging
import logging_setup
from typing import Dict, Optional
from dtypes import AttributeData, EntityData, MessageData
from spacy.tokens import Token

logging_setup.setup_logging()
logger = logging.getLogger(__name__)


class FactExtractor:

    def __init__(self):
        pass

    def _get_binary_amod_confidence(self, modifier_token: Token) -> float:
        modifier_lemma = modifier_token.lemma_.lower()
        
        high_confidence_adjectives = {
            "main", "primary", "final", "initial", "current", "next", "previous",
            "advanced", "introductory", "fundamental", "basic", "essential", 
            "required", "optional", "mandatory", "additional", "extra",
            
            "difficult", "challenging", "easy", "complex", "simple", 
            "important", "significant", "major", "minor", "key",
            "detailed", "comprehensive", "brief", "long", "short",
            
            "new", "old", "recent", "upcoming", "past", "future",
            "large", "small", "big", "huge", "tiny"
        }
        
        is_negated = any(child.dep_ == "neg" for child in modifier_token.children)
        
        if modifier_lemma in high_confidence_adjectives and not is_negated:
            return 0.85
        
        return 0.75

    def _find_entity_for_token(self, token: Token, resolved_entities: Dict[tuple, EntityData],
                                coref_mention_map: Dict[tuple, EntityData]) -> EntityData:
  
        for span, entity in resolved_entities.items():
            start_char, end_char = span
            if token.idx >= start_char and (token.idx + len(token.text)) <= end_char:
                return entity
        
        token_span = (token.idx, token.idx + len(token.text))
        if token_span in coref_mention_map:
            return coref_mention_map[token_span]
        
        for sub_token in token.subtree:
             for span, entity in resolved_entities.items():
                start_char, end_char = span
                if sub_token.idx >= start_char and (sub_token.idx + len(sub_token.text)) <= end_char:
                    return entity
        return None

    def _create_fact_action(self, subject_entity: EntityData, relation: str, value_entity: EntityData, 
                            msg: MessageData, source_entity: Optional[EntityData] = None, confidence: float = 0.85) -> Dict:
        if not all([subject_entity, relation, value_entity, msg]):
            return None
            
        action = {
            "action": "add_fact",
            "subject_id": f"ent_{subject_entity.id}",
            "relation": relation,
            "new_fact": AttributeData(
                value=value_entity, message=msg, confidence_score=confidence
            )
        }

        if source_entity:
            action["source_entity_id"] = f"ent_{source_entity.id}"
            
        return action

    def _create_attribute_action(self, subject_entity: EntityData, attribute_key: str, value: str, 
                                 msg: MessageData, source_entity: Optional[EntityData] = None, confidence: float = 0.85) -> Dict:
        if not all([subject_entity, attribute_key, value, msg]):
            return None

        action = {
            "action": "add_attribute",
            "subject_id": f"ent_{subject_entity.id}",
            "attribute_key": attribute_key,
            "new_attribute": AttributeData(
                value=value, message=msg, confidence_score=confidence
            )
        }

        if source_entity:
            action["source_entity_id"] = f"ent_{source_entity.id}"
            
        return action
    
    def _handle_svo_preposition_fact(self, matched_tokens: Dict[str, Token], 
                                     resolved_entities: Dict, msg: MessageData,
                                     coref_mention_map: Dict[tuple, EntityData],
                                     source_entity: Optional[EntityData]) -> Dict:
        
        subject_token = next(iter(t for name, t in matched_tokens.items() if 'person' in name or 'actor' in name or 'entity' in name), None)
        verb_token = matched_tokens.get("verb")
        prep_token = next(iter(t for name, t in matched_tokens.items() if 'prep' in name), None)
        object_token = next(iter(t for name, t in matched_tokens.items() if 'location' in name or 'topic' in name or 'attendee' in name), None)

        subject_entity = self._find_entity_for_token(subject_token, resolved_entities, coref_mention_map)
        object_entity = self._find_entity_for_token(object_token, resolved_entities, coref_mention_map)

        if subject_entity and object_entity and verb_token and prep_token:
            relation = f"{verb_token.lemma_}_{prep_token.lower_}"
            return self._create_fact_action(subject_entity, relation, 
                                            object_entity, msg, source_entity, confidence=0.8)
        return None

    def _handle_svo_direct_fact(self, matched_tokens: Dict[str, Token], 
                                resolved_entities: Dict, msg: MessageData,
                                coref_mention_map: Dict[tuple, EntityData],
                                source_entity: Optional[EntityData]) -> Dict:
        subject_token = matched_tokens.get("student")
        verb_token = matched_tokens.get("verb")
        object_token = matched_tokens.get("course")
        
        subject_entity = self._find_entity_for_token(subject_token, resolved_entities, coref_mention_map)
        object_entity = self._find_entity_for_token(object_token, resolved_entities, coref_mention_map)
        
        if subject_entity and object_entity and verb_token:
            return self._create_fact_action(subject_entity, verb_token.lemma_, 
                                            object_entity, msg, source_entity,
                                            confidence=0.9)
        return None
        
    def _handle_sva_attribute(self, matched_tokens: Dict[str, Token], 
                              resolved_entities: Dict, msg: MessageData,
                              coref_mention_map: Dict[tuple, EntityData], 
                              source_entity: Optional[EntityData]) -> Dict:
        subject_token = matched_tokens.get("subject")
        attribute_token = matched_tokens.get("evaluation")

        subject_entity = self._find_entity_for_token(subject_token, resolved_entities, coref_mention_map)
        if subject_entity and attribute_token:
            verb_lemma = matched_tokens.get("verb", subject_token.head).lemma_
            return self._create_attribute_action(subject_entity, f"is_{verb_lemma}", 
                                                 attribute_token.text, msg, source_entity)
        return None

    def _handle_sv_clause_attribute(self, matched_tokens: Dict[str, Token], 
                                    resolved_entities: Dict, msg: MessageData, 
                                    coref_mention_map: Dict[tuple, EntityData],
                                    source_entity: Optional[EntityData]) -> Dict:
        
        actor_token = matched_tokens.get("author")
        verb_token = matched_tokens.get("verb")
        content_token = matched_tokens.get("content")

        actor_entity = self._find_entity_for_token(actor_token, resolved_entities, coref_mention_map)
        if actor_entity and verb_token and content_token:
            return self._create_attribute_action(actor_entity, verb_token.lemma_, 
                                                 content_token.sent.text, msg, source_entity, confidence=0.8)
        return None

    def _handle_passive_fact(self, matched_tokens: Dict[str, Token], resolved_entities: Dict,
                             coref_mention_map: Dict[tuple, EntityData],
                             msg: MessageData, source_entity: Optional[EntityData]) -> Dict:
        # For passive, the grammatical object is the semantic subject, and vice versa.
        subject_of_action = self._find_entity_for_token(matched_tokens.get("subject"), resolved_entities, coref_mention_map)
        object_of_action = self._find_entity_for_token(matched_tokens.get("object"), resolved_entities, coref_mention_map)
        verb_token = matched_tokens.get("verb")
        
        if subject_of_action and object_of_action and verb_token:
            return self._create_fact_action(subject_of_action, verb_token.lemma_, 
                                            object_of_action, msg, source_entity, confidence=0.8)
        return None

    def _handle_assignment_deadline(self, matched_tokens: Dict[str, Token], resolved_entities: Dict,
                                    coref_mention_map: Dict[tuple, EntityData],
                                    msg: MessageData, source_entity: Optional[EntityData]) -> Dict:
        
        assignment_token = matched_tokens.get("assignment")
        date_token = matched_tokens.get("date")

        assignment_entity = self._find_entity_for_token(assignment_token, resolved_entities, coref_mention_map)
        if assignment_entity and date_token:
            return self._create_attribute_action(assignment_entity, "has_due_date", 
                                                 date_token.text, msg, source_entity, confidence=0.95)
        return None

    def _handle_task_obligation(self, matched_tokens: Dict[str, Token], resolved_entities: Dict, 
                                coref_mention_map: Dict[tuple, EntityData],
                                msg: MessageData, source_entity: Optional[EntityData]) -> Dict:
        actor_token = matched_tokens.get("actor")
        task_token = matched_tokens.get("task_verb")

        actor_entity = self._find_entity_for_token(actor_token, resolved_entities, coref_mention_map)
        if actor_entity and task_token:
            value = " ".join(t.text for t in task_token.subtree)
            return self._create_attribute_action(actor_entity, "has_obligation", 
                                                 value, msg, source_entity, confidence=0.9)
        return None

    def _handle_appositive_relationship(self, matched_tokens: Dict[str, Token], resolved_entities: Dict, 
                                        coref_mention_map: Dict[tuple, EntityData],
                                        msg: MessageData, source_entity: Optional[EntityData]) -> Dict:
        person_token = matched_tokens.get("person")
        role_token = matched_tokens.get("role")

        person_entity = self._find_entity_for_token(person_token, resolved_entities, coref_mention_map)
        if person_entity and role_token:
            return self._create_attribute_action(person_entity, "is_a", 
                                                 role_token.text, msg, source_entity)
        return None
    
    def _handle_adjectival_modifier(self, matched_tokens: Dict[str, Token], resolved_entities: Dict, 
                                    coref_mention_map: Dict[tuple, EntityData],
                                    msg: MessageData, source_entity: Optional[EntityData]) -> Dict:
        subject_token = matched_tokens.get("subject")
        modifier_token = matched_tokens.get("modifier")

        subject_entity = self._find_entity_for_token(subject_token, resolved_entities, coref_mention_map)
        if subject_entity and modifier_token:
            confidence = self._get_binary_amod_confidence(modifier_token)
            return self._create_attribute_action(subject_entity, "has_property", modifier_token.text, msg, source_entity, confidence=confidence)
        return None

    def _handle_possessive_relationship(self, matched_tokens: Dict[str, Token], resolved_entities: Dict, 
                                        coref_mention_map: Dict[tuple, EntityData],
                                        msg: MessageData, source_entity: Optional[EntityData]) -> Dict:
        owner_token = matched_tokens.get("owner")
        owned_token = matched_tokens.get("owned_entity")

        owner_entity = self._find_entity_for_token(owner_token, resolved_entities, coref_mention_map)
        owned_entity = self._find_entity_for_token(owned_token, resolved_entities, coref_mention_map)

        if owner_entity and owned_entity:
            return self._create_fact_action(owner_entity, "has_possession_of", owned_entity, msg, source_entity, confidence=0.9)
        return None

    def process_dependency_match(self, match: Dict, resolved_entities: Dict,
                                coref_mention_map: Dict[tuple, EntityData], 
                                msg: MessageData, source_entity: Optional[EntityData]):

        pattern_name = match.get("pattern_name")
        matched_tokens = match.get("tokens", {})
        
        handler_map = {
            "ARGUMENT_CLAIM": self._handle_sv_clause_attribute,
            "CONCEPT_DEFINITION": self._handle_passive_fact,
            "COURSE_TAKEN": self._handle_svo_direct_fact,
            "ASSIGNMENT_DEADLINE": self._handle_assignment_deadline,
            "TASK_OBLIGATION": self._handle_task_obligation,

            "SOCIAL_RELATIONSHIP_APPOS": self._handle_appositive_relationship,
            "FEELING_ABOUT_TOPIC": self._handle_svo_preposition_fact,
            "SOCIAL_ACTIVITY": self._handle_svo_preposition_fact,
            "EVALUATIVE_STATEMENT": self._handle_sva_attribute,

            "ENTITY_IS_LOCATED_IN": self._handle_svo_preposition_fact,
            "PASSIVE_RELATIONSHIP": self._handle_passive_fact,
            "ADJECTIVAL_MODIFIER_RELATIONSHIP": self._handle_adjectival_modifier,
            "POSSESSIVE_RELATIONSHIP": self._handle_possessive_relationship
        }

        handler = handler_map.get(pattern_name)

        if handler:
            try:
                return handler(matched_tokens, resolved_entities, coref_mention_map, msg, source_entity)
            except Exception as e:
                logger.error(f"Error processing pattern '{pattern_name}': {e}")
                return None
        else:
            logger.warning(f"No handler found for pattern: {pattern_name}")
        
        return None
