import json
import math
import os
import spacy
import torch
from gliner import GLiNER
from spacy.tokens import Doc, Token, SpanGroup
from dtypes import MessageData
from typing import Any, Dict, List, Union
import logging
from transformers import pipeline
from spacy.matcher import DependencyMatcher
from fastcoref import spacy_component


logger = logging.getLogger(__name__)

class NLP_PIPE:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.spacy = spacy.load('en_core_web_trf', exclude=["ner"])
            self.spacy.add_pipe("fastcoref")
            logger.info("spaCy transformer pipeline with fastcoref loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load spaCy model or fastcoref: {e}")
            raise
            
        self.CANONICAL_LABELS = {
            "PERSON", "COLLEGE_COURSE", "ASSIGNMENT", "SOCIAL_TOPIC", "CONCEPT",
            "DEPARTMENT", "COLLEGE_DEGREE", "ORGANIZATION", "TEAM", "PROJECT",
            "PRODUCT", "FEATURE", "INITIATIVE", "TECHNOLOGY", "FRAMEWORK",
            "EVENT", "HOBBY", "LOCATION", "DATE", "TIME", "SKILL",
            "FIELD_OF_STUDY", "FEELING", "STRESSOR"
        }
        
        self.gliner_labels = [label.replace("_", " ").lower() for label in self.CANONICAL_LABELS]
        self.label_map = {gl_label: can_label for gl_label, can_label in zip(self.gliner_labels, self.CANONICAL_LABELS)}
        logger.info(f"Entity labels configured for {len(self.CANONICAL_LABELS)} types.")

        try:
            if torch.cuda.is_available():
                self.gliner = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
                self.gliner = self.gliner.to(self.device)
                self.gliner.model.half().eval()
                logger.info(f"GLiNER loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                self.gliner = GLiNER.from_pretrained("urchade/gliner_base")
                logger.info("GLiNER (base) loaded on CPU.")
        except Exception as e:
            logger.error(f"GLiNER initialization failed: {e}")
            raise

        try:
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            self.emotion_classifier = pipeline("text-classification", 
                                               model=model_name, 
                                               return_all_scores=True,
                                               device=0 if torch.cuda.is_available() else -1)
            logger.info(f"Emotion classifier ({model_name}) loaded successfully.")
        except Exception as e:
            logger.error(f"Emotion classifier could not be initialized: {e}")
            raise


    def analyze_emotion(self, text: str) -> List[Dict]:
        """
        Analyzes the text and returns a list of emotions with their scores.
        """
        if not text.strip():
            return []
        
        results = self.emotion_classifier(text)
        return results[0]
        

    def deduplicate_entities(self, entities):
        if not entities:
            return []
        
        entities = sorted(entities, key=lambda x: (x['start'], -len(x['text'])))
        
        filtered = []
        last_end = -1
        
        for ent in entities:
            if ent['start'] >= last_end:
                filtered.append(ent)
                last_end = ent['end']

        return filtered
    
    def _build_verb_tree(self, verb_token: Token) -> Dict:
        """
        Recursively builds a nested dictionary for a verb and its related actions.
        """
        # Base structure for the current verb
        verb_data = {
            "text": verb_token.text,
            "lemma": verb_token.lemma_,
            "relation": verb_token.dep_,
            "sub_actions": [],
            "conjoined_actions": []
        }

        # Find children that are also verbs and part of a related action
        for child in verb_token.children:
            if child.pos_ == "VERB":
                # Verbs in sub-clauses (e.g., "decided TO RUN")
                if child.dep_ in ("xcomp", "ccomp"):
                    verb_data["sub_actions"].append(self._build_verb_tree(child))
                # Conjoined verbs (e.g., "ran AND JUMPED")
                elif child.dep_ == "conj":
                    verb_data["conjoined_actions"].append(self._build_verb_tree(child))

        return verb_data

    
    def start_process(self, msg: MessageData, 
                      entity_threshold: float = 0.7):
        res: Dict[str, List] = {
            "high_confidence_entities": [],
            "low_confidence_entities": [],
            "noun_chunks": [],
            "coref_clusters": [],
            "emotion": [],
            "tier2_flags": [],
            "verbs": [],
            "adjectives": []
            }
        
        
        if not msg.message or not msg.message.strip():
            return res
        
        msg_length = len(msg.message.strip())
        if msg_length > 800:
            payload = {
                "original_text": msg.message,
                "reason_details": f"Message length is {len(msg.message)} characters."
            }
            res["tier2_flags"].append({
                "reason": "MESSAGE_TOO_LONG",
                "priority": 4,
                "payload": payload
            })
            logger.warning(f"Flagged for Tier 2: Message too long: {len(msg.message)} chars | Payload Info: {payload} \n")
            return res # we will now ship this off to tier 2 also
        
        entities = self.gliner.predict_entities(text=msg.message, labels=self.gliner_labels)
        entities = self.deduplicate_entities(entities)

        for ent in entities:
            ent_data = {
                "text": ent["text"], "type": self.label_map.get(ent["label"], "MISC"),
                "confidence": ent['score'], "span": (ent["start"], ent["end"])
            }

            if ent['score'] >= entity_threshold:
                res['high_confidence_entities'].append(ent_data)
            else:
                res['low_confidence_entities'].append(ent_data)
        

        res["emotion"] = self.analyze_emotion(msg.message)
        doc: Doc = self.spacy(text=msg.message)

        coref_clusters = []
        if doc._.coref_clusters:
            for cluster_spans in doc._.coref_clusters:
                if len(cluster_spans) < 2:
                    continue
                
                main_span = doc[cluster_spans[0][0]:cluster_spans[0][1]]
                mention_spans = [doc[span[0]:span[1]] for span in cluster_spans]

                cluster_obj = type('CoreferenceCluster', (), {
                    'main': main_span,
                    'mentions': mention_spans,
                    'size': len(cluster_spans)
                })()
                
                coref_clusters.append(cluster_obj)

        res["coref_clusters"] = coref_clusters

        high_conf_entities = res["high_confidence_entities"]
        noun_chunks = [(chunk.text, chunk.start_char, chunk.end_char) for chunk in doc.noun_chunks]
        for i, entity in enumerate(high_conf_entities):
            ent_start, ent_end = entity["span"]
            
            for chunk_text, chunk_start, chunk_end in noun_chunks:
                if ent_start >= chunk_start and ent_end <= chunk_end:
                    high_conf_entities[i]["text"] = chunk_text
                    high_conf_entities[i]["span"] = (chunk_start, chunk_end)
                    break 

        for sentence in doc.sents:
            root_verb = sentence.root
            # ensure the root of the sentence is a verb before processing
            if root_verb.pos_ == "VERB":
                verb_tree = self._build_verb_tree(root_verb)
                res["verbs"].append(verb_tree)

        for token in doc:
            if token.pos_ == "ADJ":
                    described_noun = None
                    if token.dep_ == "amod":
                        described_noun = token.head
                    elif token.dep_ == "acomp":
                        verb_head = token.head
                        subjects = [child for child in verb_head.children if child.dep_ in ("nsubj", "nsubjpass")]
                        if subjects:
                            described_noun = subjects[0]
                    elif token.dep_ == "oprd":
                        verb_head = token.head
                        objects = [child for child in verb_head.children if child.dep_ == "dobj"]
                        if objects:
                            described_noun = objects[0]

                    if described_noun:
                        res["adjectives"].append({
                            "text": token.text,
                            "lemma": token.lemma_,
                            "span": (token.idx, token.idx + len(token.text)),
                            "describes_text": described_noun.text,
                            "describes_lemma": described_noun.lemma_
                        })
        return res