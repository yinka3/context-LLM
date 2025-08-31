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
import logging_setup
from transformers import pipeline
from spacy.matcher import DependencyMatcher
from fastcoref import spacy_component

logging_setup.setup_logging()

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
            
        self.entity_labels = ["person", "course", "assignment", "topic", "concept", "degree", "department",      
                            "organization", "team", "project", "product", "feature", "initiative", "technology",
                            "event", "hobby", "location", "date", "time"]

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

        self.dependency_matcher = DependencyMatcher(self.spacy.vocab, validate=True)
        self.dependency_patterns = {}
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pattern_path = os.path.join(script_dir, "matcher.json")
            with open(pattern_path) as f:
                self.dependency_patterns = json.load(f)
            
            for pattern_name, pattern_list in self.dependency_patterns.items():
                self.dependency_matcher.add(pattern_name, pattern_list)
            
            logger.info(f"Successfully loaded and added {len(self.dependency_patterns)} patterns.")

        except Exception as e:
            logger.error(f"Error loading or adding matcher patterns: {e}")
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
    
    def non_projective_deps(self, doc: Doc):
        crossing_pairs = set()
        tokens = list(doc)
        for i, token in enumerate(tokens):
            if token.head == token: continue
            for j, other in enumerate(tokens[i+1:], i+1):
                if other.head == other or i == j: continue

                arc1 = sorted([i, token.head.i])
                arc2 = sorted([j, other.head.i])
            
                if (arc1[0] < arc2[0] < arc1[1] < arc2[1] or 
                    arc2[0] < arc1[0] < arc2[1] < arc1[1]):
                    crossing_pairs.add(tuple(sorted([i, j])))

        return len(crossing_pairs)
    
    def calculate_long_distance_deps(self, doc: Doc):
        long_distance_count = 0
        threshold = 7
        
        for token in doc:
            if token.head != token:
                distance = abs(token.i - token.head.i)
                if distance >= threshold:
                    long_distance_count += 1
        return long_distance_count
    
    def calculate_density_factor(self, doc: Doc):
        total_deps = sum(1 for token in doc if token.head != token)
        total_tokens = len([t for t in doc if not t.is_space])
        return total_deps / total_tokens if total_tokens > 0 else 0
    
    def max_depth(self, root_token: Token):

        if not len(list(root_token.children)):
            return 1
        
        return max(self.max_depth(child) for child in root_token.children) + 1
        

    def calculate_complexity_score(self, doc: Doc, 
                                   high_conf_count: int, low_conf_count: int,
                                   coref_count: int, dep_match_count: int):
        sub_clauses = 0
        max_depth = 0
        conjunctions = 0

        subordinate_deps = {"mark", "csubj", "acl", "acl:relcl", "advcl", "ccomp", "xcomp"}

        w_uncertain_entities = 1.0 
        w_sub_clauses = 2.0
        w_depth = 1.0
        w_conjunctions = 0.75
        w_coref_clusters = 1.25
        w_dependency_matches = 0.25
        w_long_distance = 1.25
        w_non_projective = 2.5

        for sent in doc.sents:
            for token in sent:
                if token.dep_ in subordinate_deps:
                    sub_clauses += 1
                
                if token.dep_ == "cc":
                    conjunctions += 1

            root_token = sent.root
            max_depth = max(max_depth, self.max_depth(root_token))
        
        norm_depth = math.log(max_depth + 1) / math.log(2)
        uncertainty_score = low_conf_count - high_conf_count
        token_distance = self.calculate_long_distance_deps(doc=doc)
        density = self.calculate_density_factor(doc=doc)
        non_proj = self.non_projective_deps(doc=doc)
        score: float = (
            (w_non_projective * non_proj) +
            (w_long_distance * token_distance * density) +
            (w_uncertain_entities * uncertainty_score) +
            (w_sub_clauses * sub_clauses) +
            (w_depth * norm_depth) +
            (w_conjunctions * conjunctions) +
            (w_coref_clusters * coref_count) +
            (w_dependency_matches * dep_match_count)
        )

        result: Dict[str, Union[int, float]] = {
            "score": score,
            "sub_clauses": sub_clauses,
            "max_depth": norm_depth,
            "conjunctions": conjunctions,
            "high_conf_count": high_conf_count,
            "low_conf_count": low_conf_count,
            "coref_count": coref_count,
            "dep_match_count": dep_match_count,
            "non_projective": non_proj,
            "token_distances": token_distance
        }
        
        return result

    def get_live_complexity_threshold(self, msg_length):
        """Dynamic threshold based on message characteristics"""

        # Base thresholds for different message types
        if msg_length <= 20:
            return 3.0
        elif msg_length <= 50:
            return 9.0
        elif msg_length <= 150:
            return 12.0
        elif msg_length <= 300:
            return 18.0
        else:
            return 25.0
    

    def should_escalate_to_tier2(self, complexity_data, msg_length, msg_text):
        """Hard rules for chatbot complexity"""
        
        if complexity_data["non_projective"] >= 1:
            return True, "Non-projective dependencies detected"
        
        if complexity_data["token_distances"] >= 8:
            return True, "Long-distance dependencies too complex"
        
        if complexity_data["sub_clauses"] >= 4:
            return True, "Multiple nested clauses"
        
        if msg_length < 100 and complexity_data["score"] > 10:
            return True, "High complexity density - user may be struggling"
        
        threshold = self.get_live_complexity_threshold(msg_length)
        if complexity_data["score"] > threshold:
            return True, f"Complexity score {complexity_data['score']:.1f} > threshold {threshold}"
        
        return False, "Acceptable complexity"

    
    def start_process(self, msg: MessageData, 
                      entity_threshold: float = 0.7):
        res: Dict[str, List] = {
            "high_confidence_entities": [],
            "low_confidence_entities": [],
            "noun_chunks": [],
            "dependency_matches": [],
            "coref_clusters": [],
            "emotion": [],
            "tier2_flags": []
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
        
        entities = self.gliner.predict_entities(text=msg.message, labels=self.entity_labels)
        entities = self.deduplicate_entities(entities)

        for ent in entities:
            ent_data = {
                "text": ent["text"], "type": ent["label"],
                "confidence": ent['score'], "span": (ent["start"], ent["end"])
            }

            if ent['score'] >= entity_threshold:
                res['high_confidence_entities'].append(ent_data)
            else:
                res['low_confidence_entities'].append(ent_data)
        

        res["emotion"] = self.analyze_emotion(msg.message)
        doc: Doc = self.spacy(text=msg.message)

        high_conf_count = len(res['high_confidence_entities'])
        low_conf_count = len(res['low_confidence_entities'])

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
        matches = self.dependency_matcher(doc)
        for match_id, token_ids in matches:
            pattern_name = self.spacy.vocab.strings[match_id]
            matched_tokens = {}
            for i, token_id in enumerate(token_ids):
                try:
                    rule_item_name = self.dependency_patterns[pattern_name][0][i]["RIGHT_ID"]
                    matched_tokens[rule_item_name] = doc[token_id]
                except (KeyError, IndexError):
                    logger.warning(f"Could not find RIGHT_ID for a token in match '{pattern_name}'")

            res["dependency_matches"].append({
                "pattern_name": pattern_name,
                "tokens": matched_tokens
            })
        
        coref_count = len(res["coref_clusters"])
        dep_match_count = len(res["dependency_matches"])
        complexity_data = self.calculate_complexity_score(doc, high_conf_count, 
                                                          low_conf_count, coref_count, dep_match_count)
        
        should_escalate, reason = self.should_escalate_to_tier2(
            complexity_data, msg_length, msg.message)

        if should_escalate:
            payload = {
                "original_text": msg.message,
                "reason_details": complexity_data,
                "escalation_reason": reason,
                "message_length": msg_length,
                "complexity_score": complexity_data["score"]
            }
            res["tier2_flags"].append({
                "reason": "HIGH_SENTENCE_COMPLEXITY",
                "priority": 6,
                "payload": payload
            })
            logger.info(f"""Recommended Tier 2 review for high complexity: {complexity_data['score']:.2f}
                            Message: {msg.message} | Payload Info: {payload} \n""")

        for chunk in doc.noun_chunks:
            res["noun_chunks"].append({
                "text": chunk.text, "span": (chunk.start_char, chunk.end_char)
            })

        return res

        

        










