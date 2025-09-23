from collections import namedtuple
import json
import spacy
import torch
from datetime import datetime
from sutime import SUTime
from gliner import GLiNER
from spacy.tokens import Doc, Token
from shared.dtypes import MessageData
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import logging
from transformers import pipeline
from fastcoref import spacy_component


logger = logging.getLogger(__name__)

CoreferenceCluster = namedtuple('CoreferenceCluster', ['main', 'mentions', 'size'])

#I like to see what my results are
RES_LABEL = Literal["high_confidence_entities", "low_confidence_entities", 
                    "emotion", "time_expressions", "appositive_map",
                    "coref_clusters", "noun_chunk"]

class NLP_PIPE:

    PRONOUNS = {
        "i", "me", "my", "myself", "we", "our", "ourselves", "you", "your", 
        "yourself", "he", "him", "his", "himself", "she", "her", "herself", 
        "it", "itself", "they", "them", "their", "themselves", "this", "that"
    }


    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.spacy = spacy.load('en_core_web_trf', enable=["fastcoref", "doc_cleaner"], 
                                    exclude=["ner"])
            logger.info(f"spaCy transformer pipeline with these labels: {self.spacy.pipe_names}.")
        except Exception as e:
            logger.error(f"Could not load spaCy model or fastcoref: {e}")
            raise

        try:
            self.sutime = SUTime()
        except Exception as e:
            logger.error("Could not load SUTime library")
            raise
            
        self.CANONICAL_LABELS = {
            "PERSON",
            "POSSESSIVE_ENTITY",
            "GROUP_OF_ENTITIES",
            "ORGANIZATION",
            "LOCATION",
            "DATE",
            "TIME",
            "WORK_PRODUCT_OR_PROJECT",
            "ACADEMIC_CONCEPT",
            "TECHNOLOGY",
            "EVENT",
            "TOPIC"
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

    
    def analyze_time(self, text: str) -> List[Dict]:
        """
        Analyzes the text for temporal expressions.
        """

        reference_date = datetime.now().isoformat()
        time_expressions = self.sutime.parse(text, reference_date)
        return time_expressions

    def analyze_emotion(self, text: str):
        """
        Analyzes the text and returns a list of emotions with their scores.
        """
        if not text.strip():
            return []
        
        results = self.emotion_classifier(text)
        return results[0]
        

    def clean_entities(self, entities):
        if not entities:
            return []

        cleaned_entities = [
            ent for ent in entities if ent['text'].lower() not in self.PRONOUNS
        ]
        
        sorted_entities = sorted(cleaned_entities, key=lambda x: (x['start'], -len(x['text'])))
        filtered = []
        last_end = -1
        
        for ent in sorted_entities:
            if ent['start'] >= last_end:
                filtered.append(ent)
                last_end = ent['end']

        return filtered
    

    def _extract_message_features(self, msg: MessageData, entity_threshold: float, res: Dict):

        if not msg.message or not msg.message.strip():
            return res
        
        entities = None
        with torch.no_grad():
            entities = self.gliner.predict_entities(text=msg.message, labels=self.gliner_labels)

        cleaned_entities = self.clean_entities(entities)

        for ent in cleaned_entities:
            ent_data = {
                "text": ent["text"], "type": self.label_map.get(ent["label"], "MISC"),
                "confidence": ent['score'], "span": (ent["start"], ent["end"]), "contextual_mention": ""
            }

            if ent['score'] >= entity_threshold:
                res['high_confidence_entities'].append(ent_data)
            else:
                res['low_confidence_entities'].append(ent_data)
    
        res["emotion"] = self.analyze_emotion(msg.message)
        res["time_expressions"] = self.analyze_time(msg.message)
    
    def _contextual_features(self, message_block: Tuple[str, List[str]], res: Dict):

        str_version, list_version = message_block

        if message_block and str_version.strip():
            docs: List[Doc] = list(self.spacy.pipe(texts=list_version))
            current_doc = docs[-1]
            appositive_map = {}
            for token in current_doc:
                if token.dep_ == "appos":
                    head = token.head
                    appositive_map[(head.idx, head.idx + len(head))] = \
                        (token.idx, token.idx + len(token))
            res["appositive_map"] = appositive_map

            coref_clusters = []
            if current_doc._.coref_clusters:
                for cluster_spans in current_doc._.coref_clusters:
                    if len(cluster_spans) < 2:
                        continue
                    
                    main_span = current_doc[cluster_spans[0][0]:cluster_spans[0][1]]
                    mention_spans = [current_doc[span[0]:span[1]] for span in cluster_spans]

                    cluster_obj = CoreferenceCluster(
                        main=main_span,
                        mentions=mention_spans,
                        size=len(cluster_spans)
                    )
                    
                    coref_clusters.append(cluster_obj)

                res["coref_clusters"] = coref_clusters

            high_conf_entities = res["high_confidence_entities"]
            noun_chunks = [(chunk.text, chunk.start_char, chunk.end_char) for chunk in current_doc.noun_chunks]
            for i, entity in enumerate(high_conf_entities):
                ent_start, ent_end = entity["span"]
                for chunk_text, chunk_start, chunk_end in noun_chunks:
                    if ent_start >= chunk_start and ent_end <= chunk_end:
                        high_conf_entities[i]["contextual_mention"] = chunk_text
                        break

            chunks_to_analyze = {}
            for i, entity in enumerate(res["low_confidence_entities"]):
                ent_start, ent_end = entity["span"]
                for chunk_text, chunk_start, chunk_end in noun_chunks:
                    if ent_start >= chunk_start and ent_end <= chunk_end:
                        chunks_to_analyze[chunk_text] = (chunk_start, chunk_end)
                        break

            chunk_texts = list(chunks_to_analyze.keys())
            batched_results = []
            if chunk_texts:
                with torch.no_grad():
                    batched_results = self.gliner.predict_entities(text=chunks_to_analyze, labels=self.gliner_labels)


            for i, chunk_text in enumerate(chunk_texts):
                entities_in_chunk = batched_results[i]
                chunk_start, _ = chunks_to_analyze[chunk_text]
                cleaned_entities_in_chunk = self.clean_entities(entities_in_chunk)

                for new_ent in cleaned_entities_in_chunk:
                    if new_ent['score'] >= 0.8: # stricter threshold since dealing with original low conf entities
                        adjusted_start = new_ent['start'] + chunk_start
                        adjusted_end = new_ent['end'] + chunk_start

                        high_conf_entities.append({
                            "text": new_ent["text"],
                            "span": (adjusted_start, adjusted_end),
                            "type": self.label_map.get(new_ent["label"], "MISC"),
                            "confidence": new_ent["score"],
                            "contextual_mention": chunk_text
                        })
                                

    def start_process(self, message_block: Tuple[str, List[str]], msg: MessageData,
                      entity_threshold: float = 0.7):
        res: Dict[RES_LABEL, Union[List, Dict]] = {
            "high_confidence_entities": [],
            "low_confidence_entities": [],
            "noun_chunks": [],
            "coref_clusters": [],
            "emotion": [],
            "appositive_map": {}
        }

        self._extract_message_features(msg=msg, entity_threshold=entity_threshold, res=res)
        self._contextual_features(message_block=message_block, res=res)
        return res
    



        