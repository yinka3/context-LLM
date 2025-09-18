from collections import namedtuple
import json
import spacy
import torch
from datetime import datetime
from sutime import SUTime
from gliner import GLiNER
from spacy.tokens import Doc, Token
from shared.dtypes import MessageData
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from transformers import pipeline
from fastcoref import spacy_component


logger = logging.getLogger(__name__)

CoreferenceCluster = namedtuple('CoreferenceCluster', ['main', 'mentions', 'size'])

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
        # Use the current time as the reference for expressions like "tonight"
        reference_date = datetime.now().isoformat()
        time_expressions = self.sutime.parse(text, reference_date)
        return time_expressions

    def analyze_emotion(self, text: str) -> List[Dict]:
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
        res: Dict[str, Union[List, Dict]] = {
            "high_confidence_entities": [],
            "low_confidence_entities": [],
            "noun_chunks": [],
            "coref_clusters": [],
            "emotion": [],
            "tier2_flags": []
            }

        self._extract_message_features(msg=msg, entity_threshold=entity_threshold, res=res)
        self._contextual_features(message_block=message_block, res=res)
        return res
    



        # if not msg.message or not msg.message.strip():
        #     return res
        
        # msg_length = len(msg.message.strip())
        # if msg_length > 800:
        #     payload = {
        #         "original_text": msg.message,
        #         "reason_details": f"Message length is {len(msg.message)} characters."
        #     }
        #     res["tier2_flags"].append({
        #         "reason": "MESSAGE_TOO_LONG",
        #         "priority": 4,
        #         "payload": payload
        #     })
        #     logger.warning(f"Flagged for Tier 2: Message too long: {len(msg.message)} chars | Payload Info: {payload} \n")
        #     return res # we will now ship this off to tier 2 also


#          for sentence in doc.sents:
        #     root_verb = sentence.root
        #     # ensure the root of the sentence is a verb before processing
        #     if root_verb.pos_ == "VERB":
        #         verb_tree = self._build_verb_tree(root_verb)
        #         res["verbs"].append(verb_tree)

        #     for token in sentence:
        #         described_noun = None
        #         descriptor_token = token
        #         if token.pos_ == "ADJ":
        #             if token.dep_ == "amod":
        #                 described_noun = token.head
        #             elif token.dep_ == "acomp":
        #                 verb_head = token.head
        #                 subjects = [child for child in verb_head.children if child.dep_ in ("nsubj", "nsubjpass")]
        #                 if subjects:
        #                     described_noun = subjects[0]
        #             elif token.dep_ == "oprd":
        #                 verb_head = token.head
        #                 objects = [child for child in verb_head.children if child.dep_ == "dobj"]
        #                 if objects:
        #                     described_noun = objects[0]
        #         elif token.dep_ in ("acl", "relcl"):
        #             described_noun = token.head

        #         if described_noun:
        #             res["adjectives"].append({
        #                 "text": descriptor_token.text,
        #                 "lemma": descriptor_token.lemma_,
        #                 "span": (descriptor_token.idx, descriptor_token.idx + len(descriptor_token.text)),
        #                 "describes_text": described_noun.text,
        #                 "describes_lemma": described_noun.lemma_
        #             })




    # def _build_verb_tree(self, verb_token: Token, inherited_subject: Optional[Token] = None) -> Dict:
    #     """
    #     Recursively builds a nested dictionary for a verb and its related actions.
    #     """
    #     # Base structure for the current verb
    #     verb_data = {
    #         "text": verb_token.text,
    #         "lemma": verb_token.lemma_,
    #         "span": (verb_token.idx, verb_token.idx + len(verb_token.text)),
    #         "relation": verb_token.dep_,
    #         "subjects": [],
    #         "objects": [],
    #         "indirect_objects": [],
    #         "modals": [],
    #         "sub_actions": [],
    #         "conjoined_actions": [],
    #         "contextual_actions": []
    #     }

    #     subject_head_token = None

    #     for child in verb_token.children:
    #         if child.dep_ in ("nsubj", "nsubjpass", "agent", "csubj"):
    #             # Determine if it's a subject or object based on passivity
    #             participant_list = verb_data["subjects"] if child.dep_ != "nsubjpass" else verb_data["objects"]
                
    #             # Handle the 'agent' case where the noun is a child of 'by'
    #             participant_token = child
    #             if child.dep_ == "agent":
    #                 agent_nouns = [c for c in child.children if c.dep_ == "pobj"]
    #                 if agent_nouns:
    #                     participant_token = agent_nouns[0]
    #                 else:
    #                     continue # Skip if agent has no noun

    #             participant_list.append({
    #                         "text": participant_token.text,
    #                         "lemma": participant_token.lemma_,
    #                         "span": (participant_token.idx, participant_token.idx + len(participant_token.text))
    #                     })
    #             for conjunct in participant_token.conjuncts:
    #                 if conjunct.dep_ == "conj":
    #                     participant_list.append({
    #                         "text": conjunct.text,
    #                         "lemma": conjunct.lemma_,
    #                         "span": (conjunct.idx, conjunct.idx + len(conjunct.text))
    #                     })

    #             if child.dep_ == "nsubj":
    #                 subject_head_token = child

    #         elif child.dep_ == 'aux': 
    #             verb_data["modals"].append(child.text.lower())

    #         elif child.dep_ == "dobj":
    #             verb_data["objects"].append({
    #                 "text": child.text,
    #                 "lemma": child.lemma_,
    #                 "span": (child.idx, child.idx + len(child.text))
    #             })
    #             for conjunct in child.conjuncts:
    #                 if conjunct.dep_ == "conj":
    #                     verb_data["objects"].append({
    #                     "text": conjunct.text,
    #                     "lemma": conjunct.lemma_,
    #                     "span": (conjunct.idx, conjunct.idx + len(conjunct.text))
    #                 })
            
    #         elif child.dep_ == "dative":
    #             verb_data["indirect_objects"].append({
    #                 "text": child.text,
    #                 "lemma": child.lemma_,
    #                 "span": (child.idx, child.idx + len(child.text))
    #             })
    #             for conjunct in child.conjuncts:
    #                 if conjunct.dep_ == "conj":
    #                     verb_data["indirect_objects"].append({
    #                     "text": conjunct.text,
    #                     "lemma": conjunct.lemma_,
    #                     "span": (conjunct.idx, conjunct.idx + len(conjunct.text))
    #                 })
            
    #         elif child.dep_ == "prep":
    #             if (verb_token.lemma_, child.text.lower()) in self.PREPOSITIONAL_VERB_PAIRS:
    #                 prep_objects = [c for c in child.children if c.dep_ == "pobj"]
    #                 if prep_objects and not verb_data["objects"]:
    #                     pobj = prep_objects[0]
    #                     verb_data["objects"].append({
    #                         "text": pobj.text,
    #                         "lemma": pobj.lemma_,
    #                         "span": (pobj.idx, pobj.idx + len(pobj.text))
    #                     })
    #                     for conjunct in pobj.conjuncts:
    #                         verb_data["objects"].append({
    #                             "text": conjunct.text,
    #                             "lemma": conjunct.lemma_,
    #                             "span": (conjunct.idx, conjunct.idx + len(conjunct.text))
    #                         })

    #         elif child.pos_ == "VERB":
    #             current_subject = subject_head_token if subject_head_token else inherited_subject
    #             if child.dep_ in ("xcomp", "ccomp"):
    #                 verb_data["sub_actions"].append(self._build_verb_tree(child, inherited_subject=current_subject))
    #             elif child.dep_ == "conj":
    #                 verb_data["conjoined_actions"].append(self._build_verb_tree(child, inherited_subject=current_subject))
    #             elif child.dep_ == "advcl":
    #                 verb_data["contextual_actions"].append(self._build_verb_tree(child, inherited_subject=current_subject))

    #     # Handle implied subjects from the parent verb
    #     if not verb_data["subjects"] and inherited_subject:
    #         verb_data["subjects"].append({
    #             "text": inherited_subject.text,
    #             "lemma": inherited_subject.lemma_,
    #             "span": (inherited_subject.idx, inherited_subject.idx + len(inherited_subject.text))
    #         })


    #     return verb_data