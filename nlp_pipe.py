import spacy
import torch
from gliner import GLiNER
from spacy.tokens import Doc
from dtypes import MessageData
from typing import Dict, List, Literal, Union
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spacy.matcher import DependencyMatcher

class NLP_PIPE:
    
    def __init__(self, type: Literal["short", "long"] = "short"):
        self.spacy = None
        self.type = type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if torch.cuda.is_available():
                self.gliner = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
                self.gliner = self.gliner.to(self.device)
                self.gliner.model.half().eval()
                
                logging.info(f"GLiNER loaded on GPU: {torch.cuda.get_device_name()}")       
        except Exception as e:
            logging.error(f"GLiNER GPU initialization failed: {e}")
            raise

        self.entity_labels = [
            "person",
            "project",  
            "product",
            "team",
            "organization",
            "technology", 
            "location",
            "date",
            "feature",
            "initiative",
        ]

        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            logging.error("Vader not initialized")
            raise
        
        try:
            self.spacy = spacy.load('en_core_web_md')
        except:
            logging.error("Spacy not initialized")
            raise

        self.matcher = DependencyMatcher(self.spacy.vocab)
        
        # This pattern precisely finds the "X is located in Y" structure
        pattern = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {"LEMMA": "be"}
            },
            {
                "LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject",
                "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubj:pass"]}}
            },
            {
                "LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "complement",
                "RIGHT_ATTRS": {"DEP": "acomp", "LEMMA": "locate"} # Specifically for "located"
            },
            {
                "LEFT_ID": "complement", "REL_OP": ">", "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "in"} # Specifically for "in"
            },
            {
                "LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj",
                "RIGHT_ATTRS": {"DEP": "pobj"}
            }
        ]
        self.matcher.add("IS_LOCATED_IN", [pattern])
        
        if type == "short":
            logging.info("Starting live stream pipeline")
        else:
            logging.info("Starting offload pipe, this should only be for merging")


    def analyze_sentiment(self, text: str) -> str:
        scores = self.vader.polarity_scores(text)
        
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"
        

    def get_svo(self, doc: Doc):
        res = []
        

        def get_phrase_and_span(subtree):
            tokens = list(subtree)
            text = "".join(t.text_with_ws for t in tokens).strip()
            span = (tokens[0].idx, tokens[-1].idx + len(tokens[-1].text))
            return {"text": text, "span": span}

        matches = self.matcher(doc)
        if matches:
            for _, token_ids in matches:
                verb = doc[token_ids[0]]
                subject = doc[token_ids[1]]
                pobj = doc[token_ids[4]] # The object of the preposition
                
                # Manually construct the SVO for this specific pattern
                svo_tri = {
                    "verb": "locate", # Use the semantic verb
                    "subject": get_phrase_and_span(subject),
                    "prepositional_phrases": [{
                        "text": "in",
                        "span": (doc[token_ids[3]].idx, doc[token_ids[3]].idx + 2),
                        "object": get_phrase_and_span(pobj)
                    }]
                }
                res.append(svo_tri)
                logging.info("Successfully extracted fact with DependencyMatcher.")
        
        verbs_to_process = {token for token in doc if token.dep_ in ("ROOT", "ccomp")}

        for verb in verbs_to_process:
            subject = None
            object_ = None
            for child in verb.children:
                if child.dep_ == "nsubj":
                    subject = get_phrase_and_span(child.subtree)
                elif child.dep_ == "nsubjpass":
                    object_ = get_phrase_and_span(child.subtree)
                elif child.dep_ == "agent":
                    for agent_child in child.children:
                        if agent_child.dep_ == 'pobj':
                            subject = get_phrase_and_span(agent_child.subtree)
                            break
                
            if not subject:
                continue
            
            action_verb = verb
            prepositional_nodes = [action_verb]

            # If the verb is an auxiliary (like 'is', 'was'), check for a more meaningful complement
            for child in verb.children:
                if child.dep_ in ("acomp", "attr"):
                    action_verb = child  # CRITICAL: Re-assign the action to the complement
                    prepositional_nodes.append(child)
                    break
            
            # Also handle open clausal complements (e.g., "is going TO WORK")
            for child in action_verb.children:
                if child.dep_ == "xcomp":
                    action_verb = child
                    prepositional_nodes.append(child)
                    break
            
            if not object_:
                for child in action_verb.children:
                    if child.dep_ == "dobj":
                        object_ = get_phrase_and_span(child.subtree)
                        break
            
            prepositional_phrases = []
            for node in prepositional_nodes:
                for child in node.children:
                    if child.dep_ == "prep":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":

                                prep_tokens = list(child.subtree)
                                prep_text = "".join(t.text_with_ws for t in prep_tokens).strip()
                                prep_span = (prep_tokens[0].idx, prep_tokens[-1].idx + len(prep_tokens[-1].text))
                                
                                prepositional_phrases.append({
                                    "text": prep_text,
                                    "span": prep_span,
                                    "object": get_phrase_and_span(pobj.subtree)
                                })
            
            if object_ or prepositional_phrases:
                svo_tri = {"verb": action_verb.lemma_, "subject": subject}
                if object_:
                    svo_tri["object"] = object_
                if prepositional_phrases:
                    svo_tri["prepositional_phrases"] = prepositional_phrases
                res.append(svo_tri)
        
        return res
    
    def get_attributes(self, doc: Doc):
        res = []
        
        def get_phrase_and_span(subtree):
            tokens = list(subtree)
            text = "".join(t.text_with_ws for t in tokens).strip()
            span = (tokens[0].idx, tokens[-1].idx + len(tokens[-1].text))
            return {"text": text, "span": span}

        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                subject = None
                attribute = None
                
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = get_phrase_and_span(child.subtree)
                    elif child.dep_ in ("acomp", "attr"):
                        attribute = get_phrase_and_span(child.subtree)
                
                if subject and attribute:
                    res.append({
                        "subject": subject,
                        "relation": token.lemma_, 
                        "value": attribute
                    })
        return res
    
    def deduplicate_entities(self, entities):
        if not entities:
            return []
        
        # sort by position then length (prefer longer entities)
        entities = sorted(entities, key=lambda x: (x['start'], -len(x['text'])))
        
        filtered = []
        last_end = -1
        
        for ent in entities:
            if ent['start'] >= last_end:
                filtered.append(ent)
                last_end = ent['end']
        
        return filtered
    

    def start_process(self, msg: MessageData):
        res: Dict[Literal["found_entities", "noun_chunks", "SVO", "sentiment", "attributes"], Union[List[Dict], str]] = {
            "found_entities": [],
            "noun_chunks": [],
            "SVO": [],
            "sentiment": "",
            "attributes": []
            }
        
        # if "http" in msg.message:
        #     this will probably then use the web search tool, for later
        
        if not msg.message or not msg.message.strip():
            return res
        
        if len(msg.message) > 2000:
            logging.warning(f"Message too long: {len(msg.message)} chars")
            return res # this just for now, will incorporate split texting later when I know this version works first
        
        entities = self.gliner.predict_entities(text=msg.message, labels=self.entity_labels)
        entities = self.deduplicate_entities(entities)

        for ent in entities:
            res['found_entities'].append(
                {
                    "text": ent["text"],
                    "type": ent["label"],
                    "confidence": ent['score'],
                    "confidence_level": "high" if ent['score'] > 0.7 else "medium",
                    "span": (ent["start"], ent["end"])
                }
            )

        doc = self.spacy(text=msg.message)
        for chunk in doc.noun_chunks:
            res["noun_chunks"].append({
                "root": chunk.root.text,
                "text": chunk.text,
                "span": (chunk.start_char, chunk.end_char)
            })

        res["SVO"] = self.get_svo(doc=doc)
        res["attributes"] = self.get_attributes(doc=doc)
        res["sentiment"] = self.analyze_sentiment(msg.message)

        existing_spans = {tuple(ent['span']) for ent in res['found_entities']}
        for token in doc:
            if token.lemma_.lower() in ['i', 'me', 'my', 'myself']:
                is_overlapping = False
                for start, end in existing_spans:
                    if token.idx >= start and token.idx < end:
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    res['found_entities'].append({
                        "text": token.text,
                        "type": "person",
                        "confidence": 1.0,
                        "confidence_level": "high",
                        "span": (token.idx, token.idx + len(token.text))
                    })

        return res

        

        










