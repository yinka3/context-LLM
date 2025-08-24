import spacy
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from dtypes import MessageData
from typing import Dict, List, Literal, Union
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class NLP_PIPE:
    
    def __init__(self, type: Literal["short", "long"] = "short"):
        self.spacy = None
        self.type = type
        
        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            logging.info("Vader not initialized")
            raise
        
        try:
            self.spacy = spacy.load('en_core_web_md')
        except:
            logging.info("Spacy not initialized")
            raise
        
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
        
    def merge_entities(self, doc: Doc) -> Doc:
        spans: List[Span] = []
        for ent in doc.ents:
            spans.append(ent)
        
        filtered: List[Span] = filter_spans(spans)
        
        if filtered:
            doc.ents = filtered
        return doc

    def get_svo(self, doc: Doc):
        res = []
        verbs = [token for token in doc if token.dep_ == "ROOT"]

        for verb in verbs:
            svo_tri: Dict[str, Union[List, str]] = {"verb": verb.lemma_}

            for child in verb.children:

                if child.dep_ == "nsubjpass":
                    object_phrase = "".join(t.text_with_ws for t in child.subtree).strip()
                    svo_tri["object"] = object_phrase
                
                elif child.dep_ == "dobj":
                    object_phrase = "".join(t.text_with_ws for t in child.subtree).strip()
                    svo_tri["object"] = object_phrase

                if child.dep_ == "nsubj":
                    subject_phrase = "".join(t.text_with_ws for t in child.subtree).strip()
                    svo_tri["subject"] = subject_phrase
                
                
                if child.dep_ == "agent":
                    for agent_ch in child.children:
                        if agent_ch.pos_ in ("NOUN", "PROPN"):
                            agent_phrase = "".join(t.text_with_ws for t in agent_ch.subtree).strip()
                            svo_tri["subject"] = agent_phrase
            
                if child.dep_ in ("ccomp", "xcomp"):
                    complement_phrase = "".join(t.text_with_ws for t in child.subtree).strip()
                    svo_tri["complement_phrase"] = complement_phrase
                
                if child.dep_ == "advmod":
                    if "adverbials" not in svo_tri:
                        svo_tri["adverbials"] = []
                    svo_tri["adverbials"].append(child.text)

                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            if "prepositional_phrases" not in svo_tri:
                                svo_tri["prepositional_phrases"] = []
                            prep_phrase = child.text + " " + "".join(t.text_with_ws for t in pobj.subtree).strip()
                            svo_tri["prepositional_phrases"].append(prep_phrase)

            if "subject" in svo_tri and "verb" in svo_tri:
                res.append(svo_tri)
        
        return res
    
    def ner(self, doc: Doc) -> Dict[Literal["found_entities", "noun_chunks", "SVO", "sentiment"], Union[List, str]]:
        res = {"found_entities": [],
               "noun_chunks": [],
               "SVO": []}
        
        for ent in doc.ents:
            res["found_entities"].append({
                    "text": ent.text,
                    "type": ent.label_
                })
        
        for chunk in doc.noun_chunks:
            res["noun_chunks"].append({
                "root": chunk.root.text,
                "full_text": chunk.text 
            })
        
        svo = self.get_svo(doc=doc)
        res["SVO"] = svo
        return res
            

    def start_process(self, msg: MessageData):
        doc = self.spacy(text=msg.message)
        doc = self.merge_entities(doc)
        out = self.ner(doc)
        out["sentiment"] = self.analyze_sentiment(msg.message)

        return out

        

        










