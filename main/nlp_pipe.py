from gliner import GLiNER
import torch
from schema.dtypes import *
from typing import List
import logging
from transformers import pipeline


logger = logging.getLogger(__name__)

class NLPPipeline:
    
    def __init__(
        self,
        gliner_model: str = "urchade/gliner_large-v2.1",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._init_gliner(gliner_model)
        self._init_emotion(emotion_model)
    
    def _init_gliner(self, model_name: str):
        self.gliner = GLiNER.from_pretrained(model_name)
        self.gliner = self.gliner.to(self.device)
        if torch.cuda.is_available():
            self.gliner.model.half()
        self.gliner.model.eval()
    
    def _init_emotion(self, model_name: str):
        device_id = 0 if torch.cuda.is_available() else -1
        self.emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=device_id
        )
    
    def extract_mentions(self, text: str, threshold: float = 0.5) -> List[str]:
        """Just returns entity text for candidate lookup."""
        if not text or not text.strip():
            return []
        
        with torch.no_grad():
            entities = self.gliner.predict_entities(
                text=text.strip(),
                labels=["person", "organization", "location", "event", "product", "topic"]
            )
        
        return list({ent["text"] for ent in entities if ent["score"] >= threshold})
    
    def analyze_emotion(self, text: str) -> List[dict]:
        if not text or not text.strip():
            return []
        try:
            results = self.emotion_classifier(text)
            return results[0] if results else []
        except Exception:
            return []