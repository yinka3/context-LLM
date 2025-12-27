from gliner import GLiNER
from openai import AsyncOpenAI
import torch
from log.llm_trace import get_trace_logger
from main.prompts import ner_prompt
from main.service import LLMService
from schema.dtypes import *
from typing import List, Tuple
from loguru import logger
from transformers import pipeline
from main.block_list import TEMPORAL_BLOCKLIST


class EntityItem(BaseModel):
    name: str = Field(..., description="The exact text span of the Named Entity.")
    label: str = Field(..., description="A concise, lowercase semantic type describing what the entity IS.")
    topic: str = Field(..., description="The most relevant topic from the user's active topics list.")

class ExtractionResponse(BaseModel):
    entities: List[EntityItem] = Field(..., description="A list of valid Named Entities extracted from the text. Return an empty list if no specific proper nouns are found.")

class NLPPipeline:
    
    def __init__(
        self,
        llm: LLMService,
        # gliner_model: str = "numind/NuNER_Zero",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.llm_client = llm
        # self._init_gliner(gliner_model)
        self._init_emotion(emotion_model)
        
    
    # def _init_gliner(self, model_name: str):
    #     self.gliner = GLiNER.from_pretrained(model_name)
    #     self.gliner = self.gliner.to(self.device)
    #     if torch.cuda.is_available():
    #         self.gliner.model.half()
    #     self.gliner.model.eval()
    #     logger.info("Gliner model initialized")
    
    def _init_emotion(self, model_name: str):
        device_id = 0 if torch.cuda.is_available() else -1
        self.emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device_id
        )
    
    # def extract_mentions(self, text: str, threshold: float) -> List[Tuple[str, str]]:
    #     """Returns list of (mention_text, entity_type) tuples."""
    #     if not text or not text.strip():
    #         return []

    #     with torch.no_grad():
    #         ENTITY_LABELS = [
    #             "person",
    #             "organization",
    #             "academic subject",
    #             "educational institution",
    #             "holiday",
    #         ]
    #         entities = self.gliner.predict_entities(
    #             text=text.strip(),
    #             labels=ENTITY_LABELS
    #         )

        
    #     mentions = {}
    #     for ent in entities:
    #         logger.info(f"Score for {ent["text"]} => {ent['score']} => label {ent['label']}")
    #         if ent["score"] >= threshold:
    #             if ent["text"].lower() not in TEMPORAL_BLOCKLIST:
    #                 text_key = ent["text"]
    #                 if text_key not in mentions or ent["score"] > mentions[text_key][1]:
    #                     mentions[text_key] = (ent["label"].upper(), ent["score"])

    #     return [(text, type_score[0]) for text, type_score in mentions.items()]

    async def extract_mentions(self, user_name: str, topics_list: List, text: str) -> List[Tuple[str, str, str]]:
        """
        Extracts entities using Qwen-2.5-14B via OpenRouter.
        """
        if not text or not text.strip():
            return []
        logger.info(topics_list)

        system_prompt = ner_prompt(user_name, topics_list)
        
        response = await self.llm_client.call_structured(system_prompt, text, ExtractionResponse)
        return [(e.name, e.label, e.topic) for e in response.entities]

    
    def analyze_emotion(self, text: str) -> List[dict]:
        if not text or not text.strip():
            return []
        try:
            results = self.emotion_classifier(text)
            return results[0] if results else []
        except Exception:
            return []