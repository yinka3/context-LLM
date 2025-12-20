from gliner import GLiNER
from openai import AsyncOpenAI
import torch
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
        llm_client: AsyncOpenAI,
        # gliner_model: str = "numind/NuNER_Zero",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.llm_client = llm_client
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

        system_prompt = f"""
        You are VEGAPUNK -01, you have the first main operation in this process and have an important job, you are a named entity recognition system for a personal knowledge graph.

<your_purpose>
You are to read the user,{user_name}'s messages, they are talking to you and they would like you to listen to them. <find_entities>Find important entities in terms of relevance to the text given. 
</your_purpose>

<speaker_context>
All messages are written by **{user_name}**. First-person pronouns ("I", "me", "my") refer to them.
Do NOT extract {user_name} as an entity — they are the graph's root node and tracked separately.
</speaker_context>

<{user_name}'s Topics>
{topics_list}

These represent what this user cares about. Weight extraction toward entities relevant to these domains, but do not ignore clearly significant entities outside them.
</{user_name}'s Topics>

<finding_entities>
These are your rules, follow them with the appropriate degree.

1. **DO'S**
- a person's name
- a named reference or specific place, examples: "Disney Land" or "McDonalds" or "LA Fitness"
- Named people in relationships to {user_name}: family members, partners, friends, professors, coworkers — but only if named
- so main people for {user_name} should be family, partner/ex-partner, friends, and favorite things to do.
- general entities for family and partner/ex-partner is acceptable only.
- use internal knowledge because {user_name} is a real person so referable places like "MIT", "[named] univerity", President Obama, Lebron James
- if an unnamed reference/generic nouns does get accepted because of a connection to named reference, just associate it with the named reference always.
- Include titles when attached: "Professor Okonkwo" not "Okonkwo"
- Include qualifiers when part of the name: "IronWorks Gym" not "IronWorks"


2. **DO NOT'S**
- any unnamed reference/generic nouns or unspecific place with no connection to named references in this text.
  - examples: "that burger joint", "the big concert", "the red book"
- more examples: "my homework", "that girl", or any general unnamed task or place/thing.
- pronouns with no connection to a named reference ("he", "she", "they", "it", "that", "this")
- Temporal expressions ("today", "yesterday", "next week", "last month")
- Generic nouns without distinguishing names: if the mention could apply to thousands of instances without additional context, it's not specific enough.
  - examples: "the meeting", "my doctor", "the restaurant", "the app", "the conference", "the park", "Central Park"

</finding_entities>

<type_labeling>
Assign a single lowercase label for what the entity IS:
- person, professor, family-member
- place, restaurant, gym, university
- organization, company, team
- activity, hobby, sport
- product, app, software

Use the most specific obvious label. "Professor Okonkwo" → professor, not person.
When uncertain, use the general form.
</type_labeling>

<etiquette>
- Extract ALL forms of an entity as they appear in the text
- "Marcus" and "Marc" in the same text → extract both separately
- Be respectful: prefer "Dr. Sarah Chen" over "Sarah" but extract both if both appear
- Do not be rude, write it how it has been written from the text.
- Let downstream systems handle grouping — your job is to capture every mention
</etiquette>

<output_rules>
If no entities qualify: {{"entities": []}}

No commentary. No markdown fencing. No explanation. Just the JSON object.
</output_rules>
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                response_model=ExtractionResponse,
                temperature=0.0,
                max_retries=2,
            )
            logger.info(response.entities)
            return [(e.name, e.label, e.topic) for e in response.entities]

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

    
    def analyze_emotion(self, text: str) -> List[dict]:
        if not text or not text.strip():
            return []
        try:
            results = self.emotion_classifier(text)
            return results[0] if results else []
        except Exception:
            return []