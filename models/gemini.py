import os
import json
import logging
import google.generativeai as genai
from .base import AbstractLLMClient
from typing import Dict

logger = logging.getLogger(__name__)

class GeminiClient(AbstractLLMClient):
    def __init__(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name,
            system_instruction=""
        )
        logger.info(f"GeminiClient initialized with model: {model_name}")

    def generate_json_response(self, system_prompt: str, user_prompt: str) -> Dict | None:
        try:
            self.model.system_instruction = system_prompt
            
            response = self.model.generate_content(
                user_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error generating JSON response from Gemini: {e}")
            return None

    def generate_text_response(self, system_prompt: str, prompt: str) -> str | None:
        try:
            self.model.system_instruction = system_prompt
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating text response from Gemini: {e}")
            return None