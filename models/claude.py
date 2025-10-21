import os
import json
import logging
import anthropic
from .base import AbstractLLMClient
from typing import Dict

logger = logging.getLogger(__name__)

class ClaudeClient(AbstractLLMClient):
    def __init__(self, model_name: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        logger.info(f"ClaudeClient initialized with model: {self.model_name}")

    def generate_json_response(self, system_prompt: str, user_prompt: str) -> Dict | None:
        try:
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "{"}
            ]
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                max_tokens=4096,
                messages=messages
            )
            json_text = "{" + response.content[0].text
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error generating JSON response from Claude: {e}")
            return None

    def generate_text_response(self, system_prompt: str, prompt: str) -> str | None:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text response from Claude: {e}")
            return None