import json
import logging
import ollama
from .base import AbstractLLMClient
from typing import Dict

logger = logging.getLogger(__name__)

class OllamaClient(AbstractLLMClient):
    def __init__(self, model_name: str):
        base_url = "http://localhost:11434"
        self.client = ollama.Client(host=base_url)
        self.model_name = model_name
        self._check_model_availability()
        logger.info(f"OllamaClient initialized for model: {self.model_name} at {base_url}")

    def _check_model_availability(self):
        try:
            models = self.client.list().get('models', [])
            model_names = [m['name'] for m in models]
            if self.model_name not in model_names:
                logger.warning(f"Model '{self.model_name}' not found in Ollama. Please run `ollama pull {self.model_name}`.")
        except Exception as e:
            logger.error(f"Could not connect to Ollama at {self.client._host}. Is Ollama running? Error: {e}")
            raise ConnectionError("Failed to connect to Ollama service.")

    def generate_json_response(self, system_prompt: str, user_prompt: str) -> Dict | None:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format='json'
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            logger.error(f"Error generating JSON response from Ollama: {e}")
            return None

    def generate_text_response(self, system_prompt: str, prompt: str) -> str | None:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating text response from Ollama: {e}")
            return None