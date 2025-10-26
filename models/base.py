    
from abc import ABC, abstractmethod
from typing import Dict

class AbstractLLMClient(ABC):
    """
    An abstract base class for LLM clients, ensuring a consistent interface.
    """

    @abstractmethod
    def generate_json_response(self, system_prompt: str, user_prompt: str) -> Dict | None:
        """
        Generates a response from the LLM and ensures it's valid JSON.

        Args:
            system_prompt: The system-level instruction for the model.
            user_prompt: The user's specific request or data.

        Returns:
            A dictionary parsed from the LLM's JSON output, or None on failure.
        """
        pass

    @abstractmethod
    def generate_text_response(self, system_prompt: str, prompt: str) -> str | None:
        """
        Generates a simple, free-form text response from the LLM.

        Args:
            system_prompt: The system-level instruction for the model.
            prompt: The complete prompt for the model.

        Returns:
            The text content of the response, or None on failure.
        """
        pass

  