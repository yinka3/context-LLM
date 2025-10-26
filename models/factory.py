import os
from .base import AbstractLLMClient
from .gemini import GeminiClient
from .claude import ClaudeClient
from .ollama import OllamaClient

def get_llm_client() -> AbstractLLMClient:
    """
    Factory function to get an LLM client based on environment variables.

    Reads the following environment variables:
    - LLM_PROVIDER: "gemini", "claude", or "ollama"
    - GEMINI_MODEL: The model name for Gemini (e.g., "gemini-2.5-flash-lite")
    - CLAUDE_MODEL: The model name for Claude (e.g., "claude-haiku-4-5")
    - OLLAMA_MODEL: The model name for Ollama (e.g., "llama3:70b-instruct")

    Returns:
        An instance of a class that implements AbstractLLMClient.
        
    Raises:
        ValueError: If the provider is not supported or required env vars are missing.
    """
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM_PROVIDER environment variable is not set.")

    provider = provider.lower()

    if provider == "gemini":
        model_name = os.getenv("GEMINI_MODEL")
        if not model_name:
            raise ValueError("GEMINI_MODEL environment variable must be set for the Gemini provider.")
        return GeminiClient(model_name=model_name)
    
    elif provider == "claude":
        model_name = os.getenv("CLAUDE_MODEL")
        if not model_name:
            raise ValueError("CLAUDE_MODEL environment variable must be set for the Claude provider.")
        return ClaudeClient(model_name=model_name)
        
    elif provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL")
        if not model_name:
            raise ValueError("OLLAMA_MODEL environment variable must be set for the Ollama provider.")
        return OllamaClient(model_name=model_name)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")