"""
Azure OpenAI LLM implementation
"""
from typing import List, Dict, Any
from openai import AzureOpenAI
from .base import BaseLLM
from utils.logger import RAGLogger, get_logger
from components.ingest.limits import current_guard, deadline_timeout
from core.exceptions import LLMException


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI LLM implementation"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str = "gpt-4o",
        api_version: str = "2024-02-15-preview"
    ):
        """Initialize Azure OpenAI client"""
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version
        
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
        except Exception as e:
            raise LLMException(f"Failed to initialize Azure OpenAI client: {e}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200,
        **kwargs
    ) -> str:
        """Generate text from messages"""
        try:
            # Log request
            logger = get_logger("AzureOpenAILLM")
            RAGLogger.log_llm_request(logger, messages, temperature, max_tokens)

            # Under a query deadline the request gets the time that is left.
            guard = current_guard()
            if guard is not None:
                guard.check()
                kwargs.setdefault("timeout", deadline_timeout(None))
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            text = response.choices[0].message.content.strip()
            RAGLogger.log_llm_response(logger, text, success=True)
            return text
        except Exception as e:
            # Log failure
            try:
                logger = get_logger("AzureOpenAILLM")
                RAGLogger.log_llm_response(logger, f"Generation failed: {e}", success=False)
            except Exception:
                pass
            raise LLMException(f"Generation failed: {e}")
    
    def get_name(self) -> str:
        """Get the LLM provider name"""
        return "AzureOpenAI"
    
    def get_model_name(self) -> str:
        """Get the specific model name"""
        return self.deployment

