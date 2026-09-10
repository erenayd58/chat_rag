"""
Base LLM abstraction
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200,
        **kwargs
    ) -> str:
        """Generate text from messages"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the LLM provider name"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the specific model name"""
        pass
