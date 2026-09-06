# /Users/murseltasgin/projects/chat_rag/components/llm/ollama_llm.py
"""
Ollama LLM implementation for local models
"""
from typing import List, Dict, Any
import httpx
import ollama
from .base import BaseLLM
from core.exceptions import LLMException
from utils.logger import get_logger

logger = get_logger("OllamaLLM")


class OllamaLLM(BaseLLM):
    """Ollama LLM implementation for local models"""
    
    def __init__(
        self,
        model: str = "gpt-oss:120b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize Ollama client
        
        Args:
            model: Model name (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.client = ollama.Client(host=base_url, timeout=timeout)
        
        # Verify connection to Ollama
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Ollama server"""
        try:
            _ = self.client.list()
        except Exception as e:
            raise LLMException(f"Failed to connect to Ollama at {self.base_url}: {e}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200,
        **kwargs
    ) -> str:
        """Generate text from messages"""
        from utils.logger import RAGLogger
        from components.ingest.limits import current_guard

        # Cooperative only: the client timeout is fixed when it is built and
        # ``chat`` takes none per call, so a query deadline can refuse to
        # start this call but cannot shorten one that has started. The
        # query limits document this as the one unclampable transport.
        guard = current_guard()
        if guard is not None:
            guard.check()
        try:
            # Log request
            RAGLogger.log_llm_request(logger, messages, temperature, max_tokens)
            
            # Increase max_tokens to ensure complete responses
            adjusted_max_tokens = max(max_tokens, 300)  # Ensure minimum 300 tokens
            if adjusted_max_tokens != max_tokens:
                logger.debug(f"Adjusted max_tokens from {max_tokens} to {adjusted_max_tokens}")
            
            # Build options for ollama.chat and promote top-level format if provided
            options: Dict[str, Any] = {"temperature": temperature, "num_predict": adjusted_max_tokens}
            fmt = None
            if "format" in kwargs:
                fmt = kwargs.pop("format")
            if kwargs:
                options.update(kwargs)
                logger.debug(f"Additional options: {kwargs}")

            # Make request to Ollama
            resp = self.client.chat(
                model=self.model, messages=messages, options=options, format=fmt
            )
            generated_text = (resp.get("message", {}) or {}).get("content", "").strip()
            
            # Log response stats
            logger.debug(f"Response received. Length: {len(generated_text)} chars")
            
            # Validate we got content; if empty, retry once with safer settings (no 'thinking' fallback)
            if not generated_text:
                logger.warning("Ollama returned empty 'response'. Retrying once with adjusted parameters...")
                retry_options = dict(options)
                retry_options["num_predict"] = retry_options.get("num_predict", adjusted_max_tokens) + 200
                retry_options["temperature"] = min(temperature, 0.3)
                try:
                    resp = self.client.chat(
                        model=self.model,
                        messages=messages,
                        options=retry_options,
                        format=fmt,
                    )
                    generated_text = (resp.get("message", {}) or {}).get("content", "").strip()
                    logger.debug(f"Retry response length: {len(generated_text)} chars")
                except Exception as re:
                    logger.error(f"Retry failed: {re}")
                if not generated_text:
                    error_msg = "Ollama returned empty response after retry"
                    logger.error(error_msg)
                    RAGLogger.log_llm_response(logger, error_msg, success=False)
                    raise LLMException(error_msg)
            
            # Log successful response
            RAGLogger.log_llm_response(logger, generated_text, success=True)
            
            return generated_text
            
        except httpx.TimeoutException:
            error_msg = f"Request timeout after {self.timeout}s. Try increasing timeout or using a smaller model."
            logger.error(error_msg)
            RAGLogger.log_llm_response(logger, error_msg, success=False)
            raise LLMException(error_msg)
        except Exception as e:
            error_msg = f"Generation failed: {e}"
            logger.error(error_msg, exc_info=True)
            RAGLogger.log_llm_response(logger, error_msg, success=False)
            raise LLMException(error_msg)
    
    def get_name(self) -> str:
        """Get the LLM provider name"""
        return "Ollama"
    
    def get_model_name(self) -> str:
        """Get the specific model name"""
        return self.model
    
    def list_available_models(self) -> List[str]:
        """
        List available models on the Ollama server
        
        Returns:
            List of model names
        """
        try:
            data = self.client.list()
            models = data.get("models", [])
            return [model.get("name", "") for model in models]
        except Exception as e:
            print(f"Warning: Could not list models: {e}")
            return []

