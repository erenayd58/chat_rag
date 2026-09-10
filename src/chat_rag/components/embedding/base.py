"""
Base embedding abstraction
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedding(ABC):
    """Base class for embedding models"""
    
    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text(s) to embedding(s)"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the embedding model name"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        pass

