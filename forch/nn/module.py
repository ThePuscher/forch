from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class Module(ABC):
    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> np.ndarray:
        if y is not None:
            return self.forward(x, y)
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # d_out: gradient of loss w.r.t. this layer
        pass