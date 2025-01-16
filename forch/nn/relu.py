from .module import Module
import numpy as np

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return np.where(self.input > 0, d_out, 0)