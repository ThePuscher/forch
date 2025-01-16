from .module import Module
import numpy as np

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * self.output * (1 - self.output)