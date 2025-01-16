from .module import Module
import numpy as np
from ..parameter import Parameter

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # +1 for bias trick
        self.W = Parameter(np.random.randn(in_dim + 1, out_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # bias trick
        ones = np.ones((x.shape[0], 1))
        x = np.hstack((x, ones))
        self.input = x
        return x @ self.W.value
    
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        self.W.grad = self.input.T @ d_out
        d_input = d_out @ self.W.value[:-1].T  # exclude bias term
        return d_input