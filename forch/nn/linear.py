from .module import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # +1 for bias trick
        self.W = np.random.randn(in_dim + 1, out_dim)
        self.d_W = np.zeros_like(self.W)

    def forward(self, x):
        # bias trick
        ones = np.ones((x.shape[0], 1))
        x = np.hstack((x, ones))
        self.input = x
        return x @ self.W
    
    def backward(self, d_out):
        self.d_W = self.input.T @ d_out
        d_input = d_out @ self.W[:-1].T  # exclude bias term
        return d_input