from .module import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # +1 for kernel trick
        self.W = np.random.randn(in_dim + 1, out_dim)

    def forward(self, x):
        # kernel trick
        ones = np.ones((x.shape[0], 1))
        x = np.hstack((x, ones))
        return x @ self.W
    
    def backward(self):
        return super().backward()
    