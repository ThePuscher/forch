from .module import Module
import numpy as np

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)