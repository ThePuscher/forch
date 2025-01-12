from .module import Module
import numpy as np

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / np.exp(-x)
    
    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))