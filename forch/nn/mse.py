import numpy as np
from .module import Module

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        return np.mean((y_hat - y)**2)
    
    def backward(self):
        return 2 * (self.y_hat - self.y) / self.y.size