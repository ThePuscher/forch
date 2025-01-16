import numpy as np

class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(self.value)

    def zero_grad(self):
        self.grad = np.zeros_like(self.value)