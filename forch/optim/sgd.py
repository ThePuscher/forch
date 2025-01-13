from ..parameter import Parameter
from typing import List

class SGD():
    def __init__(self, parameters: List[Parameter], lr: float):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()