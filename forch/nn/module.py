from abc import ABC, abstractmethod

class Module(ABC):
    def __call__(self, x):
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x, y=None):
        pass

    @abstractmethod
    def backward(self, d_out):
        # d_out: gradient of loss w.r.t. this layer
        pass