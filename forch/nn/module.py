from abc import ABC, abstractmethod

class Module(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __call__(self, x):
        return self.forward(x)