import numpy as np
from typing import Tuple

class Dataset():
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = np.array(features)
        self.labels = np.array(labels)

    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.features[idx], self.labels[idx]