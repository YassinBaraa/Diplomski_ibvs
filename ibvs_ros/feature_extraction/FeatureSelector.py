from abc import ABC, abstractmethod
import numpy as np
from ..pipeline.IBVSContext import IBVSContext

class FeatureSelector(ABC):

    @abstractmethod
    def extract(self, ctx: IBVSContext) -> np.ndarray:
        pass
