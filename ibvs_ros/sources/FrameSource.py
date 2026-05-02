from abc import ABC, abstractmethod

class FrameSource(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def release(self):
        pass