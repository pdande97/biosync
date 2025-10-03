from abc import ABC, abstractmethod
from typing import List

class SignalSource(ABC):
    @abstractmethod
    def configure(self, *, fs: int, rate: float, amplitude: float = 1.0, **kwargs) -> None: ...
    @abstractmethod
    def step(self, n: int) -> List[float]: ...

