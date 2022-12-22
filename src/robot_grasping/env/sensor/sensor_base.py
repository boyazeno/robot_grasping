from abc import ABC, abstractmethod
from typing import Any, NoReturn, Dict

class SensorBase(ABC):
    def __init__(self, name:str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self)->str:
        return self._name

    @abstractmethod
    def configure(self, config:Dict) -> NoReturn:
        pass

    @abstractmethod
    def get_data(self)->Any:
        pass

