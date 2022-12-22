from abc import ABC, abstractmethod
from robot_grasping.env.env import Env

class BaseRewardAssigner(ABC):
    def __init__(self, env:Env) -> None:
        super().__init__()
        self._env = env

    @abstractmethod
    def get_reward(self, ) -> float:
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass