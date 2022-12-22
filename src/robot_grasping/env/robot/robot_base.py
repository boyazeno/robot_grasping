from abc import ABC
from abc import abstractclassmethod
from typing import List
from robot_grasping.utils.types import Pose


class RobotBase(ABC):
    def __init__(self) -> None:
        pass


    @abstractclassmethod
    def load(self, init_base_pose:Pose, init_joint_position:List[float], init_hand_position:float):
        return


    @abstractclassmethod
    def reset(self):
        return


    @abstractclassmethod
    def set_joint_velocity(self, velocity:List[float]):
        return

    
    @abstractclassmethod
    def set_joint_position(self, position:List[float]):
        return


    @abstractclassmethod
    def set_hand_position(self, position:float):
        return


    @abstractclassmethod
    def set_hand_velocity(self, velocity:float):
        return

    
    @abstractclassmethod
    def get_joint_position(self)->List[float]:
        return 

    
    @abstractclassmethod
    def get_joint_velocity(self)->List[float]:
        return


    @abstractclassmethod
    def get_tcp_pose(self)->Pose:
        return


    @abstractclassmethod
    def get_tcp_velocity(self)->List[float]:
        return