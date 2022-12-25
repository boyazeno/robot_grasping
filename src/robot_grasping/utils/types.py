from dataclasses import dataclass, field
from typing import List, Dict
from collections import namedtuple
import transformations
import numpy as np


@dataclass
class Pose:
    position: List = field(default_factory=lambda:[0.0, 0.0, 0.0])
    orientation: List = field(default_factory=lambda:[1.0, 0.0, 0.0, 0.0])


    @classmethod
    def from_numpy(cls, mat: np.ndarray):
        orientation = transformations.quaternion_from_matrix(mat)
        position = mat[:3, 3]
        return Pose(position=position, orientation=orientation)


    def to_numpy(self):
        rot = transformations.quaternion_matrix(self.orientation)
        rot[:3, 3] = self.position
        return rot


    def rotate_local(self, x=0.0, y=0.0, z=0.0):
        transform_x = transformations.euler_matrix(x, 0., 0., axes="sxyz")
        transform_y = transformations.euler_matrix(0. , y, 0., axes="sxyz")
        transform_z = transformations.euler_matrix(0. , 0., z, axes="sxyz")
        return Pose.from_numpy(self.to_numpy()@transform_x@transform_y@transform_z)


    def translate_local(self, x=0.0, y=0.0, z=0.0):
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        return Pose.from_numpy(self.to_numpy()@transform)


    def rotate_global(self, x=0.0, y=0.0, z=0.0):
        transform = transformations.euler_matrix(x, y, z, axes="sxyz")
        return Pose.from_numpy(transform@self.to_numpy())


    def translate_global(self, x=0.0, y=0.0, z=0.0):
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        return Pose.from_numpy(transform@self.to_numpy())

    @property
    def orientation_wxyz(self):
        return self.orientation

    @property
    def orientation_xyzw(self):
        return self.orientation[1:]+[self.orientation[0]]

    
    def dot(self, mat):
        if isinstance(mat,Pose):
            return self.from_numpy(self.to_numpy()@mat.to_numpy())
        elif isinstance(mat, np.ndarray) and mat.shape == (4,4):
            return self.from_numpy(self.to_numpy()@mat)
        else:
            raise TypeError("Input mat should be 4*4 matrix or Pose.")

    def inv(self):
        return Pose.from_numpy(np.linalg.inv(self.to_numpy()))


    def __str__(self) -> str:
        return f"Position: {self.position}, Orientation: {self.orientation}"


@dataclass
class NameWithPose:
    name: str
    pose: Pose = Pose()

    def __str__(self) -> str:
        return f"Name: {self.name}, Pose: [{self.pose}]"


@dataclass
class Action:
    robot_joint_velocity: List[float]
    robot_hand_velocity: float


@dataclass
class State:
    joint_value: List[float]
    object_model_infos: List[NameWithPose]
    visual_data: Dict[str, Dict[str, np.ndarray]]
    is_terminated: bool
    

JointInfo = namedtuple(
    "jointInfo",
    [
        "id",
        "name",
        "type",
        "damping",
        "friction",
        "lowerLimit",
        "upperLimit",
        "maxForce",
        "maxVelocity",
        "controllable",
    ],
)
