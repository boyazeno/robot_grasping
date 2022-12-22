import logging
from enum import Enum
from pathlib import Path
from robot_grasping.utils.types import Pose
from typing import List
from abc import ABC
from abc import abstractclassmethod
import pybullet as p


class ObjectModelBase(ABC):
    def __init__(self, name: str) -> None:
        self._name = name
        self._object_id = None
        self._init_pose = None

    @abstractclassmethod
    def load(self, init_pose: Pose):
        pass

    def set_pose(self, pose: Pose):
        p.resetBasePositionAndOrientation(self._object_id, pose.position, pose.orientation_xyzw)

    def get_pose(self)->Pose:
        position, orientation = p.getBasePositionAndOrientation(self._object_id)
        return Pose(position=position, orientation=[orientation[3],orientation[0],orientation[1],orientation[2]])

    def get_velocity(self)->List[float]:
        linear_v, angular_v = p.getBaseVelocity(self._object_id)
        return linear_v, angular_v

    def reset(self):
        if self._init_pose is not None:
            logging.info(f"Reset {self._name} to {self._init_pose}.")
            self.set_pose(self._init_pose)
        else:
            logging.info(f"Reset {self._name} to origin.")
            self.set_pose(Pose())

    @property
    def name(self):
        return self._name


class MeshObjectMdoel(ObjectModelBase):
    def __init__(self, name: str, mesh_dir: str, mass: float = 1.0) -> None:
        super().__init__(name)
        self._mesh_path = Path(mesh_dir) / (self._name + ".ply")
        if not self._mesh_path.exists():
            raise FileNotFoundError(
                f"Not found mesh for {name} in {self._mesh_path.as_posix()}"
            )
        self._scale = 1.0
        self._mass = mass
        self._object_id = None
        self._init_pose = None

    def load(self, init_pose: Pose):
        col_mesh_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=self._mesh_path.as_posix(), meshScale=self._scale)
        self._init_pose = init_pose
        self._object_id = p.createMultiBody(self._mass, col_mesh_id, -1, self._init_pose.position, self._init_pose.orientation_xyzw)


class PrimitiveType(Enum):
    Cube = 1
    Sphere = 2
    Cylinder = 3


class PrimitiveObjectModel(ObjectModelBase):
    def __init__(self, name: str, type: PrimitiveType, size: List[float], mass: float = 1.0) -> None:
        self._size = size
        self._type = type
        self._mass = mass
        super().__init__(name)


    def load(self, init_pose: Pose):
        if self._type is PrimitiveType.Cube:
            assert len(self._size) == 3
            col_primitve_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=self._size)
        elif self._type is PrimitiveType.Sphere:
            assert len(self._size) == 1
            col_primitve_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=self._size[0])
        elif self._type is PrimitiveType.Cylinder:
            assert len(self._size) == 1
            col_primitve_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=self._size[0], height=self._size[1])
        else:
            raise KeyError(f"Primitive type not implemented yet!")

        self._init_pose = init_pose
        self._object_id = p.createMultiBody(self._mass, col_primitve_id, -1, self._init_pose.position, self._init_pose.orientation_xyzw)



