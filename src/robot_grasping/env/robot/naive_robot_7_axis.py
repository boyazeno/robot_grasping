from pathlib import Path
from typing import List, Tuple, Optional
import pybullet as p
import numpy as np
import logging
from robot_grasping.env.robot.robot_base import RobotBase
from robot_grasping.utils.types import Pose


class CollisionPairSet:
    def __init__(
        self, init_collision_pairs: Optional[List[List[float]]]
    ) -> None:
        """Used to check if a given collision pair is the same as pre-saved collision pair (order will be ignored)"""
        self._set = set()
        if init_collision_pairs is not None:
            for collision_pair in init_collision_pairs:
                self.add(collision_pair=collision_pair)

    def add(self, collision_pair: List[float]):
        self._set.add(self._to_tuple(collision_pair=collision_pair))

    def __contains__(self, collision_pair: List[float]) -> int:
        self._to_tuple(collision_pair=collision_pair)
        return self._to_tuple(collision_pair=collision_pair) in self._set

    def _to_tuple(self, collision_pair: List[float]) -> Tuple[float]:
        return tuple([max(collision_pair), min(collision_pair)])


class NaiveRobot7Axis(RobotBase):
    def __init__(self) -> None:
        super().__init__()
        self._robot_urdf = Path(
            "/home/boya/noetic_ws/src/RL/robot_grasping/asset/urdf/naive_robot_7_axis.urdf"
        )
        if not self._robot_urdf.exists():
            raise FileNotFoundError(
                f"Robot urdf not exist in path {self._robot_urdf}"
            )

        self._init_base_pose = None
        self._init_joint_position = None
        self._init_hand_position = None
        self._robot_id = None
        self._hand_constraint_id = None
        self._joint_num = 7
        # Set the allowed self collision pair
        self._allowed_self_collision = CollisionPairSet(
            [[-1, 1], [0, 2], [2, 4], [4, 6], [8, 9], [7, 8], [7, 9]]
        )
        return

    def load(
        self,
        init_base_pose: Pose,
        init_joint_position: List[float],
        init_hand_position: float,
    ):
        self._init_base_pose = init_base_pose
        self._init_joint_position = init_joint_position
        self._init_hand_position = init_hand_position

        self._robot_id = p.loadURDF(
            self._robot_urdf.as_posix(),
            self._init_base_pose.position,
            self._init_base_pose.orientation_xyzw,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
            | p.URDF_USE_SELF_COLLISION,
        )
        self._hand_constraint_id = p.createConstraint(
            self._robot_id,
            8,
            self._robot_id,
            9,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(
            self._hand_constraint_id, gearRatio=-1, maxForce=50, erp=0.1
        )
        # Set collision pair to be ignoried
        for pair in [[-1, 1], [0, 2], [2, 4], [4, 6], [8, 9], [7, 8], [7, 9]]:
            p.setCollisionFilterPair(bodyUniqueIdA=self._robot_id, bodyUniqueIdB=self._robot_id, linkIndexA=pair[0] ,linkIndexB=pair[1], enableCollision=False)

        self.set_joint_position(self._init_joint_position)
        self.set_hand_position(self._init_hand_position)
        logging.info(
            f"Finish loading robot to pose: {self._init_base_pose}, with init joint position: {self._init_joint_position}"
        )
        return

    def reset(self):
        self.set_joint_position(self._init_joint_position)
        self.set_hand_position(self._init_hand_position)
        return

    def set_joint_velocity(self, velocity: List[float]):
        p.setJointMotorControlArray(
            self._robot_id,
            list(range(self._joint_num)),
            p.VELOCITY_CONTROL,
            targetVelocities=velocity,
        )
        return

    def set_joint_position(self, position: List[float]):
        p.setJointMotorControlArray(
            self._robot_id,
            list(range(self._joint_num)),
            p.POSITION_CONTROL,
            targetPositions=position,
        )
        return

    def set_hand_velocity(self, velocity: float):
        p.setJointMotorControlArray(
            self._robot_id,
            [8, 9],
            p.VELOCITY_CONTROL,
            targetVelocities=[velocity] * 2,
        )
        return

    def set_hand_position(self, position: float):
        p.setJointMotorControlArray(
            self._robot_id,
            [8, 9],
            p.POSITION_CONTROL,
            targetPositions=[0.1 - np.clip(position, 0.0, 0.1)] * 2,
        )
        return

    def get_joint_position(self) -> List[float]:
        infos = p.getJointStates(self._robot_id, list(range(self._joint_num)))
        return [info[0] for info in infos]

    def get_joint_velocity(self) -> List[float]:
        infos = p.getJointStates(self._robot_id, list(range(self._joint_num)))
        return [info[1] for info in infos]

    def get_tcp_pose(self) -> Pose:
        info = p.getLinkState(self._robot_id, 10)
        position = info[4]
        orientation = info[5]
        tcp_pose = Pose(
            position=position,
            orientation=[
                orientation[3],
                orientation[0],
                orientation[1],
                orientation[2],
            ],
        )
        return tcp_pose

    def get_tcp_velocity(self) -> List[float]:
        info = p.getLinkState(id, 10)
        linear_velocity = info[6]
        angular_velocity = info[7]
        return linear_velocity, angular_velocity

    def is_self_collide(self) -> bool:
        contact_points = p.getContactPoints(
            bodyA=self._robot_id, bodyB=self._robot_id
        )
        for contact_point in contact_points:
            if [contact_point[3],contact_point[4]] not in self._allowed_self_collision:
                return True
        else:
            return False

    def is_collide_with(self, id: int, ignore_hand:bool=True) -> bool:
        contact_points = p.getContactPoints(
            bodyA=self._robot_id, bodyB=id
        )
        # Find out allowed collision pair
        allowed_collision_pair = CollisionPairSet([[-1, -1]])
        if ignore_hand:
            allowed_collision_pair.add([7, -1])
            allowed_collision_pair.add([8, -1])
            allowed_collision_pair.add([9, -1])

        for contact_point in contact_points:
            if [contact_point[3],contact_point[4]] not in allowed_collision_pair:
                return True
        else:
            return False

    def is_touched_with(self, id: int, both_finger:bool=False) -> bool:
        left_finger_contact_points = p.getContactPoints(
            bodyA=self._robot_id, linkIndexA=8 , bodyB=id, 
        )
        right_finger_contact_points = p.getContactPoints(
            bodyA=self._robot_id, linkIndexA=9 , bodyB=id, 
        )
        is_touched = left_finger_contact_points and right_finger_contact_points if both_finger else left_finger_contact_points or right_finger_contact_points
        return is_touched