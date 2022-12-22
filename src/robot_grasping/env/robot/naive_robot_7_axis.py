from pathlib import Path
from typing import List
import pybullet as p
import numpy as np
import logging
from robot_grasping.env.robot.robot_base import RobotBase
from robot_grasping.utils.types import Pose


class NaiveRobot7Axis(RobotBase):
    def __init__(self) -> None:
        super().__init__()
        self._robot_urdf = Path("/home/boya/noetic_ws/src/RL/robot_grasping/asset/urdf/naive_robot_7_axis.urdf")
        if not self._robot_urdf.exists():
            raise FileNotFoundError(f"Robot urdf not exist in path {self._robot_urdf}")

        self._init_base_pose = None
        self._init_joint_position = None
        self._init_hand_position = None
        self._robot_id = None
        self._hand_constraint_id = None
        self._joint_num = 7
        return
    

    def load(self, init_base_pose:Pose, init_joint_position:List[float], init_hand_position:float):
        self._init_base_pose = init_base_pose
        self._init_joint_position = init_joint_position
        self._init_hand_position = init_hand_position

        self._robot_id = p.loadURDF(self._robot_urdf.as_posix(), self._init_base_pose.position,self._init_base_pose.orientation_xyzw, useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self._hand_constraint_id = p.createConstraint( self._robot_id, 8,
                                self._robot_id, 9,
                                jointType=p.JOINT_GEAR,
                                jointAxis=[0, 1, 0],
                                parentFramePosition=[0, 0, 0],
                                childFramePosition=[0, 0, 0])
        p.changeConstraint(self._hand_constraint_id, gearRatio=-1, maxForce=50, erp=0.1)


        self.set_joint_position(self._init_joint_position)
        self.set_hand_position(init_hand_position)
        logging.info(f"Finish loading robot to pose: {self._init_base_pose}, with init joint position: {self._init_joint_position}")
        return


    def reset(self):
        self.set_joint_position(self._init_joint_position)
        self.set_hand_position(self._init_hand_position)
        return


    def set_joint_velocity(self, velocity:List[float]):
        p.setJointMotorControlArray(self._robot_id, list(range(self._joint_num)), p.VELOCITY_CONTROL, targetVelocities=velocity)
        return

    
    def set_joint_position(self, position:List[float]):
        p.setJointMotorControlArray(self._robot_id, list(range(self._joint_num)), p.POSITION_CONTROL, targetPositions=position)
        return


    def set_hand_velocity(self, velocity:float):
        p.setJointMotorControlArray(self._robot_id, [8,9], p.VELOCITY_CONTROL, targetVelocities=[velocity]*2)
        return


    def set_hand_position(self, position:float):
        p.setJointMotorControlArray(self._robot_id, [8,9], p.POSITION_CONTROL, targetPositions=[0.1-np.clip(position,0.0,0.1)]*2)
        return

    
    def get_joint_position(self)->List[float]:
        infos = p.getJointStates(self._robot_id, list(range(self._joint_num)))
        return [info[0] for info in infos]

    
    def get_joint_velocity(self)->List[float]:
        infos = p.getJointStates(self._robot_id, list(range(self._joint_num)))
        return [info[1] for info in infos]


    def get_tcp_pose(self)->Pose:
        info = p.getLinkState(self._robot_id,10)
        position = info[4]
        orientation = info[5]
        tcp_pose = Pose(position=position, orientation=[orientation[3],orientation[0],orientation[1],orientation[2]])
        return tcp_pose


    def get_tcp_velocity(self)->List[float]:
        info = p.getLinkState(id,10)
        linear_velocity = info[6]
        angular_velocity = info[7]
        return linear_velocity, angular_velocity