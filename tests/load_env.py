from typing import List, Dict
import time
import numpy as np
from robot_grasping.env.robot.naive_robot_7_axis import NaiveRobot7Axis
from robot_grasping.env.env import Env
from robot_grasping.utils.types import Pose, NameWithPose
from robot_grasping.env.object_model import PrimitiveObjectModel, PrimitiveType
from robot_grasping.utils.types import Action
from robot_grasping.utils.parameter_io import load_config_from_yaml
from robot_grasping.env.reward.naive_reward_assigner import NaiveRewardAssigner
import logging
import importlib

if __name__=="__main__":
    logging.info(f"Load env!")
    robot = NaiveRobot7Axis()
    env_config = load_config_from_yaml(file_path="/home/boya/noetic_ws/src/RL/robot_grasping/config/env/env_config.yaml")

    #env = Env(robot=robot, object_models=[], sensors=[])
    env = Env(robot=robot, object_models=[PrimitiveObjectModel(name="box", type=PrimitiveType.Cube, size=[0.05, 0.1, 0.07], mass=0.5)], sensors=[],config=env_config)
    robot_init_base_pose = Pose()
    robot_init_joint_position = [0.0, -0.25*np.pi, 0.0, 0.75*np.pi, 0.0, 0.25*np.pi, 0.0]
    robot_init_hand_position = 0.08
    #object_init_poses = []
    object_init_poses = [NameWithPose(name="box",pose=Pose(position=[0.4,0.,0.]))]
    env.load(robot_init_base_pose=robot_init_base_pose, robot_init_joint_position=robot_init_joint_position, robot_init_hand_position=robot_init_hand_position, object_init_poses=object_init_poses)
    reward_assigner = NaiveRewardAssigner(env=env, max_episode_steps=100, target_object_name="box", target_height = 0.5, velocity_thresholds = (1e-1,1e-2))
    env.load_reward_assigner(reward_assigner=reward_assigner)
    time.sleep(2.0)
    action = Action(robot_joint_velocity=[0.01]*7,robot_hand_velocity=0.01)
    _, s = env.execute_action(action=None)
    logging.info(f"Get rgb: {s.visual_data['global_camera']['rgb'].shape}, {s.visual_data['global_camera']['rgb'].max()}, {s.visual_data['global_camera']['rgb'].min()}")
    logging.info(f"Get depth: {s.visual_data['global_camera']['depth'].shape}, {s.visual_data['global_camera']['depth'].max()}, {s.visual_data['global_camera']['depth'].min()}")
    while True:
        action = env.get_human_action()
        env.execute_action(action=action)