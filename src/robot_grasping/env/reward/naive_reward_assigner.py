from robot_grasping.env.reward.base_reward_assigner import BaseRewardAssigner
from robot_grasping.utils.types import State
import robot_grasping.env.reward.utils as utils
from robot_grasping.env.env import Env
import numpy as np
from typing import Tuple
import logging

class NaiveRewardAssigner(BaseRewardAssigner):
    def __init__(self, env:Env, max_episode_steps:int, target_object_name:str, target_height:float = 0.5, velocity_thresholds:Tuple[float,float] = (1e-1,1e-2)) -> None:
        """Let the robot pick the block and keep it over 0.5m.

        Parameters
        ----------
        env : Env
            The simulation environment.
        """
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._target_object_name = target_object_name
        self._target_height = target_height
        self._velocity_thresholds = velocity_thresholds


    def get_reward(self, state:State) -> float:
        reward = 0.0
        # 1. Final reward:
        # 1.1 height/speed reward: 
        if self._is_final_condition_fulfilled():
            reward += 1e2
            logging.info(f"[R] final_reward:{1e2}")

        # 2. Intermediate reward:
        # 2.1 height reward:
        height_diff = np.abs(self._target_height - self._get_object_height(object_name=self._target_object_name))
        height_reward = utils.value_decrease_derivative_decrease_n(height_diff, minv=0., maxv=self._target_height)
        reward += height_reward
        logging.info(f"[R] height_reward:{height_reward}  height_diff:{height_diff}")

        # 2.2 speed reward:
        linear_speed, angle_speed = self._get_object_velocity(object_name=self._target_object_name)
        linear_speed_diff, angle_speed_diff = np.maximum(np.abs(linear_speed)-self._velocity_thresholds[0],0.0), np.maximum(np.abs(angle_speed)-self._velocity_thresholds[1],0.0)
        linear_speed_reward = utils.value_decrease_derivative_decrease_n(linear_speed_diff, minv=0., maxv=2.0)*0.1
        angle_speed_reward = utils.value_decrease_derivative_decrease_n(angle_speed_diff, minv=0., maxv=1.0)*0.1
        reward += linear_speed_reward 
        reward += angle_speed_reward 
        logging.info(f"[R] linear_speed:{linear_speed_reward}  linear_speed:{linear_speed}")
        logging.info(f"[R] angle_speed:{angle_speed_reward}  angle_speed:{angle_speed}")

        # 2.3 tcp object distance
        distance = self._get_object_tcp_distance(object_name=self._target_object_name)
        distance_reward = utils.value_decrease_derivative_decrease_n(distance, minv=0., maxv=3.0)*10
        reward += distance_reward
        logging.info(f"[R] distance_reward:{distance_reward}  distance:{distance}")

        # 2.4 Touch object:
        if not self._env._robot.is_touched_with(id=self._env._object_models[self._target_object_name]._object_id, both_finger = False):
            logging.info(f"[R] touch_reward:{-1e-1}")
            reward += -1e-1

        # 2.5 See object:
        if not self._is_object_visible(object_name=self._target_object_name, state=state):
            logging.info(f"[R] see_object_reward:{-1e-1}")
            reward += -1e-1

        # 3. Time reward:
        reward += -0.1
        logging.info(f"[R] time_reward:{-0.1}")

        # 4. Collision penalty
        if self._env._robot.is_self_collide() or self._env._robot.is_collide_with(self._env._ground_id):
            reward += -1e2
            logging.info(f"[R] collision_reward:{-1e2}")

        return reward


    def is_terminated(self) -> bool:
        is_terminated = False
        if self._env.get_executed_step() >= self._max_episode_steps:
            logging.info(f"[Reward Assigner] Max step size condition fulfilled!")
            is_terminated = True
        elif self._is_final_condition_fulfilled():
            logging.info(f"[Reward Assigner] Final condition fulfilled!")
            is_terminated = True
        elif self._env._robot.is_self_collide() or self._env._robot.is_collide_with(id=self._env._ground_id, ignore_hand=True):
            logging.info(f"[Reward Assigner] Collision condition fulfilled!")
            is_terminated = True
        return is_terminated


    def _is_final_condition_fulfilled(self)->bool:
        is_final_condition_fulfilled = False
        # Get data
        object_height = self._get_object_height(object_name=self._target_object_name)
        object_linear_velocity, object_angle_velocity = self._get_object_velocity(object_name=self._target_object_name)
        # Judge
        if object_height > self._target_height and object_linear_velocity < self._velocity_thresholds[0] and object_angle_velocity < self._velocity_thresholds[1]:
            is_final_condition_fulfilled = True
        return is_final_condition_fulfilled 

    def _is_object_visible(self, object_name:str, state:State)->bool:
        return self._env._object_models[object_name]._object_id in np.unique(state.visual_data["wrist_camera"]["mask"]).tolist()

    def _get_object_height(self, object_name:str)->float:
        object_pose = self._env._object_models[object_name].get_pose()
        return object_pose.position[2]

    def _get_object_velocity(self, object_name:str)->Tuple[float,float]:
        object_linear_velocity, object_angle_velocity = self._env._object_models[object_name].get_velocity()
        return np.linalg.norm(np.array(object_linear_velocity)), np.linalg.norm(np.array(object_angle_velocity))

    def _get_object_tcp_distance(self, object_name:str)->float:
        object_pose = self._env._object_models[object_name].get_pose()
        tcp_pose = self._env._robot.get_tcp_pose()
        return np.linalg.norm(np.array(tcp_pose.position) - np.array(object_pose.position))