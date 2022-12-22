from robot_grasping.env.reward.base_reward_assigner import BaseRewardAssigner
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


    def get_reward(self, ) -> float:
        reward = 0.0
        # 1. Final reward:
        # 1.1 height/speed reward: 
        if self._is_final_condition_fulfilled():
            reward += 1e2

        # 2. Intemediate reward:
        # 2.1 heigh reward:
        height_diff = np.abs(self._target_height - self._get_object_height(object_name=self._target_object_name))
        reward += utils.value_decrease_derivative_decrease_n(height_diff, minv=0., maxv=self._target_height)

        # 2.2 speed reward:
        linear_speed, angle_speed = self._get_object_velocity(object_name=self._target_object_name)
        linear_speed_diff, angle_speed_diff = np.maximum(np.abs(linear_speed)-self._velocity_thresholds[0],0.0), np.maximum(np.abs(angle_speed)-self._velocity_thresholds[1],0.0)
        reward += utils.value_decrease_derivative_decrease_n(linear_speed_diff, minv=0., maxv=1.0)
        reward += utils.value_decrease_derivative_decrease_n(angle_speed_diff, minv=0., maxv=0.5)

        # 3. Time reward:
        reward += -1.*self._env.get_executed_step()
        return reward


    def is_terminated(self) -> bool:
        is_terminated = False
        if self._env.get_executed_step() >= self._max_episode_steps:
            is_terminated = True
        elif self._is_final_condition_fulfilled():
            logging.info(f"[Reward Assigner] Final confition fulfilled!")
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

    def _get_object_height(self, object_name:str)->float:
        object_pose = self._env._object_models[object_name].get_pose()
        return object_pose.position[2]

    def _get_object_velocity(self, object_name:str)->Tuple[float,float]:
        object_linear_velocity, object_angle_velocity = self._env._object_models[object_name].get_velocity()
        return np.linalg.norm(np.array(object_linear_velocity)), np.linalg.norm(np.array(object_angle_velocity))