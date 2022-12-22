
from robot_grasping.utils.types import State, Action
import torch
import numpy as np
from copy import deepcopy

class DataConverter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def to_a(a:Action)-> torch.Tensor:
        a_np = deepcopy(a.robot_joint_velocity)
        a_np.append(a.robot_hand_velocity)
        a_ind_t = torch.as_tensor(a_np, dtype=int) + 5
        a_t = torch.zeros((8,11),dtype=float)
        a_t[torch.arange(a_ind_t.size(0)), a_ind_t] = 1.0
        return torch.unsqueeze(a_t.flatten(), dim=0).float()

    @staticmethod
    def to_r(r:float)-> torch.Tensor:
        return torch.unsqueeze(torch.as_tensor(r), dim=0).float()

    @staticmethod
    def to_s(s:State)-> torch.Tensor:
        global_rgb = s.visual_data["global_camera"]["rgb"]/255.0
        global_depth = np.clip(s.visual_data["global_camera"]["depth"], 0.0, 10.0) / 10.0
        wrist_rgb = s.visual_data["wrist_camera"]["rgb"]/255.0
        wrist_depth = np.clip(s.visual_data["wrist_camera"]["depth"], 0.0, 10.0) / 10.0
        imgs_np = np.concatenate([global_rgb[:,:,:3].transpose(2,0,1),global_depth[np.newaxis, :, :],wrist_rgb[:,:,:3].transpose(2,0,1),wrist_depth[np.newaxis, :, :]], axis=0 )
        return torch.unsqueeze(torch.from_numpy(imgs_np), dim=0).float()

    def to_action(a:torch.Tensor)->Action:
        a_np:np.ndarray = a.detach().numpy().reshape(8,11)
        a_value = a_np.argmax(axis=1) - 5 # (0-10) -> (-5,5)
        return Action(robot_joint_velocity=a_value[:7].tolist(), robot_hand_velocity=a_value[7])