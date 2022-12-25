from robot_grasping.env.robot.naive_robot_7_axis import NaiveRobot7Axis
from robot_grasping.utils.parameter_io import load_config_from_yaml
from robot_grasping.env.object_model import PrimitiveObjectModel
from robot_grasping.env.object_model import PrimitiveType
from robot_grasping.utils.types import Pose
from robot_grasping.utils.types import NameWithPose
from robot_grasping.env.env import Env
from robot_grasping.env.reward.naive_reward_assigner import NaiveRewardAssigner
from robot_grasping.alg.ddpg.ddpg import DDPG, DDPGConfig, DDPGTrainConfig
import time
import numpy as np
import logging
from pathlib import Path

# Config env
logging.info(f"#" * 80)
logging.info(f"Load env!")
robot = NaiveRobot7Axis()
env_config = load_config_from_yaml(
    file_path="/home/boya/noetic_ws/src/RL/robot_grasping/config/env/env_config.yaml"
)
env = Env(
    robot=robot,
    object_models=[
        PrimitiveObjectModel(
            name="box",
            type=PrimitiveType.Cube,
            size=[0.05, 0.1, 0.07],
            mass=0.5,
        )
    ],
    sensors=[],
    config=env_config,
)
robot_init_base_pose = Pose()
robot_init_joint_position = [
    0.0,
    0.0 * np.pi,
    0.0,
    0.75 * np.pi,
    0.0,
    0.25 * np.pi,
    0.0,
]
robot_init_hand_position = 0.08
object_init_poses = [
    NameWithPose(name="box", pose=Pose(position=[0.7, 0.0, 0.0]))
]
env.load(
    robot_init_base_pose=robot_init_base_pose,
    robot_init_joint_position=robot_init_joint_position,
    robot_init_hand_position=robot_init_hand_position,
    object_init_poses=object_init_poses,
)
reward_assigner = NaiveRewardAssigner(
    env=env,
    max_episode_steps=100,
    target_object_name="box",
    target_height=0.5,
    velocity_thresholds=(1e-1, 1e-2),
)
env.load_reward_assigner(reward_assigner=reward_assigner)
time.sleep(2.0)
logging.info(f"Load env finished!")
logging.info(f"#" * 80)

# Config DDPG
train_config = DDPGTrainConfig(
    epoch_num=100,
    episode_num=50,
    max_step_num=40,
    retrain=True,
    last_epoch_num=0,
    batch_size=8,
    save_path=Path("/media/boya/B67E7B8B7E7B42E3/Users/18486/ubuntu_file/robot_grasping/data/test"),
    save_epoch_interval=1,
    learning_rate=1e-4,
    scheduler_params={
        "decay_rate": 0.1,
        "delta_threshold": 1e-6,
        "epoch_step_num": 20,
        "tolerance_epoch_num": 5,
    },
)
DDPG_config = DDPGConfig(
    train_config=train_config,
    e_greedy_rate=0.1,
    r_decay_rate=0.1,
    q_net_model_name="SimpleQNet",
    policy_net_model_name="SimplePolicyNet",
    q_net_device="cuda:0",
    policy_net_device="cuda:0",
    input_image_shape= (256,144)
)
logging.info(f"#" * 80)
logging.info(f"Load DDPG!")
DDPG = DDPG(config=DDPG_config, env=env)
logging.info(f"Load DDPG finished!")
logging.info(f"#" * 80)
logging.info(f"Start training!")
train_summary = DDPG.train()
logging.info(f"#" * 80)
logging.info(f"Finish training!")
evaluation_summary = DDPG.evaluate()
