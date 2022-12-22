from typing import List, Dict, Tuple
import time
import numpy as np
import pybullet as p
import pybullet_data
from robot_grasping.utils.types import Pose
from robot_grasping.utils.types import NameWithPose
from robot_grasping.utils.types import Action
from robot_grasping.utils.types import State
from robot_grasping.env.robot.robot_base import RobotBase
from robot_grasping.utils.visualizer import Visualizer
from robot_grasping.env.object_model import ObjectModelBase
from robot_grasping.env.sensor.sensor_base import SensorBase
from robot_grasping.env.sensor.camera_sensor import CameraSensor
from robot_grasping.env.sensor.camera_sensor import FixedCameraSensor
from robot_grasping.env.sensor.camera_sensor import WristCameraSensor
import logging
import importlib


class Env:
    def __init__(
        self,
        robot: RobotBase,
        object_models: List[ObjectModelBase],
        sensors: List[SensorBase],
        config: Dict = {},
    ) -> None:
        self._config = config
        self._robot = robot
        self._object_models = {
            object_model.name: object_model for object_model in object_models
        }
        self._sensors = {sensor.name: sensor for sensor in sensors}
        self._cameras:List[CameraSensor] = [FixedCameraSensor(name="global_camera", pose=Pose(position=[0.,2.5,1.5]).rotate_local(x=np.pi*0.5, z=np.pi).rotate_local(x=-20.*np.pi/180.).rotate_global(z=-0.5*np.pi)),
                         WristCameraSensor(name="wrist_camera", robot=self._robot, transform_camera_2_tcp=Pose(position=[0.0, 0.1, -0.2]).rotate_local(z=np.pi).rotate_local(x=-15.*np.pi/180.))]
        self._loaded_object_names = set()
        self._time_step_size = int(
            self._config.get("time_step_size", 1.0) * 240
        )  # use default 1/240s as time increment
        self._use_gui = self._config.get("use_gui", True)
        self._visualize = self._config.get("visualize", False)
        self._executed_step = 0
        self._visualizer = Visualizer()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        logging.info(f"Exit pybullet environment!")
        p.disconnect()

    def load(
        self,
        robot_init_base_pose: Pose,
        robot_init_joint_position: List[float],
        robot_init_hand_position: float,
        object_init_poses: List[NameWithPose],
    ):
        # Load environment
        if self._use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, 0)
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(
            self._config.get("use_real_time_simulation", False)
        )
        self._time_step = 0

        # Load robot and object model
        self._robot.load(
            init_base_pose=robot_init_base_pose,
            init_joint_position=robot_init_joint_position,
            init_hand_position=robot_init_hand_position,
        )
        for object_init_pose in object_init_poses:
            self._object_models[object_init_pose.name].load(
                object_init_pose.pose
            )
            self._loaded_object_names.add(object_init_pose.name)

        # Configure sensors
        camera_config = {
            "return_rgb": True,
            "return_depth": True,
            "return_mask": False,
            "return_pc": False,
        }
        for camera in self._cameras:
            camera.configure(config=camera_config)

    def load_reward_assigner(self, reward_assigner):
        # Configure reward
        self._reward_assigner = reward_assigner 

    def reset(self):
        self._executed_step = 0
        self._robot.reset()
        for object_name in self._loaded_object_names:
            self._object_models[object_name].reset()

    def execute_action(self, action: Action) -> Tuple[float, State]:
        # Execute action
        if action is not None:
            robot_joint_velocity, robot_hand_velocity = self._convert_discret_action_2_real_action(action)
            self._robot.set_joint_velocity(robot_joint_velocity)
            self._robot.set_hand_velocity(robot_hand_velocity)

        for _ in range(self._time_step_size):
            p.stepSimulation()
        self._executed_step += 1

        # Get next state & reward
        next_state = self.get_state()
        reward = self.get_reward()

        return reward, next_state
    
    def get_executed_step(self)->int:
        return 0 if self._executed_step is None else self._executed_step

    def get_reward(self) -> float:
        reward = self._reward_assigner.get_reward()
        return reward

    def get_state(self) -> State:
        joint_value = self._robot.get_joint_position()
        object_model_infos = {
            name: object_model.get_pose()
            for name, object_model in self._object_models.items()
        }
        visual_data = self._get_visual_data()
        is_terminated = self._is_terminated()
        return State(
            joint_value=joint_value,
            object_model_infos=object_model_infos,
            visual_data=visual_data,
            is_terminated=is_terminated,
        )

    def _is_terminated(self) -> bool:
        return self._reward_assigner.is_terminated()

    def _get_visual_data(self) -> Dict:
        #self._camera.reset_pose(camera_pose=Pose())
        visual_data={}
        for camera in self._cameras:
            rgb, depth, mask, pc = camera.get_data()
            visual_data[camera.name] = {"rgb": rgb,
                                        "depth": depth,
                                        "mask": mask,
                                        "pc": pc,}
            # Debug
            if self._visualize:
                self._visualizer.add_transformation(transformations=[camera.pose.to_numpy()],size=0.5)
                self._visualizer.add_image(img=rgb)
                self._visualizer.add_image(img=depth)
                random_idx = np.arange(pc.shape[0])
                np.random.shuffle(random_idx)
                self._visualizer.add_pc(pc=pc[random_idx[:10000]], color=np.array([0.5,0.5,0.0]), size=0.005)
                self._visualizer.remove_all()
        return visual_data

    def get_human_action(self) -> Action:
        data = input("input action:")
        print(data)
        # Parser data
        if len(data) != 8 * 2:
            print("Wrong input format! Should be (+/-)(0-5)*8")
            return self.get_human_action()
        robot_joint_velocity = [
            int(data[2*i : 2*i + 2])
            for i in range(len(data) // 2 - 1)
        ]
        robot_hand_velocity = int(data[-2:])
        action = Action(
            robot_joint_velocity=robot_joint_velocity,
            robot_hand_velocity=robot_hand_velocity,
        )
        return self._convert_discret_action_2_real_action(action)

    def _convert_discret_action_2_real_action(self, action:Action):
        joint_amplitude = self._config.get("joint_amplitude", 0.1)
        finger_amplitude = self._config.get("finger_amplitude", 0.01)
        robot_joint_velocity = [d * joint_amplitude / 5.0  for d in action.robot_joint_velocity]
        robot_hand_velocity = action.robot_hand_velocity * finger_amplitude / 5.0
        return robot_joint_velocity, robot_hand_velocity
