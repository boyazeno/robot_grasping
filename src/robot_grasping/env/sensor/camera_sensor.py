from robot_grasping.env.sensor.sensor_base import SensorBase
from robot_grasping.env.robot.robot_base import RobotBase
from robot_grasping.utils.types import Pose
from typing import Any, NoReturn, Dict, Tuple
import numpy as np
import pybullet as p
import pytransform3d.rotations as ptr
import pytransform3d.transformations as ptt


class CameraSensor(SensorBase):
    def __init__(self, name:str) -> None:
        super().__init__(name=name)
        self._camera_pose = Pose()


    def configure(self, config:Dict) -> NoReturn:
        self._return_rgb = config.get("return_rgb", True)
        self._return_depth = config.get("return_depth", True)
        self._return_mask = config.get("return_mask", True)
        self._return_pc = config.get("return_pc", True)
        self._camera_resolution = config.get("camera_resolution", (720,1280))
        self._camera_intrinsic = config.get("camera_intrinsic", np.array([925.1,0.,634.3,0.,925.1,366.4]).reshape(2,3))
        
        self._near = 0.01
        self._far = 5.0
        self._projection_matrix = self._intrinsic_2_projection(intrinsic=self._camera_intrinsic, width=self._camera_resolution[1], height=self._camera_resolution[0], far=self._far, near=self._near)

        flip_rot = ptr.active_matrix_from_extrinsic_euler_xyz(np.array([ np.pi, 0, 0]))
        self._cam_to_opengl = ptt.transform_from(
            flip_rot, np.array([0, 0, 0]), strict_check=False
        )

    def get_data(self)->Any:
        self._view_matrix = (self._cam_to_opengl@self._camera_pose.inv().to_numpy())
        imgs = p.getCameraImage(self._camera_resolution[1],
                          self._camera_resolution[0],
                          self._view_matrix.T.reshape(-1),
                          self._projection_matrix.T.reshape(-1),
                          #shadow=True,
                          renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb, depth, mask, pc = None, None, None, None
        # Get depth
        if self._return_depth:
            depth = self._get_depth(return_buffer = imgs)

        # Get rgb
        if self._return_rgb:
            rgb = self._get_rgb(return_buffer=imgs)

        # Get mask
        if self._return_mask:
            mask = self._get_mask(return_buffer=imgs)

        # Get pc
        if self._return_pc:
            pc = self._get_pc(return_buffer=imgs)
        return rgb, depth, mask, pc 


    def reset_pose(self, camera_pose : Pose):
        self._camera_pose = camera_pose
    
    @property
    def pose(self):
        return self._camera_pose

    def _get_depth(self, return_buffer:Tuple)->np.ndarray:
        depth_buffer = np.reshape(return_buffer[3], [self._camera_resolution[0], self._camera_resolution[1]])
        depth_buffer = 2.0 * depth_buffer - 1.0
        depth_buffer = (
            2.0
            * self._near
            * self._far
            / (self._far + self._near - depth_buffer * (self._far - self._near))
        )
        return depth_buffer


    def _get_rgb(self, return_buffer:Tuple)->np.ndarray:
        return return_buffer[2]


    def _get_mask(self, return_buffer:Tuple)->np.ndarray:
        return return_buffer[4]


    def _get_pc(self, return_buffer:np.ndarray)->np.ndarray:
        depth = np.reshape(return_buffer[3], [self._camera_resolution[0], self._camera_resolution[1]])
        tran_pix_world = np.linalg.inv(self._projection_matrix@self._view_matrix)
        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / depth.shape[0], -1:1:2 / depth.shape[1]]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 1.0-1e-4]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]
        return points


    def _intrinsic_2_projection(self, intrinsic:np.ndarray, width:float, height:float, far:float, near:float) -> np.ndarray:
        """See more https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet

        Parameters
        ----------
        intrinsic : np.ndarray
            Intrinsic matrix for the camera
        width : float
            Width of the image
        height : float
            Height of the image
        far : float
            Farthest visible distance
        near : float
            nearest visible distance

        Returns
        -------
        np.ndarray
            Projection matrix
        """
        # inspired from http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        s = intrinsic[0, 1]
        alpha = intrinsic[0, 0]
        beta = intrinsic[1, 1]
        x0 = intrinsic[0, 2]
        y0 = intrinsic[1, 2]
        y0 = height - y0
        A = near + far
        B = near * far
        persp = np.array(
            [[alpha, s, -x0, 0], [0, beta, -y0, 0], [0, 0, A, B], [0, 0, -1, 0]]
        )
        left, right, bottom, top = 0, width, 0, height
        tx = -(right + left) / (right - left)
        ty = -(top + bottom) / (top - bottom)
        tz = -(far + near) / (far - near)
        NDC = np.array(
            [
                [2 / (right - left), 0, 0, tx],
                [0, 2 / (top - bottom), 0, ty],
                [0, 0, -2 / (far - near), tz],
                [0, 0, 0, 1],
            ]
        )
        proj = NDC @ persp
        return proj


class FixedCameraSensor(CameraSensor):
    def __init__(self, name: str, pose:Pose) -> None:
        super().__init__(name)
        self.reset_pose(camera_pose=pose)


class WristCameraSensor(CameraSensor):
    def __init__(self, name: str, robot: RobotBase, transform_camera_2_tcp: Pose) -> None:
        super().__init__(name)
        self._robot = robot
        self._transform_camera_2_tcp = transform_camera_2_tcp
    
    def get_data(self) -> Any:
        tcp_pose = self._robot.get_tcp_pose()
        camera_pose = tcp_pose.dot(self._transform_camera_2_tcp)
        self.reset_pose(camera_pose=camera_pose)
        return super().get_data()
    