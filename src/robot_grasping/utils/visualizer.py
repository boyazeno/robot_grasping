import numpy as np
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import open3d as o3d
import logging

class Visualizer:
    def __init__(self, use_bullet:bool = True) -> None:
        self._ids = []
        self._use_bullet = use_bullet
        if self._use_bullet:
            import pybullet as p


    def add_pc(self, pc:Optional[np.ndarray], color:Optional[np.ndarray], size:float = 0.002):
        if pc is None:
            logging.info(f"No pc as input.")
        assert len(pc.shape) == 2 and pc.shape[-1] == 3
        if len(color.shape) == 1:
            color = np.tile(color.reshape(1,-1),(pc.shape[0],1))
        elif len(color.shape) == 2 and color.shape[0] == pc.shape[0]:
            pass
        else:
            raise ValueError(f"Input color size and pc should match, or color should be array with len 3")
        if self._use_bullet:
            self._ids.append(p.addUserDebugPoints(pc, color, size, 25.0))
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            o3d.visualization.draw_geometries([pcd])
        pass


    def add_transformation(self, transformations:Any, size:float = 0.05):
        tfs = None
        if isinstance(transformations, list):
            tfs = np.stack(transformations)
        elif isinstance(transformations, np.ndarray) and len(transformations.shape) >= 2 and transformations[-1] == 4 and transformations[-2] == 4:
            tfs = transformations.reshape(-1,3,3)
        else:
            raise TypeError(f"Input transformation is not supported!")
        

        for i in range(tfs.shape[0]):
            origin = tfs[i][:3,3]
            x = origin + tfs[i][:3,0] * size
            y = origin + tfs[i][:3,1] * size
            z = origin + tfs[i][:3,2] * size
            self._ids.append(p.addUserDebugLine(origin, x, [1.0,0.0,0.0]))
            self._ids.append(p.addUserDebugLine(origin, y, [0.0,1.0,0.0]))
            self._ids.append(p.addUserDebugLine(origin, z, [0.0,0.0,1.0]))

    def add_image(self, img:Optional[np.ndarray]):
        if img is None:
            logging.info(f"No image as input.")
            return
        plt.imshow(img)
        plt.show()

    def remove_all(self):
        p.removeAllUserDebugItems()

