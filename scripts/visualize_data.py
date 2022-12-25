from typing import List, Dict
import numpy as np
from robot_grasping.alg.ddpg.ddpg import DDPGData
from robot_grasping.utils.visualizer import Visualizer
import logging
from pathlib import Path
import pickle

def main(id:int):
    save_path=Path("/media/boya/B67E7B8B7E7B42E3/Users/18486/ubuntu_file/robot_grasping/data/test/database")
    exist_file_names = set(i.name for i in save_path.iterdir())
    if str(id) not in exist_file_names:
        raise KeyError(f"Data with id {id} not exist in file!")
    # database start from 0
    file_name = save_path / str(id)
    with file_name.open("rb") as f:
        data:DDPGData = pickle.load(f)

    visualizer = Visualizer(use_bullet=False)
    logging.info(f"Visualizer rgb from global!")
    visualizer.add_image(data.s.visual_data["global_camera"]["rgb"].astype("int"))
    logging.info(f"Visualizer depth from global!")
    visualizer.add_image(data.s.visual_data["global_camera"]["depth"])
    logging.info(f"Visualizer mask from global!")
    visualizer.add_image(data.s_n.visual_data["global_camera"]["mask"])
    logging.info(f"Visualizer depth from wrist!")
    visualizer.add_image(data.s_n.visual_data["wrist_camera"]["rgb"].astype("int"))
    logging.info(f"Visualizer depth from wrist!")
    visualizer.add_image(data.s_n.visual_data["wrist_camera"]["depth"])
    logging.info(f"Visualizer mask from wrist!")
    visualizer.add_image(data.s_n.visual_data["wrist_camera"]["mask"])


if __name__ == "__main__":
    logging.info(f"Visualizer data:")
    main(5)
    