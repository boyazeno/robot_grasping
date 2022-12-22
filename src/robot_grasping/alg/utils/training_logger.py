from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, save_path:Path) -> None:
        self._save_path = save_path
        self._writer = SummaryWriter(log_dir=save_path)

    def add_losses(self, net_names:List[str], losses:List[torch.Tensor], step_num:int):
        for net_name, loss in zip(net_names, losses):
            name = net_name + '_loss'
            self._writer.add_scalar(tag=name, scalar_value=loss, global_step=step_num)
    
    def add_learning_rates(self, net_names:List[str], learning_rates:List[float], step_num:int):
        for net_name, learning_rate in zip(net_names, learning_rates):
            name = net_name + '_lr'
            self._writer.add_scalar(tag=name, scalar_value=learning_rate, global_step=step_num)

    def add_net(self, net:nn.Module):
        self._writer.add_graph(net)

    def __del__(self):
        self._writer.close()