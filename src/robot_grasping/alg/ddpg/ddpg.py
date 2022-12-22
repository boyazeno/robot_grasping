import sys
import importlib
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, List, Dict
from pathlib import Path
import shutil
import pickle
from tqdm import tqdm
import logging
from robot_grasping.utils.types import State, Action
from robot_grasping.env.env import Env
from robot_grasping.utils.parameter_io import ParameterIO
from robot_grasping.alg.ddpg.model import ModelBase
from robot_grasping.alg.utils.learning_rate_scheduler import (
    StepLearningRateScheduler,
)
from robot_grasping.alg.utils.data_converter import DataConverter
from robot_grasping.alg.utils.training_logger import TrainingLogger

import torch
import torch.nn as nn


@dataclass
class DDPGTrainConfig:
    epoch_num: int
    episode_num: int
    max_step_num: int
    retrain: bool
    last_epoch_num: int
    batch_size: int
    save_path: Path
    save_epoch_interval: int
    learning_rate: float
    scheduler_params: Dict


@dataclass
class DDPGConfig:
    train_config: DDPGTrainConfig
    e_greedy_rate: float = 0.05
    r_decay_rate: float = 0.1
    q_net_model_name: str = "SimpleQNet"
    policy_net_model_name: str = "SimplePolicyNet"


@dataclass
class DDPGTrainSummary:
    max_step_num: int
    episode_num: int
    average_reward: float
    average_step_num: float
    success_rate: float


@dataclass
class DDPGEvaluationSummary:
    max_step_num: int
    episode_num: int
    average_reward: float
    average_step_num: float
    success_rate: float


@dataclass
class DDPGData:
    s: State
    s_n: State
    r: float
    a: Action


@dataclass
class DDPGTrainingBatchData:
    s: torch.Tensor
    s_n: torch.Tensor
    r: torch.Tensor
    a: torch.Tensor
    is_terminate: List[bool]

    @staticmethod
    def from_data(data_batch: List[DDPGData]):
        a_batch = torch.concat([DataConverter.to_a(data.a) for data in data_batch],dim=0)
        r_batch = torch.concat([DataConverter.to_r(data.r) for data in data_batch], dim=0)
        s_batch = torch.concat([DataConverter.to_s(data.s) for data in data_batch], dim=0)
        s_n_batch = torch.concat([DataConverter.to_s(data.s_n) for data in data_batch], dim=0)  
        is_terminate_batch = [data.s_n.is_terminated for data in data_batch] 
        return DDPGTrainingBatchData(a=a_batch, r=r_batch, s=s_batch, s_n=s_n_batch, is_terminate=is_terminate_batch)


class DDPGDatabase:
    def __init__(self, save_path: Path, max_buffer_size: float = 1024) -> None:
        self._save_path = save_path
        self._max_buffer_size = max_buffer_size
        self._buffer = []
        self.reset()

    def add(self, data: DDPGData):
        if self._is_buffer_full():
            self._save_to_file(data)
        else:
            self._buffer.append(data)

    def get(self, id: int) -> DDPGData:
        if id < self._get_size_in_buffer():
            return self._load_from_buffer(id)
        else:
            return self._load_from_file(id)

    def get_batch(self, batch_size: int) -> Tuple[DDPGData]:
        # TODO sample a batch of data and return
        # Random get idx
        random_indices = np.random.randint(
            low=0, high=self.len(), size=batch_size
        ).tolist()
        batch_data = [deepcopy(self.get(id=id)) for id in random_indices]
        return batch_data

    def reset(self):
        self._buffer = []
        shutil.rmtree(self._save_path, ignore_errors=True)
        self._save_path.mkdir(parents=True, exist_ok=True)

    def len(self) -> int:
        return self._get_size_in_file() + self._get_size_in_buffer()

    def _save_to_file(self, data: DDPGData):
        exist_file_num = self.len()
        # database start from 0
        file_name = self._save_path / str(exist_file_num)
        with file_name.open("wb") as f:
            pickle.dump(data, f)

    def _load_from_buffer(self, id: int) -> DDPGData:
        if id >= self._get_size_in_buffer():
            raise KeyError(f"Data with id {id} not exist in buffer!")
        return self._buffer[id]

    def _load_from_file(self, id: int) -> DDPGData:
        exist_file_names = set(self._save_path.iterdir())
        if str(id) not in exist_file_names:
            raise KeyError(f"Data with id {id} not exist in file!")
        # database start from 0
        file_name = self._save_path / str(id)
        with file_name.open("rb") as f:
            data = pickle.load(f)
        return data

    def _is_buffer_full(self) -> bool:
        return (
            sys.getsizeof(self._buffer) / (1024.0**2) >= self._max_buffer_size
        )

    def _get_size_in_file(self) -> int:
        return len(list(self._save_path.iterdir()))

    def _get_size_in_buffer(self) -> int:
        return len(self._buffer)


class DDPG:
    def __init__(self, config: DDPGConfig, env: Env) -> None:
        self._config = config
        self._env = env

    def _get_param_file_name(self, net_id: str, net_type: str) -> str:
        param_file_name = self._config.train_config.save_path / (
            net_type
            + "_"
            + str(net_id)
            + "_epoch_"
            + str(self._config.train_config.last_epoch_num)
        )
        return param_file_name

    def train(self):
        self._training_logger = TrainingLogger(self._config.train_config.save_path)

        # Prepare model
        QNetNetworkModel = self._get_model(self._config.q_net_model_name)
        PolicyNetNetworkModel = self._get_model(self._config.policy_net_model_name)
        self._q_net_1: nn.Module = QNetNetworkModel()
        self._q_net_2: nn.Module = QNetNetworkModel()
        self._policy_net_1: nn.Module = PolicyNetNetworkModel()
        self._policy_net_2: nn.Module = PolicyNetNetworkModel()
        self._database = DDPGDatabase(
            save_path=self._config.train_config.save_path / "database",
            max_buffer_size=1024,
        )
        # Load previous epoch
        if not self._config.train_config.retrain:
            self._q_net_1.load_state_dict(
                torch.load(self._get_param_file_name(1, "q_net"))
            )
            self._q_net_2.load_state_dict(
                torch.load(self._get_param_file_name(2, "q_net"))
            )
            self._policy_net_1.load_state_dict(
                torch.load(self._get_param_file_name(1, "policy_net"))
            )
            self._policy_net_2.load_state_dict(
                torch.load(self._get_param_file_name(2, "policy_net"))
            )

        # Init optimizer, scheduler
        self._optimizer_q_net_1 = torch.optim.AdamW(
            self._q_net_1.parameters(), lr=self._config.train_config.learning_rate,
        )  # TODO
        self._optimizer_q_net_2 = torch.optim.AdamW(
            self._q_net_2.parameters(), lr=self._config.train_config.learning_rate,
        )  # TODO
        self._optimizer_policy_net_1 = torch.optim.AdamW(
            self._policy_net_1.parameters(), lr=self._config.train_config.learning_rate,
        )  # TODO
        self._optimizer_policy_net_2 = torch.optim.AdamW(
            self._policy_net_2.parameters(), lr=self._config.train_config.learning_rate,
        )  # TODO
        self._scheduler = StepLearningRateScheduler(
            [
                self._optimizer_q_net_1,
                self._optimizer_policy_net_2,
                self._optimizer_q_net_2,
                self._optimizer_policy_net_2,
            ],
            decay_rate=self._config.train_config.scheduler_params["decay_rate"],
            delta_threshold=self._config.train_config.scheduler_params[
                "delta_threshold"
            ],
            epoch_step_num=self._config.train_config.scheduler_params[
                "epoch_step_num"
            ],
            tolerance_epoch_num=self._config.train_config.scheduler_params[
                "tolerance_epoch_num"
            ],
            training_logger = self._training_logger,
        )

        global_steps = 0
        for train_epoch in range(self._config.train_config.epoch_num):
            logging.info(f"Start epoch {train_epoch}")

            average_losses = [0.0 for _ in range(4)]
            for train_episode in tqdm(
                range(self._config.train_config.episode_num)
            ):
                state = self._env.get_state()
                action = DataConverter.to_action(self._policy_net_1(DataConverter.to_s(state)))  # TODO get action
                current_step_count = 0

                for train_step in range(self._config.train_config.max_step_num):
                    global_steps += 1

                    # Get next state and reward
                    reward, state_n = self._env.execute_action(action=action)

                    # Update database
                    self._database.add(
                        DDPGData(s=state, s_n=state_n, r=reward, a=action)
                    )

                    if state_n.is_terminated:
                        break

                    # Sample mini batch
                    mini_batch = self._database.get_batch(
                        batch_size=self._config.train_config.batch_size
                    )
                    # Convert to training torch format
                    mini_batch = DDPGTrainingBatchData.from_data(mini_batch)

                    # Update parameter
                    episode_losses = self._update_parameters(
                        mini_batch=mini_batch
                    )
                    average_losses = [
                        average_loss
                        + (episode_loss - average_loss) / (train_episode + 1)
                        for average_loss, episode_loss in zip(
                            average_losses, episode_losses
                        )
                    ]
                    
                    self._training_logger.add_losses(["loss_q_net_1","loss_q_net_2","loss_policy_net_1","loss_policy_net_2"], episode_losses)

            current_learning_rates = self._scheduler.update(losses=average_losses, epoch_num=train_epoch)
            self._training_logger.add_learning_rates(["loss_q_net_1","loss_q_net_2","loss_policy_net_1","loss_policy_net_2"], current_learning_rates)
        pass

    def evaluate(self):
        logging.warning(f"Jump over evaluation for now!")
        pass

    def _update_parameters(
        self, mini_batch: DDPGTrainingBatchData
    ) -> torch.Tensor:
        # Caluculate cost
        loss_q_net_1 = self._calculate_q_net_loss(
            q_net=self._q_net_1,
            target_net=self._q_net_2,
            policy_net=self._policy_net_1,
            mini_batch=mini_batch,
        )
        loss_q_net_2 = self._calculate_q_net_loss(
            q_net=self._q_net_2,
            target_net=self._q_net_1,
            policy_net=self._policy_net_2,
            mini_batch=mini_batch,
        )
        loss_policy_net_1 = self._calculate_policy_net_loss(
            policy_net=self._policy_net_1,
            q_net=self._q_net_1,
            mini_batch=mini_batch,
        )
        loss_policy_net_2 = self._calculate_policy_net_loss(
            policy_net=self._policy_net_2,
            q_net=self._q_net_2,
            mini_batch=mini_batch,
        )

        # Update net weight
        self._optimizer_q_net_1.zero_grad()
        self._optimizer_q_net_2.zero_grad()
        self._optimizer_policy_net_1.zero_grad()
        self._optimizer_policy_net_2.zero_grad()

        loss_q_net_1.backward()
        loss_q_net_2.backward()
        loss_policy_net_1.backward()
        loss_policy_net_2.backward()

        self._optimizer_q_net_1.step()
        self._optimizer_q_net_2.step()
        self._optimizer_policy_net_1.step()
        self._optimizer_policy_net_2.step()

        return [
            loss_q_net_1,
            loss_q_net_2,
            loss_policy_net_1,
            loss_policy_net_2,
        ]

    def _calculate_q_net_loss(
        self,
        q_net: nn.Module,
        target_net: nn.Module,
        policy_net: nn.Module,
        mini_batch: DDPGTrainingBatchData,
    ) -> torch.Tensor:
        """Calculate loss for a mini batch.

        Parameters
        ----------
        mini_batch : DDPGTrainingBatchData
            Mini batch

        Returns
        -------
        torch.Tensor
            Loss
        """
        # for data in mini_batch:
        #     with torch.no_grad():
        #         if data.s_n is None:
        #             target_q_value = torch.as_tensor(0.0).float()
        #         else:
        #             target_q_value = target_net(data.s_n, policy_net(data.s_n))
        #     total_loss += torch.pow(
        #         data.r
        #         + self._config.r_decay_rate * target_q_value
        #         - q_net(data.s, data.a),
        #         torch.as_tensor(2.0).float(),
        #     )
        # loss = total_loss / len(mini_batch)
        with torch.no_grad():
            target_q_value = target_net(mini_batch.s_n, policy_net(mini_batch.s_n))
            target_q_value = torch.mul(target_q_value, torch.logical_not(torch.as_tensor(mini_batch.is_terminate)).float()) # ignore the target q value for those final state
        q_value = q_net(mini_batch.s, mini_batch.a)
        loss = torch.pow( mini_batch.r + self._config.r_decay_rate*target_q_value - q_value, torch.as_tensor(2.0).float())
        loss = torch.sum(loss)/mini_batch.s.shape[0]
        return loss

    def _calculate_policy_net_loss(
        self,
        policy_net: nn.Module,
        q_net: nn.Module,
        mini_batch: DDPGTrainingBatchData,
    ) -> torch.Tensor:
        q_net.requires_grad_(requires_grad=False)
        # for data in mini_batch:
        #     q_value = q_net(data.s, policy_net(data.s))
        #     total_loss += q_value
        # loss = total_loss / len(mini_batch)
        q_value = q_net(mini_batch.s, policy_net(mini_batch.s))
        q_net.requires_grad_(requires_grad=True)
        loss = torch.sum(q_value)/mini_batch.s.shape[0]
        return loss

    def _get_model(self, model_name: str) -> nn.Module:
        return importlib.import_module(
            "robot_grasping.alg.ddpg.model"
        ).__dict__[model_name]


if __name__ == "__main__":
    pass