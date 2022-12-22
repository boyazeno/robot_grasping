"""Different learning rate schedulers
"""
import torch
import logging
from typing import List
from robot_grasping.alg.utils.training_logger import TrainingLogger


class StepLearningRateScheduler:
    def __init__(
        self,
        optimizers: List[torch.optim.Optimizer],
        training_logger: TrainingLogger,
        decay_rate: float = 1e-1,
        delta_threshold: float = 1e-6,
        epoch_step_num: int = 10,
        tolerance_epoch_num: int = 5,
    ) -> None:
        self._optimizers = optimizers
        self._decay_rate = decay_rate
        self._delta_threshold = delta_threshold
        self._epoch_step_num = epoch_step_num
        self._tolerance_epoch_num = tolerance_epoch_num
        self._training_logger = training_logger
        self.reset(len(self._optimizers))

    def update(self, losses: List[torch.Tensor], epoch_num: int):
        for optimizer_id, loss in  enumerate(losses):
            if (
                epoch_num - self._last_decay_epoch_num[optimizer_id] > self._epoch_step_num
                and torch.abs(loss - self._last_loss[optimizer_id]) < self._delta_threshold
            ):
                self._experienced_tolerance_epoch_num[optimizer_id] += 1
                if (
                    self._experienced_tolerance_epoch_num[optimizer_id]
                    >= self._tolerance_epoch_num
                ):
                    self._experienced_tolerance_epoch_num[optimizer_id] = 0
                    self._last_decay_epoch_num[optimizer_id] = epoch_num
                    # Have to decrease the lr
                    self._optimizers[optimizer_id].param_groups[0]["lr"] *= self._decay_rate
                    logging.warning(
                        f"[Learning Rate SCheduler]: Reduce lr to {self._optimizers[optimizer_id].param_groups[0]['lr']}."
                    )
            else:
                self._experienced_tolerance_epoch_num[optimizer_id] = 0

            self._last_loss[optimizer_id] = loss
        return [self._optimizers[optimizer_id].param_groups[0]['lr'] for optimizer_id in range(len(losses))]

    def reset(self, optimizer_num: int):
        self._last_loss = [torch.nan for _ in range(optimizer_num)]
        self._last_decay_epoch_num = [0 for _ in range(optimizer_num)]
        self._experienced_tolerance_epoch_num = [0 for _ in range(optimizer_num)]

    def print(self):
        params = {
            "optimizer": self._optimizer,
            "decay_rate": self._decay_rate,
            "delta_threshold": self._delta_threshold,
            "epoch_step_num": self._epoch_step_num,
            "tolerance_epoch_num": self._tolerance_epoch_num,
        }
        logging.info(
            f"[Learning Rate SCheduler]: Init with parameters: {params}"
        )
