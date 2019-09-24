from app.worker.base_worker.trainer import Trainer
from app.data.data_structure.transition import Transition
from app.util.processing.processing import Processing
from itertools import count
from app.config.config_factory import ConfigFactory
from app.util.logger.logger import Logger
from app.data.data_loader.data_loader_factory import DataLoaderFactory
from app.util.saver.saver import Savers
import torch
import random
import math
import numpy as np
from typing import Dict, Any


class AE(Trainer):
    def __init__(self, config: ConfigFactory,
                 model: Dict[str: Any],
                 logger: Logger,
                 dataloaders: Dict[str: DataLoaderFactory],
                 savers: Savers) -> None:

        super().__init__(config, model, logger)
        self.config = config
        self.model = model
        self.logger = logger
        self.dataloaders = dataloaders
        self.savers = savers

    def get_inference(self, phase: str) -> None:
        pass

    def get_train(self, phase: str) -> None:
        total_loss = 0

        for i_batch, (state, action, next_state, reward, done) in enumerate(self.dataloaders[phase].dataloader):

            (state, next_state, action) = self.items_to_device((state, next_state, action))

            self.model['optimizer'].zero_grad()

            # forward + backward + optimize
            with torch.set_grad_enabled(phase == 'train'):
                out_images = self.model['ae'](state, action)

                loss = self.model['criterion'](out_images, next_state)
                total_loss += loss.item()
                # backward + optimize only if in training phase
                if phase == 'train':
                    self.perform_train_step(loss)

        self.logger.stamp_data({"loss" : total_loss}, self.iterations, phase)
        self.iterations += 1

        if self.if_debug():
            return

    def get_val(self, phase) -> None:
        pass