from app.config.config_factory import ConfigFactory
from app.model.ddqn_model.ddqn import DDQN
from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from typing import Dict, Any


class DDQNHelper:
    def __init__(self, config:ConfigFactory) -> None:
        self.config = config
        self.model = {}
        self.model_directory = self.config.output_directory + self.config.model_directory

    def DDQN_test(self) -> None:
        self.model = DDQN()

    def DDQN_train(self) -> Dict[str, Any]:
        self.model['policy'] = DDQN(self.config)
        self.model['target'] = DDQN(self.config)

        self.model['target'].load_state_dict(self.model['policy'].state_dict())
        self.model['target'].eval()


        self.model['optimizer'] = optim.RMSprop(self.model['policy'].parameters(), lr=self.config.hyperparameters.lr, weight_decay=1e-5)
        self.model['scheduler'] = lr_scheduler.StepLR(self.model['optimizer'], step_size=self.config.hyperparameters.scheduler_step_size, gamma=self.config.hyperparameters.scheduler_gamma)

        return self.model


