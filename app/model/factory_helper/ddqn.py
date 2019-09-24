from app.config.config_factory import ConfigFactory
from app.model.ddqn_model.ddqn import DDQN
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

    def DDQN_train(self) -> Dict[str: Any]:
        self.model['policy'] = DDQN(self.config)
        self.model['target'] = DDQN(self.config)

        self.model['target'].load_state_dict(self.model['policy'].state_dict())
        self.model['target'].eval()


        self.model['optimizer'] = optim.RMSprop(self.model['policy'].parameters(), lr=self.config.hyperparameters.lr)
        # self.model['criterion'] = torch.nn.SmoothL1Loss()
        self.model['criterion'] = torch.nn.SmoothL1Loss()  # mse_loss -> https://discuss.pytorch.org/t/dqn-example-from-pytorch-diverged/4123/6

        return self.model
