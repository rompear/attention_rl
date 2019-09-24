from app.config.config_factory import ConfigFactory
from app.model.ae_model.ae import AE
import torch.optim as optim
import torch
import torch.nn.functional as F
from typing import Dict, Any

class AEHelper:
    def __init__(self, config:ConfigFactory) -> None:
        self.config = config
        self.model = {}
        self.model_directory = self.config.output_directory + self.config.model_directory

    def AE_test(self) -> None:
        self.model = AE()

    def AE_train(self) -> Dict[str: Any]:
        self.model['ae'] = AE(self.config)

        self.model['optimizer'] = optim.RMSprop(self.model['ae'].parameters(), lr=self.config.hyperparameters.lr)
        # self.model['criterion'] = torch.nn.SmoothL1Loss()
        self.model['criterion'] = torch.nn.MSELoss()  # mse_loss -> https://discuss.pytorch.org/t/dqn-example-from-pytorch-diverged/4123/6

        return self.model
