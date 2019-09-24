import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from app.config.config_factory import ConfigFactory


class DDQN(nn.Module):
    def __init__(self, config: ConfigFactory) -> None:
        nn.Module.__init__(self)
        self.config = config
        if not self.config.ram:
            self.conv = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            self.input_dimension = self._get_conv_out((4, 84,84))
            self.linear = nn.Sequential(
                nn.Linear(self.input_dimension, 512),
                nn.ReLU(),
                nn.Linear(512, config.n_actions)
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(config.input_dimension * (4 if config.stacked_frames else 1) , 64),
                nn.ReLU(),
                nn.Linear(64, 12),
                nn.ReLU(),
                nn.Linear(12, config.n_actions)
            )

    def forward(self, x) -> torch.Tensor:
        if not self.config.ram:
            x = self.conv(x)
            x = x.view(x.size()[0], -1)
        return self.linear(x)

    def _get_conv_out(self, shape) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))