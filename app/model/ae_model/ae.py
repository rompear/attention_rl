import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from app.config.config_factory import ConfigFactory

class AE(nn.Module):
    def __init__(self, config: ConfigFactory) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.input_shape = (4, 84, 84)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.lin1 = nn.Linear(65, 64)
        input_size = 8
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, input_size*64, 4, 1, 0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size*64, input_size*32, 8, 1, 0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size * 32, input_size * 16, 16, 1, 0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size * 16, input_size * 8, 16, 1, 0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size * 8, input_size * 4, 21, 1, 0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size * 4, 4, 24, 1, 0, bias=False)
        )

    def forward(self, x, action) -> torch.Tensor:
        x = self.encoder(x)

        x = x.view(x.shape[0], -1)
        x  = torch.cat((x, action),1)

        x = self.lin1(x)
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)

        x = self.decoder(x)
        return x

    def _get_conv_out(self, shape) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))