import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from app.config.config_factory import ConfigFactory

class MulitHeadAttentionLayer(nn.Module):
    def __init__(self, input_channel, number_of_layers = 6) -> None:
        super(MulitHeadAttentionLayer, self).__init__()
        self.number_of_layers = number_of_layers
        self.layers = nn.ModuleList([AttentionLayer(input_channel) for i in range(number_of_layers)])

    def forward(self, x) -> torch.Tensor:
        output = []
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                output = layer(x)
            else:
                if len(self.layers) > 1 and idx == 1:
                    output = output.unsqueeze(1)
                output = torch.cat((output, layer(x).unsqueeze(1)), dim=1)

        output = output.mean(dim=1)
        output = output + x

        return output

class AttentionLayer(nn.Module):
    def __init__(self, input_channel) -> None:
        super(AttentionLayer, self).__init__()
        self.input_channel = input_channel

        self.F1 = nn.Conv3d(input_channel, input_channel, kernel_size=1, stride=1, padding=0)
        self.F2 = nn.Conv3d(input_channel, input_channel, kernel_size=1, stride=1, padding=0)
        self.G1 = nn.Conv3d(input_channel, input_channel, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

        self.inter_channels = input_channel // 2
        self.i = 0

    def forward(self, x) -> torch.Tensor:
        self.batch = x.shape[0]
        self.channel = x.shape[1]
        self.time = x.shape[2]
        self.height = x.shape[3]
        self.width = x.shape[4]

        self.THW = self.time * self.height*self.width
        self.sqrt = torch.sqrt(torch.tensor(self.THW).to(dtype=torch.float).cuda())


        f1_output = self.F1(x)
        f1_output = f1_output.permute(0,2,3,4,1)
        f1_output = f1_output.view(self.batch, -1, self.channel)

        f2_output = self.F2(x)
        f2_output = f2_output.view(self.batch, self.channel , self.THW)

        combined = torch.matmul(f1_output, f2_output) / self.sqrt
        combined = self.softmax(combined)

        g1_output = self.G1(x).permute(0,2,3,4,1).view(-1, self.THW, self.channel)

        combined = torch.matmul(combined, g1_output)
        combined = combined.view(self.batch, self.time, self.height, self.width, self.channel).permute(0,4,1,2,3)

        return combined


class DDQN(nn.Module):
    def __init__(self, config: ConfigFactory) -> None:
        super(DDQN, self).__init__()
        self.config = config

        if not self.config.ram:
            self.conv = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=(1,8,8), stride=(1,4,4)),
                nn.ReLU(),
                MulitHeadAttentionLayer(32, number_of_layers=4),
                nn.Conv3d(32, 64, kernel_size=(2,4,4), stride=(1,2,2)),
                nn.ReLU(),
            )
            self.input_dimension = self._get_conv_out((1, 4, 84, 84))
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
        o = self.conv(torch.rand(1, *shape))
        return int(np.prod(o.size()))