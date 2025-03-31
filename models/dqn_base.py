import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNBase(nn.Module):
    def __init__(self, input_shape=(5, 7, 7), num_actions=4):
        super(DQNBase, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        ).to(self.device)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape, device=self.device))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        y = self.fc(x)
        return y
