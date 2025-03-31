import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticBase(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Args:
            input_shape (tuple): Shape of the input (C, H, W).
            num_actions (int): Number of discrete actions.
        """
        super(ActorCriticBase, self).__init__()

        # Convolutional feature extractor.
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Dynamically compute the size of the conv output.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy_input)
            conv_out_size = conv_out.view(1, -1).size(1)

        # Fully connected layer.
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size * 4, 512),
            nn.ReLU(),
        )
        # Policy head: outputs logits for each action.
        self.policy = nn.Linear(512, num_actions)
        # Value head: outputs a single value estimate.
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch, C, H, W).

        Returns:
            policy_logits (torch.Tensor): Logits for the action distribution.
            value (torch.Tensor): Value estimate of the state.
        """
        x = self.conv(x)
        x = x.view(1, -1)
        x = self.fc(x)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value