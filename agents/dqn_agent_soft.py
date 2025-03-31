import torch
from agents.dqn_agent_base import DQNAgentBase


class DQNAgentSoftUpdate(DQNAgentBase):
    def __init__(self, input_shape, num_actions, config):
        super().__init__(input_shape, num_actions, config)
        self.soft_update_tau = config.get("soft_update_tau", 0.005)

    def update_target_network(self):
        tau = self.soft_update_tau
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
