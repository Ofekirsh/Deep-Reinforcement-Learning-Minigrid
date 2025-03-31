from .dqn_agent_soft import DQNAgentSoftUpdate
from .dqn_agent_huber_loss import DQNAgentHuberLoss


class DQNAgentSoftHuber(DQNAgentHuberLoss, DQNAgentSoftUpdate):
    """
    Agent that combines Soft Update and Huber Loss.
    It inherits from both DQNAgentSoftUpdate and DQNAgentHuberLoss.
    """
    def __init__(self, input_shape, num_actions, config):
        super().__init__(input_shape, num_actions, config)
