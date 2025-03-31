from .dqn_agent_soft import DQNAgentSoftUpdate
from .dqn_agent_double_huber import DQNAgentDoubleHuber


class DQNAgentSoftDoubleHuber(DQNAgentDoubleHuber, DQNAgentSoftUpdate):
    """
    Agent that combines Soft Update, Double DQN, and Huber Loss.
    It inherits from both DQNAgentDoubleHuber and DQNAgentSoftUpdate.
    """
    def __init__(self, input_shape, num_actions, config):
        super().__init__(input_shape, num_actions, config)
