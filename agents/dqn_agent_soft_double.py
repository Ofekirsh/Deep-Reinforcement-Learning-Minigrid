from .dqn_agent_soft import DQNAgentSoftUpdate
from .dqn_agent_double import DQNAgentDouble


class DQNAgentSoftDouble(DQNAgentDouble, DQNAgentSoftUpdate):
    """
    Agent that combines Soft Update and Double DQN.
    It inherits from both DQNAgentSoftUpdate and DQNAgentDouble.
    """
    def __init__(self, input_shape, num_actions, config):
        super().__init__(input_shape, num_actions, config)
