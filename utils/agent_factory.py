
from agents import (
    DQNAgentBase, DQNAgentSoftUpdate, DQNAgentHuberLoss,
    DQNAgentDouble, DQNAgentDoubleHuber, DQNAgentSoftHuber,
    DQNAgentSoftDouble, DQNAgentSoftDoubleHuber, ActorCriticAgentBase
)


def get_agent_class(config, input_shape, num_actions):
    """
    Determines the correct agent class based on the agent_variant field in config.
    Supports:
      - "base" -> DQNAgentBase
      - "soft_update" -> DQNAgentSoftUpdate
      - "huber_loss" -> DQNAgentHuberLoss
      - "double_dqn" -> DQNAgentDouble
      - "double_huber" -> DQNAgentDoubleHuber
      - "soft_update_huber" -> DQNAgentSoftHuber
      - "soft_update_double" -> DQNAgentSoftDouble
      - "soft_update_double_huber" -> DQNAgentSoftDoubleHuber
    """
    agent_variant = config.get("agent_variant", "base")

    if agent_variant == "base":
        return DQNAgentBase(input_shape, num_actions, config)
    elif agent_variant == "soft_update":
        return DQNAgentSoftUpdate(input_shape, num_actions, config)
    elif agent_variant == "huber_loss":
        return DQNAgentHuberLoss(input_shape, num_actions, config)
    elif agent_variant == "double_dqn":
        return DQNAgentDouble(input_shape, num_actions, config)
    elif agent_variant == "double_huber":
        return DQNAgentDoubleHuber(input_shape, num_actions, config)
    elif agent_variant == "soft_update_huber":
        return DQNAgentSoftHuber(input_shape, num_actions, config)
    elif agent_variant == "soft_update_double":
        return DQNAgentSoftDouble(input_shape, num_actions, config)
    elif agent_variant == "soft_update_double_huber":
        return DQNAgentSoftDoubleHuber(input_shape, num_actions, config)
    elif agent_variant == "actor_critic":
        return ActorCriticAgentBase(input_shape, num_actions)
    else:
        raise ValueError(f"Unknown agent_variant: {agent_variant}")
