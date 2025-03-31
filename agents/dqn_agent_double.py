import torch
import torch.nn.functional as F
from agents.dqn_agent_base import DQNAgentBase
from utils import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgentDouble(DQNAgentBase):
    """Double DQN Agent to reduce overestimation bias in Q-learning."""

    def __init__(self, input_shape, num_actions, config):
        super().__init__(input_shape, num_actions, config)

    def update(self):
        """Performs the Double DQN update step with the target network."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Handle prioritized replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
                self.batch_size, beta=0.4)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Select next action using policy network but get value from target network
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)  # Best action from policy net
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # Value from target net
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            loss_values = F.mse_loss(q_values, target_q_values, reduction='none')
            loss = (loss_values * weights.squeeze(1)).mean()
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
        else:
            loss = F.mse_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
