import torch
import torch.nn.functional as F
from agents.dqn_agent_base import DQNAgentBase
from utils import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgentDoubleHuber(DQNAgentBase):
    """
    DQN Agent that combines Double DQN with Huber Loss.
    """

    def __init__(self, input_shape, num_actions, config):
        super().__init__(input_shape, num_actions, config)
        self.huber_loss_weight = config.get("huber_loss_weight", 0.01)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Check if we're using a prioritized replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, beta=0.4)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute Q-values for current states
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: Use policy network to select actions, target network to get Q-values
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Use Huber Loss instead of MSE Loss
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            loss_values = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
            loss = (loss_values * weights.squeeze(1)).mean()
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
        else:
            loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
