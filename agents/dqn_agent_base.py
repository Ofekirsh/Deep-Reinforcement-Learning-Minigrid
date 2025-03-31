import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from models import DQNBase
from utils import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgentBase:
    def __init__(self, input_shape, num_actions, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.policy_net = DQNBase(input_shape, num_actions).to(self.device)
        self.target_net = DQNBase(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Convert learning rate to float in case it is a string.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(config.get("learning_rate", 1e-4)))
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 10000)
        self.step_count = 0

        buffer_type = config.get("replay_buffer", "uniform")
        capacity = config.get("buffer_size", 10000)
        if buffer_type == "prioritized":
            # You can pass additional hyperparameters (e.g., alpha) as needed.
            self.replay_buffer = PrioritizedReplayBuffer(capacity, alpha=config.get("priority_alpha", 0.6))
        else:
            self.replay_buffer = ReplayBuffer(capacity)

    def select_action(self, state):
        self.step_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0).to(self.device))
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Check if we're using a prioritized replay buffer.
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # The prioritized buffer returns extra outputs: indices and weights.
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, beta=0.4)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute Q-values for current states.
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Select max Q-value from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss differently for prioritized vs uniform replay.
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):

            # Compute element-wise loss without reduction.
            loss_values = F.mse_loss(q_values, target_q_values, reduction='none')

            # Weight the loss by importance-sampling weights.
            loss = (loss_values * weights.squeeze(1)).mean()

            # Compute TD error for updating priorities.
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

        else:
            loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        # hard update
        self.target_net.load_state_dict(self.policy_net.state_dict())
