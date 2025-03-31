from utils.replay_buffer import ReplayBuffer
from collections import deque
import torch


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6):
        super().__init__(capacity)
        self.alpha = alpha
        # Instead of just a deque, we store priorities in parallel.
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Assign a high priority for new experiences (typically maximum priority so far)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
        super().push(state, action, reward, next_state, done)

    def sample(self, batch_size, beta=0.4):
        # Convert priorities to a tensor and compute probabilities.
        priorities = torch.tensor(list(self.priorities), dtype=torch.float32, device=self.device)
        # Exponentiate priorities to control how much prioritization is used.
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices according to the probability distribution.
        indices = torch.multinomial(probs, batch_size, replacement=False).tolist()

        # Retrieve samples.
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.stack([s.clone().detach().float().to(self.device) for s in states])
        next_states = torch.stack([ns.clone().detach().float().to(self.device) for ns in next_states])
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Compute importance-sampling weights.
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # for stability
        weights = weights.unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        # Update the priorities at the sampled indices.
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority