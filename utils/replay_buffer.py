import random
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack([s.clone().detach().float().to(self.device) for s in states])
        next_states = torch.stack([ns.clone().detach().float().to(self.device) for ns in next_states])
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
