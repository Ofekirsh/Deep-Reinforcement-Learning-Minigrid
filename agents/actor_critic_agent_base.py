import torch
import torch.optim as optim
import torch.nn.functional as F
from models import ActorCriticBase
import numpy as np

class ActorCriticAgentBase:
    def __init__(self, input_shape, num_actions, lr=1e-3, gamma=0.99):
        """
        Args:
            input_shape (tuple): Shape of the environment input.
            num_actions (int): Number of discrete actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCriticBase(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        """
        Select an action given the current state.

        Args:
            state (np.array): Input state (should match input_shape).

        Returns:
            action (int): Selected action.
            log_prob (torch.Tensor): Log probability of the selected action.
            value (torch.Tensor): Value estimate of the state.
        """
        # Convert state to torch tensor and add batch dimension.
        policy_logits, value = self.model(state)
        # Compute probabilities and sample an action.
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def compute_returns(self, rewards, dones, next_value):
        """
        Compute discounted returns for a trajectory.

        Args:
            rewards (list[float]): Rewards collected.
            dones (list[bool]): Done flags for each timestep.
            next_value (float): Value estimate for the state following the trajectory.

        Returns:
            returns (list[float]): Discounted returns.
        """
        returns = []
        R = next_value
        # Process rewards backwards.
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0  # Reset return if episode ended.
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self, log_probs, values, rewards, dones, next_state):
        """
        Perform an update step using the collected trajectory.

        Args:
            log_probs (list[torch.Tensor]): Log probabilities for actions taken.
            values (list[torch.Tensor]): Value estimates for states.
            rewards (list[float]): Collected rewards.
            dones (list[bool]): Done flags per timestep.
            next_state (np.array): The state following the trajectory.

        Returns:
            loss_value (float): The combined actor-critic loss.
        """
        # Get the value for the next state.
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        _, next_value = self.model(next_state)
        next_value = next_value.item()

        # Compute discounted returns.
        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.tensor(returns).to(self.device)
        values = torch.cat(values).squeeze(-1)
        log_probs = torch.cat(log_probs)

        # Compute advantages.
        advantages = returns - values

        # Actor loss: encourage actions that yielded higher returns.
        actor_loss = -(log_probs * advantages.detach()).mean()
        # Critic loss: mean squared error for value estimation.
        critic_loss = advantages.pow(2).mean()
        # Combined loss.
        loss = actor_loss + critic_loss

        # Gradient step.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()