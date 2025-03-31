# Configuration File (`config.yaml`) Guide

This document explains each parameter in the `config.yaml` file, which defines the hyperparameters and settings for training and evaluating a DQN agent in the MiniGrid environment.

---

## Configuration Parameters

### **Environment & General Settings**
- **`env_name`** (`str`): Name of the MiniGrid environment.
  Example: `"MiniGrid-MultiRoom-N4-S5-v0"`

- **`num_episodes`** (`int`): Number of episodes for training.
  Example: `20`

- **`eval_episodes`** (`int`): Number of episodes for evaluation.
  Example: `5`

- **`num_actions`** (`int`): Number of possible actions in the environment.
  Example: `7`

---

## **Hyperparameters for DQN**
- **`learning_rate`** (`float`): Learning rate for the optimizer.
  Example: `0.0001`

- **`gamma`** (`float`): Discount factor for future rewards.
  Example: `0.99`

- **`batch_size`** (`int`): Number of transitions sampled from the replay buffer for training.
  Example: `32`

- **`epsilon_start`** (`float`): Initial exploration rate (probability of selecting a random action).
  Example: `1.0`

- **`epsilon_end`** (`float`): Minimum exploration rate.
  Example: `0.1`

- **`epsilon_decay`** (`int`): Number of steps over which `epsilon` is annealed from `epsilon_start` to `epsilon_end`.
  Example: `10000`

- **`target_update`** (`int`): Frequency (in steps) at which the target network is updated.
  Example: `2`

---

## **Replay Buffer**
- **`buffer_size`** (`int`): Maximum number of transitions stored in the replay buffer.
  Example: `10000`

- **`replay_buffer`** (`str`): Type of replay buffer to use.
  Options:
  - `"uniform"` (Standard Experience Replay)
  - `"prioritized"` (Prioritized Experience Replay, see [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952))
  Example: `"prioritized"`

- **`priority_alpha`** (`float`): Prioritization factor for the prioritized replay buffer (only used when `replay_buffer` is `"prioritized"`).
  - Determines how much prioritization is applied (0 = uniform sampling, 1 = fully prioritized).
  Example: `0.6`

---

## **Preprocessing & Agent Variants**
- **`preprocessing_pipeline`** (`str`): Defines how observations are preprocessed before being fed into the model.
  Options:
  - `"deepmind"` (Preprocesses observations into 84x84 grayscale images)
  - `"custom"` (Uses a custom preprocessing method based on RGB and positional encoding)
  Example: `"deepmind"`

- **`agent_variant`** (`str`): Specifies which agent implementation to use.
  Example: `"base"`

---
