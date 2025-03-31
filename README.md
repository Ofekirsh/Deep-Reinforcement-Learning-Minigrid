# Deep Reinforcement Learning Minigrid

This repository contains a Deep Reinforcement Learning project implemented in PyTorch. It starts with a basic Deep Q-Network (DQN) applied to a MiniGrid environment and then introduces several improvements such as prioritized experience replay.

![Evaluation Demo](videos/evaluation.gif)


## Project Structure

```plaintext
Deep-Reinforcement-Learning-Minigrid/
├── configs/
│   └── config.yaml
├── models/
│   ├── __init__.py
│   ├── dqn_base.py
│   ├── dqn_double.py
│   ├── dqn_dueling.py
│   ├── actor_critic_base.py
│   └── actor_critic_advanced.py
├── agents/
│   ├── __init__.py
│   ├── dqn_agent_base.py
│   ├── dqn_agent_double.py
│   ├── dqn_agent_dueling.py
│   ├── actor_critic_agent_base.py
│   └── actor_critic_agent_advanced.py
├── experiments/
│   ├── dqn/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── actor_critic/
│       ├── train.py
│       └── evaluate.py
├── preprocessing/
│   ├── __init__.py
│   ├── custom_preprocess.py
│   └── deepmind_preprocess.py
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py
│   └── prioritized_replay_buffer.py
├── videos/
├── requirements.txt
└── README.md
```

## Setup

Before running the code, install the required dependencies.

### Python Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

### Install ffmpeg

For video evaluation, you need the `ffmpeg` executable. On macOS, open a terminal and run:

```bash
brew install ffmpeg
```

This command installs `ffmpeg`, which is used by ImageIO to record evaluation videos.
_If you're on Windows, please download a static build of ffmpeg and add its `bin` directory to your system PATH. On Linux, install ffmpeg via your package manager._

## Implementation Details

### Basic DQN

- The basic DQN network is implemented in `models/dqn_base.py`.
- The DQN agent (`DQNAgentBase`) is implemented in `agents/dqn_agent_base.py` using an ε-greedy exploration strategy and stores transitions in a replay buffer.
- The training script in `experiments/dqn/train.py` sets up the MiniGrid environment (using Gymnasium and MiniGrid wrappers), selects a preprocessing pipeline, and trains the DQN.

### Improvements

1. **Experience Replay with Prioritization**
   - **Motivation:**
     Prioritized experience replay improves sample efficiency by replaying more informative transitions more often.
     For details, see the paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

   - **Implementation:**
     A new class `PrioritizedReplayBuffer` was added in `utils/prioritized_replay_buffer.py` that inherits from the uniform `ReplayBuffer`.

   - **Changes in Agent:**
     In `agents/dqn_agent_base.py`, the `update()` method is modified to:
       - Sample indices and importance-sampling weights when using a prioritized replay buffer.
       - Compute a weighted loss using these weights.
       - Update the priorities based on the temporal-difference (TD) error.

   - **Configuration:**
     In `configs/config.yaml`, you can set:

     ```yaml
     replay_buffer: "prioritized"  # or "uniform"
     priority_alpha: 0.6
     ```

2. **Soft Update for Target Network**

   - **Motivation:**
     Hard updates replace the entire target network at set intervals, but this can cause instability.
     Soft updates, using Polyak averaging, provide smoother updates, improving stability and convergence.

   - **Implementation:**
     We introduced `DQNAgentSoftUpdate` in `agents/dqn_agent_soft.py`, which inherits from `DQNAgentBase` and overrides `update_target_network()` to implement soft updates.

   - **Changes in Agent:**
     - The `update_target_network()` method was modified to support soft updates with Polyak averaging.
     - The `train.py` script was updated to select either `DQNAgentBase` (hard updates) or `DQNAgentSoftUpdate` (soft updates) based on configuration.

   - **Configuration:**
     In `configs/config.yaml`, you can set:

     ```yaml
     agent_variant: "soft_update"
     soft_update_tau: 0.005
     ```

3. **Huber Loss for Stability**
   - **Motivation:**
     Mean Squared Error (MSE) can be overly sensitive to large errors, which can destabilize training. The Huber loss is more robust as it behaves like MSE for small errors but transitions to an absolute loss for large errors, reducing the impact of outliers.

   - **Implementation:**
     We introduced `DQNAgentHuberLoss` in `agents/dqn_agent_huber.py`, which inherits from `DQNAgentBase` and overrides the `update()` function to use Huber loss instead of MSE loss.

   - **Changes in Agent:**
     - The `update()` method now applies the Huber loss instead of the standard MSE loss when using this agent.
     - The `train.py` script was updated to allow selecting `DQNAgentHuberLoss` via configuration.

   - **Configuration:**
     In `configs/config.yaml`, you can set:

     ```yaml
     agent_variant: "huber_loss"
     huber_loss_weight: 0.01
    ```

4. **Double DQN for More Stable Learning**

   - **Motivation:**
     The standard DQN tends to **overestimate** Q-values, which can slow down or destabilize learning. **Double DQN (DDQN)** mitigates this issue by using **two separate networks**:
     - One for selecting the action (policy network).
     - One for evaluating the action (target network).
     This prevents over-optimistic value estimates and improves training stability.

   - **Implementation:**
     We introduced `DQNAgentDouble` in `agents/dqn_agent_double.py`, which modifies the `update()` method to:
     - Use the **policy network** to select the best action.
     - Use the **target network** to evaluate the Q-value of that action.

   - **Changes in Agent:**
     - The `update()` method now follows the Double DQN logic to compute target Q-values.
     - The `train.py` script was updated to allow selecting `DQNAgentDouble` via configuration.

   - **Configuration:**
     In `configs/config.yaml`, you can set:

     ```yaml
     agent_variant: "double_dqn"
     ```
---

## Running the Project

- **Training:**
  To train the DQN agent, run:

  ```bash
  python experiments/dqn/train.py
  ```
- **Evaluation:**
  To evaluate the trained agent and record a video of its performance, run:

  ```bash
  python experiments/dqn/evaluate.py
  ```

## Summary

- This project implements a basic DQN applied to a MiniGrid environment.
- Improvements include adding a prioritized experience replay mechanism (see [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)).
- The replay buffer type (uniform or prioritized) can be configured via `configs/config.yaml`.
- The project is modular, making it easy to extend with further improvements.

