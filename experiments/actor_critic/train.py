import os
import random

import numpy as np
import torch
import pickle
import yaml
import collections
from preprocessing import custom_preprocess, deepmind_preprocess
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from utils import get_agent_class  # Dynamically loads the correct agent class


def load_config(path="/configs/actor_critic.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="/configs/actor_critic.yaml"):
    config = load_config(config_path)
    env_name = config.get("env_name", "MiniGrid-MultiRoom-N2-S4-v0")
    highlight = False
    render_mode = "rgb_array"

    # Create the environment and apply MiniGrid wrappers.
    env = gym.make(env_name, render_mode=render_mode, highlight=highlight)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    # Choose preprocessing pipeline and set input shape accordingly.
    # We are stacking 4 frames per state.
    pipeline = config.get("preprocessing_pipeline", "custom")
    if pipeline == "deepmind":
        preprocess_fn = deepmind_preprocess.process_image
        input_shape = (4, 84, 84)
    else:
        preprocess_fn = custom_preprocess.process_image
        input_shape = (5, 7, 7)

    num_actions = config.get("num_actions", 4)

    # Dynamically choose the actor-critic agent based on configuration.
    agent = get_agent_class(config, input_shape, num_actions)

    num_episodes = config.get("num_episodes", 50)
    max_steps = 500
    total_steps = 0
    rewards_per_episode = []
    steps_per_episode = []  # To store the number of steps per episode.

    for episode in range(num_episodes):
        state, _ = env.reset()
        # Initialize a frame stack (deque with maxlen=4) with the first frame repeated.
        frame_stack = collections.deque(maxlen=4)
        processed_frame = preprocess_fn(state).float()
        for _ in range(4):
            frame_stack.append(processed_frame)

        door_diff = 0
        total_reward = 0
        done = False
        episode_steps = 0  # Count steps in this episode.

        while not done:
            # Build the current state by stacking the 4 frames.
            state_stack = torch.stack(list(frame_stack), dim=0)  # shape: (4, H, W)


            # Select an action using the actor-critic agent.
            action, log_prob, value = agent.select_action(state_stack)
            # output 3 in the instruction is pickup (unused) so we move to toggle (open the door action)
            changed_action = action
            if changed_action == 3:
                changed_action = 5

            # Take the action in the environment.
            next_state, reward, done, _, _ = env.step(changed_action)

            if done:
                # the reward calculate with the formula - reward = reward = 1 - 0.9 * steps / max_steps
                # but we want to redefine max_steps and normalize the reward
                reward = 20 * (1 - 0.9 * episode_steps / max_steps)

            # to make sure the agent doesnt find a bug to farm reward of opening door, there is a bound
            change_reward = door_change(state, next_state, door_diff)
            door_diff += change_reward
            reward += change_reward
            done = done or episode_steps > max_steps
            # Preprocess the new frame and update the frame stack.
            processed_next_frame = preprocess_fn(next_state).float()
            frame_stack.append(processed_next_frame)
            next_state_stack = torch.stack(list(frame_stack), dim=0)

            # ---------------------------
            # Per-Step Actor-Critic Update
            # ---------------------------
            # Compute the value of the next state.
            with torch.no_grad():
                _, next_value = agent.model(next_state_stack.to(agent.device))
            next_value = next_value.item() if not done else 0.0

            # Compute the TD target and TD error.
            target = reward + agent.gamma * next_value
            td_error = target - value  # value is a tensor

            # Actor loss: policy gradient weighted by the TD error.
            actor_loss = - log_prob * td_error.detach()

            # Critic loss: mean squared error on the TD error.
            critic_loss = td_error.pow(2)

            loss = actor_loss + critic_loss

            # Update the model.

            num = random.random()
            if reward != 0 or num < 1/8:
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

            total_reward += reward
            episode_steps += 1
            total_steps += 1
            state = next_state

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(episode_steps)
        print(f"Episode {episode}: Total Reward = {total_reward}, Steps = {episode_steps}")

        # Save model and log data every 200 episodes.
        if (episode + 1) % 200 == 0:
            log_data = {
                "rewards_per_episode": rewards_per_episode,
                "steps_per_episode": steps_per_episode,
                "total_steps": total_steps
            }
            os.makedirs("logs", exist_ok=True)
            log_filename = f"logs/log_data_episode_{episode + 1}.pkl"
            with open(log_filename, "wb") as f:
                pickle.dump(log_data, f)
            # Optionally, save model weights as well.
            os.makedirs("weights", exist_ok=True)
            model_filename = f"weights/model_episode_{episode + 1}.pth"
            torch.save(agent.model.state_dict(), model_filename)

    # Save final logs and model weights.
    os.makedirs("logs", exist_ok=True)
    preprocess_name = config.get("preprocessing_pipeline", "custom")
    agent_variant = config.get("agent_variant", "base")
    log_filename = f"logs/rewards_{preprocess_name}_{agent_variant}_actor_critic.pkl"
    with open(log_filename, "wb") as f:
        pickle.dump(rewards_per_episode, f)

    os.makedirs("weights", exist_ok=True)
    torch.save(agent.model.state_dict(), f"weights/{preprocess_name}_{agent_variant}_actor_critic.pth")


def door_change(last_state, current_state, door_diff):
    last_state = custom_preprocess.process_grid(last_state)
    current_state = custom_preprocess.process_grid(current_state)
    # Convert lists to numpy arrays if they aren't already
    last_state = np.array(last_state)
    current_state = np.array(current_state)

    # Create a boolean mask where last_state equals 3 and current_state equals 5
    mask = (last_state == 3) & (current_state == 5)
    if np.any(mask) and door_diff < 5:
        return 1
    mask = (last_state == 5) & (current_state == 3)
    if np.any(mask) and door_diff > 0:
        return -1
    return 0


if __name__ == "__main__":
    main()
