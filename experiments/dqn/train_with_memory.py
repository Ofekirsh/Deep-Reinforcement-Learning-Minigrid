import os
import torch
import pickle
import yaml
from collections import deque
from preprocessing import custom_preprocess, deepmind_preprocess
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from utils import get_agent_class
import numpy as np  
import matplotlib.pyplot as plt
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="configs/config.yaml"):
    print("trainV2")
    config = load_config(config_path)
    env_name = config.get("env_name", "MiniGrid-MultiRoom-N4-S5-v0")
    highlight = False
    render_mode = "rgb_array"
    env = gym.make(env_name, render_mode=render_mode, highlight=highlight)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    # Choose preprocessing pipeline and set input shape accordingly
    pipeline = config.get("preprocessing_pipeline", "custom")
    if pipeline == "deepmind":
        preprocess_fn = deepmind_preprocess.process_image
        input_shape = (1, 84, 84)
    else:
        preprocess_fn = custom_preprocess.process_image
        input_shape = (5, 7, 7)

    k = 4  # Number of steps to remember
    input_shape = list(input_shape)  # Convert tuple to list
    input_shape[0] = input_shape[0] * k  # Adjust the input shape for memory
    input_shape = tuple(input_shape)  # Convert back to tuple

    num_actions = config.get("num_actions", 4)

    # Dynamically choose agent based on agent_variant field
    agent = get_agent_class(config, input_shape, num_actions)

    num_episodes = config.get("num_episodes", 50)
    target_update = config.get("target_update", 1000)
    total_steps = 0
    rewards_per_episode = []
    steps_per_episode = []
    epsilons_per_episode = []
    max_steps = 500

    # Initialize state buffer
    state_buffer = deque(maxlen=k)
    
    # Ensure the logs folder exists.
    os.makedirs("logs", exist_ok=True)

    # Build the log filename using the preprocessing pipeline and agent variant.
    preprocess_name = config.get("preprocessing_pipeline", "custom")
    agent_variant = config.get("agent_variant", "base")
    log_filename = f"logs/rewards_{os.path.splitext(os.path.basename(config_path))[0]}_dqn_with_memory.pkl"

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        init_state_tensor = preprocess_fn(state).float().to(agent.device)
        state_buffer.append(init_state_tensor)
        combined_state = torch.cat([init_state_tensor] * k,dim=0)
        #combined_next_state = torch.cat([init_state_tensor] * k,dim=0)
        door_diff = 0
        while not done:
            action = agent.select_action(combined_state)
            # output 3 in the instruction is pickup (unused) so we move to toggle (open the door action)
            changed_action =action
            if changed_action == 3:
                changed_action = 5
            ##

            next_state, original_reward, done, truncated, _ = env.step(changed_action)
            next_state_tensor = preprocess_fn(next_state).float().to(agent.device)
            state_buffer.append(next_state_tensor)

            if len(state_buffer) < k:
                combined_next_state = torch.cat(list(state_buffer) + [next_state_tensor] * (k - len(state_buffer)), dim=0)
            else:
                combined_next_state = torch.cat(list(state_buffer), dim=0)

            reward = 0

            if done:
                # the reward calculate with the formula - reward = reward = 1 - 0.9 * steps / max_steps
                # but we want to redefine max_steps and normalize the reward
                reward = 20*(1 - 0.9 * steps / max_steps)

            # to make sure the agent doesnt find a bug to farm reward of opening door, there is a bound
            change_reward = door_change(state,next_state,door_diff)
            door_diff += change_reward
            reward += change_reward
            ##

            done = done or steps >= max_steps
            agent.store_transition(combined_state, action, reward, combined_next_state, done)
            state= next_state
            loss = agent.update()
            combined_state = combined_next_state
            total_reward += reward
            steps += 1
            total_steps += 1
            if total_steps % target_update == 0:
                agent.update_target_network()
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilons_per_episode.append(agent.epsilon)
        
        print(f"Episode {episode}: Total Reward = {total_reward} epsilon = {agent.epsilon} door_diff = {door_diff}")

        # Save model and data every 1000 episodes
        if (episode + 1) % 200 == 0:
            log_data = {
                "rewards_per_episode": rewards_per_episode,
                "steps_per_episode": steps_per_episode,
                "epsilons_per_episode": epsilons_per_episode,
                "total_steps": total_steps
            }

            with open(log_filename, "wb") as f:
                pickle.dump(log_data, f)

            torch.save(agent.policy_net.state_dict(), f"weights/{os.path.splitext(os.path.basename(config_path))[0]}_dqn_with_memory.pth")

    log_data = {
        "rewards_per_episode": rewards_per_episode,
        "steps_per_episode": steps_per_episode,
        "epsilons_per_episode": epsilons_per_episode,
        "total_steps": total_steps
    }

    with open(log_filename, "wb") as f:
        pickle.dump(log_data, f)

    torch.save(agent.policy_net.state_dict(), f"weights/{os.path.splitext(os.path.basename(config_path))[0]}_dqn_with_memory.pth")


def door_change(last_state,current_state,door_diff):
    last_state =custom_preprocess.process_grid(last_state)
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
