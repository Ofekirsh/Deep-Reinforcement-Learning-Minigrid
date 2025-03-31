import os
import torch
import pickle
import yaml
from preprocessing import custom_preprocess, deepmind_preprocess
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from utils import get_agent_class


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="configs/config.yaml"):
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

    num_actions = config.get("num_actions", 4)

    # Dynamically choose agent based on agent_variant field
    agent = get_agent_class(config, input_shape, num_actions)

    num_episodes = config.get("num_episodes", 50)
    target_update = config.get("target_update", 1000)
    total_steps = 0
    rewards_per_episode = []
    steps_per_episode = []
    epsilons_per_episode = []
    max_steps = 5000

    # Ensure the logs folder exists.
    os.makedirs("logs", exist_ok=True)

    # Build the log filename using the preprocessing pipeline and agent variant.
    preprocess_name = config.get("preprocessing_pipeline", "custom")
    agent_variant = config.get("agent_variant", "base")
    log_filename = f"logs/rewards_{os.path.splitext(os.path.basename(config_path))[0]}_dqn.pkl"


    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_loss = 0
        while not done:
            state_tensor = preprocess_fn(state).float().to(agent.device)
            action = agent.select_action(state_tensor)
            ##
            changed_action = action
            if changed_action == 3:
                changed_action = 5
            ##
            next_state, original_reward, done, truncated, _ = env.step(changed_action)
            next_state_tensor = preprocess_fn(next_state).float().to(agent.device)
            reward = 0
            if done:
                reward = 1 - 0.9 * steps / max_steps
            done = done or steps >= max_steps
            agent.store_transition(state_tensor, action, reward, next_state_tensor, done)
            loss = agent.update()
            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1
            if total_steps % target_update == 0:
                agent.update_target_network()
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilons_per_episode.append(agent.epsilon)
        print(f"Episode {episode}: Total Reward = {total_reward} epsilon = {agent.epsilon}")

        # Save model and data every 1000 episodes
        if (episode + 1) % 1000 == 0:
            log_data = {
                "rewards_per_episode": rewards_per_episode,
                "steps_per_episode": steps_per_episode,
                "epsilons_per_episode": epsilons_per_episode,
                "total_steps": total_steps
            }

            with open(log_filename, "wb") as f:
                pickle.dump(log_data, f)

            torch.save(agent.policy_net.state_dict(), f"weights/{os.path.splitext(os.path.basename(config_path))[0]}_dqn.pth")

    log_data = {
        "rewards_per_episode": rewards_per_episode,
        "steps_per_episode": steps_per_episode,
        "epsilons_per_episode": epsilons_per_episode,
        "total_steps": total_steps
    }

    with open(log_filename, "wb") as f:
        pickle.dump(log_data, f)

    torch.save(agent.policy_net.state_dict(), f"weights/{os.path.splitext(os.path.basename(config_path))[0]}_dqn.pth")


if __name__ == "__main__":
    main()
