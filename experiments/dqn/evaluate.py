import torch
import yaml
import imageio
import gymnasium as gym
from preprocessing import custom_preprocess, deepmind_preprocess
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper  # Direct import from minigrid
from utils import get_agent_class
import os

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="configs/config.yaml"):
    config = load_config(config_path)

    # Environment parameters
    env_name = config.get("env_name", "MiniGrid-MultiRoom-N4-S5-v0")
    render_mode = "rgb_array"
    highlight = False  # Set as needed

    # Create and wrap the environment
    env = gym.make(env_name, render_mode=render_mode, highlight=highlight)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    # Video output filename (ensure the directory exists)
    video_filename = "videos/evaluation.mp4"

    # Choose preprocessing pipeline and set input shape accordingly
    pipeline = config.get("preprocessing_pipeline", "custom")
    if pipeline == "deepmind":
        preprocess_fn = deepmind_preprocess.process_image
        input_shape = (1, 84, 84)
    else:
        preprocess_fn = custom_preprocess.process_image
        input_shape = (5, 7, 7)

    num_actions = config.get("num_actions", 7)

    # Dynamically instantiate the correct agent
    agent = get_agent_class(config, input_shape, num_actions)

    agent.policy_net.load_state_dict(torch.load(f"weights/{os.path.splitext(os.path.basename(config_path))[0]}_dqn.pth", map_location=agent.device))
    agent.policy_net.eval()
    done = False
    total_reward = 0
    steps = 0
    action = 0
    max_steps = 5000
    # Evaluation loop with video recording
    with imageio.get_writer(video_filename, fps=10) as video:
        obs, _ = env.reset()
        while not done:
            state_tensor = preprocess_fn(obs).float()
            with torch.no_grad():
                action = agent.policy_net(state_tensor.unsqueeze(0)).argmax().item()
            if action == 3:
                action = 5
            obs, _, done, _, _ = env.step(action)
            reward = 0
            if done:
                reward = 1 - 0.9 * steps / max_steps
            done = done or steps >= max_steps
            total_reward += reward
            steps += 1
            frame = env.render()  # capture the rendered frame
            video.append_data(frame)
            if done:
                print("Evaluation complete:", "reward =", total_reward, "steps =", steps)
                break


if __name__ == "__main__":
    main()
