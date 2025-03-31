import torch
import yaml
import imageio
import gymnasium as gym
from preprocessing import custom_preprocess, deepmind_preprocess
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from utils import get_agent_class
import os
from collections import deque

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

    pipeline = config.get("preprocessing_pipeline", "custom")
    if pipeline == "deepmind":
        preprocess_fn = deepmind_preprocess.process_image
        input_shape = (1, 84, 84)
    else:
        preprocess_fn = custom_preprocess.process_image
        input_shape = (5, 7, 7)

    k = 4  # Number of steps to remember
    input_shape = list(input_shape)
    input_shape[0] = input_shape[0] * k
    input_shape = tuple(input_shape)
    num_actions = config.get("num_actions", 4)
    agent = get_agent_class(config, input_shape, num_actions)

    agent.policy_net.load_state_dict(torch.load(f"weights/{os.path.splitext(os.path.basename(config_path))[0]}_dqn_with_memory.pth", map_location=agent.device))
    agent.policy_net.eval()
    done = False
    total_reward = 0
    steps = 0
    max_steps = 499

    state_buffer = deque(maxlen=k)

    video_filename = f"videos/{os.path.splitext(os.path.basename(config_path))[0]}.mp4"
    os.makedirs(os.path.dirname(video_filename), exist_ok=True)

    with imageio.get_writer(video_filename, fps=10) as video:
        obs, _ = env.reset()
        while not done:
            state_tensor = preprocess_fn(obs).float().to(agent.device)
            state_buffer.append(state_tensor)
            
            if len(state_buffer) < k:
                combined_state = torch.cat([state_tensor] * (k - len(state_buffer)) + list(state_buffer), dim=0)
            else:
                combined_state = torch.cat(list(state_buffer), dim=0)
            with torch.no_grad():
                # print(agent.policy_net(combined_state.unsqueeze(0)))
                # action = agent.policy_net(combined_state.unsqueeze(0)).argmax().item()
                agent.epsilon = 0
                action = agent.select_action(combined_state)
            if action == 3:
                action = 5
            obs, _, done, _, _ = env.step(action)
            reward = 0
            if done:
                reward = 1 - 0.9 * steps / max_steps
            done = done or steps >= max_steps
            total_reward += reward
            steps += 1
            frame = env.render()
            video.append_data(frame)
            if done:
                print("Evaluation complete:", "reward =", total_reward, "steps =", steps)
                break

if __name__ == "__main__":
    main()
