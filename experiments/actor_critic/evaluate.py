import torch
import yaml
import imageio
import gymnasium as gym
import os
import collections

from preprocessing import custom_preprocess, deepmind_preprocess
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from utils import get_agent_class

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

def load_config(path="../../configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Environment parameters
    env_name = config.get("env_name", "MiniGrid-MultiRoom-N2-S4-v0")
    render_mode = "rgb_array"
    highlight = False

    # Create and wrap the environment
    env = gym.make(env_name, render_mode=render_mode, highlight=highlight)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    # Video output filename (ensure the directory exists)
    video_filename = "../../videos/evaluation_actor_critic.mp4"
    os.makedirs(os.path.dirname(video_filename), exist_ok=True)

    # Choose preprocessing pipeline and set input shape accordingly.
    # For actor-critic training we use a frame stack of 4 frames.
    pipeline = config.get("preprocessing_pipeline", "custom")
    if pipeline == "deepmind":
        preprocess_fn = deepmind_preprocess.process_image
        input_shape = (4, 84, 84)
    else:
        preprocess_fn = custom_preprocess.process_image
        input_shape = (4, 7, 7)

    num_actions = config.get("num_actions", 7)

    # Instantiate the actor-critic agent.
    agent = get_agent_class(config, input_shape, num_actions)

    # Load the actor-critic model weights.
    preprocess_name = config.get("preprocessing_pipeline", "custom")
    agent_variant = config.get("agent_variant", "base")
    weight_path = f"../../weights/{preprocess_name}_{agent_variant}_actor_critic.pth"
    agent.model.load_state_dict(torch.load(weight_path, map_location=agent.device))
    agent.model.eval()

    # Prepare evaluation by initializing the frame stack.
    state, _ = env.reset()
    frame_stack = collections.deque(maxlen=4)
    processed_frame = preprocess_fn(state).float()
    for _ in range(4):
        frame_stack.append(processed_frame)

    total_reward = 0
    step = 0
    done = False
    truncated = False

    # Evaluation loop with video recording.
    with imageio.get_writer(video_filename, fps=10) as video:
        # Capture the initial frame.
        frame = env.render()
        video.append_data(frame)

        while not (done or truncated):
            # Stack frames to create the state input.
            state_stack = torch.stack(list(frame_stack), dim=0)  # shape: (4, H, W)
            state_stack = state_stack.unsqueeze(0).to(agent.device)  # add batch dimension

            with torch.no_grad():
                # Assuming the agent model returns (action_logits, state_value)
                action_logits, _ = agent.model(state_stack)
                # Choose the action with highest logit (greedy action).
                action = torch.argmax(action_logits, dim=1).item()

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

            # Update the frame stack.
            processed_next_frame = preprocess_fn(next_state).float()
            frame_stack.append(processed_next_frame)

            # Capture and record the rendered frame.
            frame = env.render()
            video.append_data(frame)

    print("Evaluation complete: reward =", total_reward, "steps =", step)

if __name__ == "__main__":
    main()
