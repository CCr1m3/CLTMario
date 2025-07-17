import os
import yaml
import torch
import gym
import numpy as np
from utils.model import CNNPolicy
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from utils.actions import CUSTOM_MOVEMENT
from utils.env_wrapper import PreprocessFrame, SkipFrame
from gym.wrappers import FrameStack
import time

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_test_env(env_name, env_config):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    if env_config.get("gray_scale", False):
        env = PreprocessFrame(env, shape=tuple(env_config["resize_shape"]), grayscale=True)
    if env_config.get("frame_skip", 1) > 1:
        env = SkipFrame(env, skip=env_config["frame_skip"])
    env = FrameStack(env, env_config.get("frame_stack", 4))
    return env

def test():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = config["test"]["env_name"]
    render = config["test"]["render"]
    model_path = config["training"]["save_path"]
    max_steps = config["test"]["max_steps"]
    env_config = config["env"]

    num_attempts = config["test"].get("num_attempts", 1)

    env = make_test_env(env_name, env_config)

    model = CNNPolicy(num_actions=len(CUSTOM_MOVEMENT)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for attempt in range(1, num_attempts + 1):
        print(f"\n=== Playthrough Attempt {attempt} ===")
        obs = env.reset()
        total_reward = 0
        steps = 0

        # Get initial info by taking a NOOP step
        noop_action = 0
        obs, _, done, info = env.step(noop_action)
        obs_arr = np.array(obs)
        if obs_arr.shape[-1] == 1:
            obs_arr = np.squeeze(obs_arr, axis=-1)
        state = torch.tensor(obs_arr).unsqueeze(0).float().to(device) / 255.0

        prev_x = info.get("x_pos", 0)
        prev_y = info.get("y_pos", 0)

        while True:
            if render:
                env.render()
                time.sleep(0.05)

            # Calculate delta_x and delta_y
            cur_x = info.get("x_pos", 0)
            cur_y = info.get("y_pos", 0)
            delta_x = cur_x - prev_x
            delta_y = cur_y - prev_y
            extra = torch.tensor([[delta_x, delta_y]], dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(state, extra=extra)
                probs = torch.softmax(logits, dim=1)
                action = torch.multinomial(probs, num_samples=1).item()
                # action = torch.argmax(logits, dim=1).item()

            obs, reward, done, info = env.step(action)
            obs_arr = np.array(obs)
            if obs_arr.shape[-1] == 1:
                obs_arr = np.squeeze(obs_arr, axis=-1)
            state = torch.tensor(obs_arr).unsqueeze(0).float().to(device) / 255.0

            total_reward += reward
            steps += 1

            prev_x = cur_x
            prev_y = cur_y

            if done or steps >= max_steps:
                break

        print(f"Attempt {attempt} completed. Total reward: {total_reward}, Steps: {steps}")

    env.close()
    print("All playthroughs completed.")

if __name__ == "__main__":
    test()