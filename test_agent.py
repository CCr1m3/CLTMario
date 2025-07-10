import os
import yaml
import torch
import gym
import numpy as np
from utils.model import CNNPolicy
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import cv2
from utils.actions import CUSTOM_MOVEMENT

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def preprocess(obs):
    obs = cv2.resize(obs, (128, 120))  # Match training size
    obs = np.transpose(obs, (2, 0, 1))  # HWC to CHW
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0

def test():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = config["test"]["env_name"]
    render = config["test"]["render"]
    model_path = config["training"]["model_path"]
    max_steps = config["test"]["max_steps"]

    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, CUSTOM_MOVEMENT)

    model = CNNPolicy().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs = env.reset()
    total_reward = 0
    steps = 0

    # Consistency with training (optional): use Q-values if lambda_bc was small
    lambda_bc = config["training"].get("lambda_bc", 1.0)
    lambda_bc_threshold = 0.5
    use_q_values = lambda_bc < lambda_bc_threshold

    while True:
        if render:
            env.render()

        state = preprocess(obs).to(device)
        with torch.no_grad():
            logits, q_values = model(state)
            if use_q_values:
                action = torch.argmax(q_values, dim=1).item()
            else:
                action = torch.argmax(logits, dim=1).item()

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if done or steps >= max_steps:
            break

    env.close()
    print(f"Test completed. Total reward: {total_reward}, Steps: {steps}")

if __name__ == "__main__":
    test()
