import os
import glob
import random
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import numpy as np
from utils.model import CNNPolicy
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from utils.env_wrapper import wrap_env
from utils.actions import CUSTOM_MOVEMENT

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action):
        self.buffer.append((state, action))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    data_path = config["data"]["expert_data_path"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["epochs"]
    learning_rate = config["training"]["lr"]
    initial_lambda_bc = config["training"]["lambda_bc"]
    decay_rate = config["training"]["lambda_bc_decay"]
    replay_buffer_size = config["training"]["replaybuffersize"]
    gamma = config["training"]["gamma"]
    dqn_steps_per_epoch = config["training"]["dqn_steps_per_epoch"]
    epsilon_start = config["training"]["epsilon_start"]
    epsilon_end = config["training"]["epsilon_end"]
    epsilon_decay = config["training"]["epsilon_decay"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_size)

    for epoch in range(num_epochs):
        all_files = glob.glob(os.path.join(data_path, "*", "*.pt"))
        if not all_files:
            print("No expert data found.")
            return

        selected = random.choice(all_files)
        stage_folder = os.path.basename(os.path.dirname(selected))
        try:
            world, level = map(int, stage_folder.split("-"))
            stage = (world, level)
        except ValueError:
            print(f"Invalid stage format in folder name: {stage_folder}")
            return

        # Load a single expert trajectory
        dataset = torch.load(selected)
        states = dataset["states"]
        actions = dataset["actions"]

        for s, a in zip(states, actions):
            replay_buffer.push(s.float() / 255.0, a.item(), 0.0, s.float() / 255.0, False)

        lambda_bc = initial_lambda_bc * (decay_rate ** epoch)
        use_q_values = lambda_bc < 0.3
        print(f"[Epoch {epoch+1}] Stage: {stage}, Lambda BC: {lambda_bc:.5f}, Replay Buffer Size: {len(replay_buffer)}")

        if len(replay_buffer) < batch_size:
            continue

        for _ in range(dqn_steps_per_epoch):
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            logits, q_values = model(states)
            bc_loss = F.cross_entropy(logits, actions)

            with torch.no_grad():
                _, next_q_values = model(next_states)
                max_next_q = next_q_values.max(1)[0]
                q_target = rewards + gamma * (1 - dones) * max_next_q

            q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            dqn_loss = F.mse_loss(q_pred, q_target)

            loss = lambda_bc * bc_loss + (1 - lambda_bc) * dqn_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Exploration Phase: Run game using current policy on same stage
        env_id = f"SuperMarioBros-{stage[0]}-{stage[1]}-v0"
        try:
            env = gym.make(env_id)
        except gym.error.Error:
            print(f"Environment {env_id} not found, defaulting to SuperMarioBros-1-1-v0")
            env = gym.make("SuperMarioBros-1-1-v0")

        env = JoypadSpace(env, CUSTOM_MOVEMENT)
        env = wrap_env(env)
        state = env.reset()
        state = torch.tensor(state).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        done = False

        # Rewards
        prev_x = info["x_pos"]
        prev_w = info["world"]
        max_x = prev_x
        subarea = False

        while not done:
            with torch.no_grad():
                logits, q_values = model(state)
                epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** epoch))
                if random.random() < epsilon:
                    action = random.randint(0, logits.shape[1] - 1)
                else:
                    if use_q_values:
                        action = torch.argmax(q_values, dim=1).item()
                    else:
                        action = torch.argmax(logits, dim=1).item()

            next_state_img, _, done, info = env.step(action)
            next_state = torch.tensor(next_state_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

            # Calculate reward components
            x_pos = info["x_pos"]
            delta_x = x_pos - prev_x
            max_x = max(max_x, x_pos)

            shaped_reward = 0

            # reward for new max distance record 
            if max_x == x_pos and x_pos != prev_x and not subarea:
                shaped_reward += delta_x

            # disable max distance reward for subarea
            if np.abs(delta_x) > 20 and x_pos < 60:
                subarea = not subarea

            # reward for moving right
            if delta_x > 7 and delta_x < 15:
                shaped_reward += 1


            # Bonus if flag reached
            if info["flag_get"]:
                shaped_reward += 500

            # Extreme Bonus for Warpzone
            cur_world = info["world"]
            if cur_world != prev_w:
                shaped_reward += (cur_world - prev_w) * 500

            replay_buffer.push(state.squeeze(0), action, shaped_reward, next_state.squeeze(0), done)
            prev_x = x_pos
            state = next_state

        print(f"Epoch {epoch+1}/{num_epochs} complete. Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), config["training"].get("model_path", "trained_model.pt"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()