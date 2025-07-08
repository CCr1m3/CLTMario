import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import gym
import yaml
import os
import numpy as np

from utils.env_wrapper import make_env
from utils.model import MarioPolicyNet


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_expert_data(path):
    data = torch.load(path)
    return TensorDataset(data["states"], data["actions"])


def train():
    config = load_config()
    device = torch.device(config["device"])

    env = make_env(config["env"])
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = MarioPolicyNet(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    writer = SummaryWriter(log_dir=config["log"]["log_dir"])

    # Load BC dataset
    expert_dataset = load_expert_data(config["data"]["expert_data_path"])
    expert_loader = DataLoader(expert_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    expert_iter = iter(expert_loader)

    imitation_weight = config["train"]["imitation_weight"]
    gamma = config["train"]["gamma"]

    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    timestep = 0
    episode_reward = 0
    episode = 0

    while timestep < config["train"]["total_timesteps"]:
        logits = model(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        # RL Loss (Policy Gradient Estimate)
        advantage = torch.tensor([reward], dtype=torch.float32, device=device)
        log_prob = dist.log_prob(action)
        rl_loss = -log_prob * advantage

        # BC Loss (from batch)
        bc_loss = None
        try:
            expert_states, expert_actions = next(expert_iter)
        except StopIteration:
            expert_iter = iter(expert_loader)
            expert_states, expert_actions = next(expert_iter)

        expert_states = expert_states.to(device)
        expert_actions = expert_actions.to(device)
        bc_logits = model(expert_states)
        bc_loss = nn.CrossEntropyLoss()(bc_logits, expert_actions)

        # Total loss
        total_loss = rl_loss + imitation_weight * bc_loss if bc_loss is not None else rl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        state = next_state_tensor
        episode_reward += reward
        timestep += 1

        if done:
            writer.add_scalar("Reward/Episode", episode_reward, episode)
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            episode += 1

        if timestep % config["train"]["log_interval"] == 0:
            print(f"Timestep {timestep}, Loss: {total_loss.item():.4f}")
            writer.add_scalar("Loss/Total", total_loss.item(), timestep)

        if timestep % config["log"]["save_interval"] == 0:
            torch.save(model.state_dict(), config["train"]["save_path"])
            print(f"Saved model at timestep {timestep}")

    env.close()
    writer.close()


if __name__ == "__main__":
    train()