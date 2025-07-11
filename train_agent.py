import os
import glob
import random
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
import gym
from gym.wrappers import FrameStack
from utils.env_wrapper import PreprocessFrame, SkipFrame
from nes_py.wrappers import JoypadSpace
from utils.model import CNNPolicy
from utils.actions import CUSTOM_MOVEMENT
import multiprocessing as mp
import copy

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, delta_x, delta_y):
        self.buffer.append((state, action, reward, next_state, done, delta_x, delta_y))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, delta_xs, delta_ys = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        delta_xs = torch.tensor(delta_xs, dtype=torch.float32)
        delta_ys = torch.tensor(delta_ys, dtype=torch.float32)
        return states, actions, rewards, next_states, dones, delta_xs, delta_ys

    def __len__(self):
        return len(self.buffer)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_env(stage, env_config):
    env_id = f"SuperMarioBros-{stage[0]}-{stage[1]}-v0"
    try:
        env = gym.make(env_id)
    except gym.error.Error:
        env = gym.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    if env_config.get("gray_scale", False):
        env = PreprocessFrame(env, shape=tuple(env_config["resize_shape"]), grayscale=True)
    if env_config.get("frame_skip", 1) > 1:
        env = SkipFrame(env, skip=env_config["frame_skip"])
    env = FrameStack(env, env_config.get("frame_stack", 4))
    return env

def calculate_shaped_reward(info, prev_x, prev_w, max_x, subarea):
    x_pos = info["x_pos"]
    delta_x = x_pos - prev_x
    max_x = max(max_x, x_pos)
    shaped_reward = 0
    # Max distance reached
    if max_x == x_pos and x_pos != prev_x and not subarea:
        shaped_reward += delta_x
    # check entering subarea
    if np.abs(delta_x) > 15 and x_pos < 100:
        subarea = not subarea
    elif np.abs(delta_x) > 15 and subarea:
        subarea = not subarea
    # maintaining sprint
    if delta_x > 7 and delta_x < 15:
        shaped_reward += 1
    # flag reached
    if info.get("flag_get", False):
        shaped_reward += 500
    # warpzone
    cur_world = info["world"]
    if cur_world != prev_w:
        shaped_reward += (cur_world - prev_w) * 12000
    return shaped_reward, x_pos, cur_world, subarea

def agent_worker(agent_id, config, shared_best_weights, shared_best_score, agent_hyperparams, result_queue):
    # Set up agent-specific objects
    data_path = config["data"]["expert_data_path"]
    batch_size = agent_hyperparams["batch_size"]
    learning_rate = agent_hyperparams["lr"]
    lambda_bc = agent_hyperparams["lambda_bc"]
    lambda_bc_decay = agent_hyperparams["lambda_bc_decay"]
    replay_buffer_size = config["training"]["replaybuffersize"]
    gamma = config["training"]["gamma"]
    dqn_steps_per_epoch = config["training"]["dqn_steps_per_epoch"]
    epsilon_start = config["training"]["epsilon_start"]
    epsilon_end = config["training"]["epsilon_end"]
    epsilon_decay = config["training"]["epsilon_decay"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNPolicy(len(CUSTOM_MOVEMENT)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    writer = SummaryWriter(log_dir=f"runs/mario_training_agent_{agent_id}")
    global_step = 0

    num_epochs = config["training"]["epochs"]
    best_score = -float('inf')
    best_weights = None

    try:
        for epoch in range(num_epochs):
            print(f"[==========EPOCH {epoch + 1}==========]")
            # --- Load expert data for BC ---
            all_files = glob.glob(os.path.join(data_path, "*", "*.pt"))
            if not all_files:
                print(f"[Agent {agent_id}] No expert data found.")
                return

            selected = random.choice(all_files)
            stage_folder = os.path.basename(os.path.dirname(selected))
            try:
                world, level = map(int, stage_folder.split("-"))
                stage = (world, level)
            except ValueError:
                print(f"[Agent {agent_id}] Invalid stage format in folder name: {stage_folder}")
                return

            dataset = torch.load(selected)
            states = dataset["states"]
            actions = dataset["actions"]
            delta_x = dataset["delta_x"]
            delta_y = dataset["delta_y"]

            # Fill replay buffer with expert data (use delta_x, delta_y)
            for s, a, dx, dy in zip(states, actions, delta_x, delta_y):
                replay_buffer.push(s.float() / 255.0, a.cpu().item(), 0.0, s.float() / 255.0, False, dx.item(), dy.item())

            lambda_bc_epoch = lambda_bc * (lambda_bc_decay ** epoch)
            writer.add_scalar("Lambda/BC", lambda_bc_epoch, epoch)
            writer.add_scalar("Hyperparams/LearningRate", agent_hyperparams["lr"], epoch)
            writer.add_scalar("Hyperparams/LambdaBC", agent_hyperparams["lambda_bc"], epoch)
            writer.add_scalar("Hyperparams/LambdaBC_Epoch", lambda_bc_epoch, epoch)

            if len(replay_buffer) < batch_size:
                continue

            # --- Behavior Cloning Phase ---
            expert_dataset = TensorDataset(
                states.float() / 255.0,
                actions,
                delta_x,
                delta_y
            )
            expert_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
            for batch_states, batch_actions, batch_delta_x, batch_delta_y in expert_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_extra = torch.stack([batch_delta_x, batch_delta_y], dim=1).to(device)
                logits = model(batch_states, extra=batch_extra)
                bc_loss = F.cross_entropy(logits, batch_actions)
                optimizer.zero_grad()
                bc_loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/BC", bc_loss.item(), global_step)
                global_step += 1

            # --- Exploration Phase ---
            env = make_env(stage, config["env"])
            state = env.reset()
            noop_action = 0  # or whatever index is NOOP in your action space
            next_state_img, _, done, info = env.step(noop_action)
            next_state_img = np.array(next_state_img)  # Ensure it's a numpy array
            if next_state_img.shape[-1] == 1:
                next_state_img = np.squeeze(next_state_img, axis=-1)  # (4, 120, 128)
            state = torch.tensor(next_state_img).unsqueeze(0).float().to(device) / 255.0  # (1, 4, 120, 128)
            done = False
            prev_x = info["x_pos"]
            prev_y = info["y_pos"]
            prev_w = info["world"]
            max_x = prev_x
            subarea = False
            episode_reward = 0

            while not done:
                # Calculate deltas for this step
                delta_x_val = info["x_pos"] - prev_x
                delta_y_val = info["y_pos"] - prev_y
                extra = torch.tensor([[delta_x_val, delta_y_val]], dtype=torch.float32).to(device)

                with torch.no_grad():
                    logits = model(state, extra=extra)
                    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** epoch))
                    if random.random() < epsilon:
                        action = random.randint(0, logits.shape[1] - 1)
                    else:
                        if random.random() < lambda_bc_epoch:
                            probs = torch.softmax(logits, dim=1)
                            action = torch.multinomial(probs, num_samples=1).item()
                        else:
                            action = torch.argmax(logits, dim=1).item()
                next_state_img, _, done, info = env.step(action)
                next_state_img = np.array(next_state_img)
                if next_state_img.shape[-1] == 1:
                    next_state_img = np.squeeze(next_state_img, axis=-1)
                next_state = torch.tensor(next_state_img).unsqueeze(0).float().to(device) / 255.0  # (1, 4, 120, 128)

                # Rewards
                shaped_reward, x_pos, cur_world, subarea = calculate_shaped_reward(
                    info, prev_x, prev_w, max_x, subarea
                )
                # Update prev_x, prev_y, prev_w, max_x for next step
                prev_x = x_pos
                prev_y = info["y_pos"]
                prev_w = cur_world
                max_x = max(max_x, x_pos)

                episode_reward += shaped_reward
                replay_buffer.push(state.squeeze(0), action, shaped_reward, next_state.squeeze(0), done, delta_x_val, delta_y_val)
                state = next_state

            # --- DQN Phase ---
            for _ in range(dqn_steps_per_epoch):
                if len(replay_buffer) < batch_size:
                    break
                states, actions, rewards, next_states, dones, delta_xs, delta_ys = replay_buffer.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                batch_extra = torch.stack([delta_xs, delta_ys], dim=1).to(device)

                q_values = model(states, extra=batch_extra)
                with torch.no_grad():
                    next_q_values = model(next_states, extra=batch_extra)
                    max_next_q = next_q_values.max(1)[0]
                    q_target = rewards + gamma * (1 - dones) * max_next_q

                q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                dqn_loss = F.mse_loss(q_pred, q_target)

                optimizer.zero_grad()
                dqn_loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/DQN", dqn_loss.item(), global_step)
                writer.add_scalar("Loss/Total", lambda_bc_epoch * bc_loss + (1 - lambda_bc_epoch) * dqn_loss.item(), global_step)
                global_step += 1

            # --- Evaluation ---
            eval_env = make_env(stage, config["env"])
            eval_state = eval_env.reset()
            noop_action = 0
            eval_next_state_img, _, eval_done, eval_info = eval_env.step(noop_action)
            eval_next_state_img = np.array(eval_next_state_img)
            if eval_next_state_img.shape[-1] == 1:
                eval_next_state_img = np.squeeze(eval_next_state_img, axis=-1)
            eval_state = torch.tensor(eval_next_state_img).unsqueeze(0).float().to(device) / 255.0
            eval_done = False
            prev_x = eval_info["x_pos"]
            prev_y = eval_info["y_pos"]
            prev_w = eval_info["world"]
            max_x = prev_x
            subarea = False
            eval_reward = 0
            while not eval_done:
                delta_x_val = eval_info["x_pos"] - prev_x
                delta_y_val = eval_info["y_pos"] - prev_y
                eval_extra = torch.tensor([[delta_x_val, delta_y_val]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    eval_logits = model(eval_state, extra=eval_extra)
                    eval_action = torch.argmax(eval_logits, dim=1).item()
                eval_next_state_img, _, eval_done, eval_info = eval_env.step(eval_action)
                eval_next_state_img = np.array(eval_next_state_img)
                if eval_next_state_img.shape[-1] == 1:
                    eval_next_state_img = np.squeeze(eval_next_state_img, axis=-1)
                eval_next_state = torch.tensor(eval_next_state_img).unsqueeze(0).float().to(device) / 255.0
                shaped_reward, x_pos, cur_world, subarea = calculate_shaped_reward(
                    eval_info, prev_x, prev_w, max_x, subarea
                )
                eval_reward += shaped_reward
                prev_x = x_pos
                prev_y = eval_info["y_pos"]
                prev_w = cur_world
                max_x = max(max_x, x_pos)
                eval_state = eval_next_state

            writer.add_scalar("Eval/Reward", eval_reward, epoch)

            # --- PBT: Share best agent ---
            if eval_reward > best_score:
                best_score = eval_reward
                best_weights = copy.deepcopy(model.state_dict())
                shared_best_weights[agent_id] = best_weights
                shared_best_score[agent_id] = best_score

            # --- PBT: Exploit/Explore ---
            if epoch > 0 and epoch % config["training"]["pbt_exploit_interval"] == 0:
                all_scores = list(shared_best_score.values())
                max_score = max(all_scores)
                if best_score < max_score:
                    best_agent_id = all_scores.index(max_score)
                    model.load_state_dict(shared_best_weights[best_agent_id])
                    agent_hyperparams["lr"] *= np.random.uniform(0.8, 1.2)
                    agent_hyperparams["lambda_bc"] *= np.random.uniform(0.8, 1.2)
                    agent_hyperparams["lr"] = float(np.clip(agent_hyperparams["lr"], 1e-5, 1e-2))
                    agent_hyperparams["lambda_bc"] = float(np.clip(agent_hyperparams["lambda_bc"], 0.01, 1.0))
                    optimizer = torch.optim.Adam(model.parameters(), lr=agent_hyperparams["lr"])
                    print(f"[Agent {agent_id}] Exploited best agent {best_agent_id} and mutated hyperparams: lr={agent_hyperparams['lr']:.5f}, lambda_bc={agent_hyperparams['lambda_bc']:.3f}")

    except KeyboardInterrupt:
        print(f"\n[Agent {agent_id}] Training interrupted by user. Saving model...")
        torch.save(model.state_dict(), f"trained_model_agent_{agent_id}_interrupt.pt")
        result_queue.put((agent_id, best_score))
        return

    finally:
        # Always save the latest model at the end or on interrupt
        torch.save(model.state_dict(), f"trained_model_agent_{agent_id}.pt")
        result_queue.put((agent_id, best_score))

def population_based_training():
    config = load_config()
    num_agents = config["training"]["pbt_population_size"]
    manager = mp.Manager()
    shared_best_weights = manager.dict()
    shared_best_score = manager.dict()
    result_queue = manager.Queue()

    agent_hyperparams_list = []
    for _ in range(num_agents):
        agent_hyperparams_list.append({
            "batch_size": config["training"]["batch_size"],
            "lr": np.random.uniform(1e-4, 1e-3),
            "lambda_bc": np.random.uniform(0.5, 1.0),
            "lambda_bc_decay": config["training"]["lambda_bc_decay"]
        })

    processes = []
    for agent_id in range(num_agents):
        shared_best_weights[agent_id] = None
        shared_best_score[agent_id] = -float('inf')
        p = mp.Process(target=agent_worker, args=(
            agent_id, config, shared_best_weights, shared_best_score, agent_hyperparams_list[agent_id], result_queue
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda x: x[1], reverse=True)
    print("Population-Based Training complete. Best agents:")
    for agent_id, score in results:
        print(f"Agent {agent_id}: Best Eval Reward = {score}")

    # === Save the best agent's model to the configured path ===
    if results:
        best_agent_id = results[0][0]
        best_model_path = f"trained_model_agent_{best_agent_id}.pt"
        save_path = config["training"].get("save_path", "models/combined_model.pt")
        # Load the best model's weights and save to the desired path
        best_model = CNNPolicy(len(CUSTOM_MOVEMENT))
        best_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model.state_dict(), save_path)
        print(f"\nBest agent's model saved to {save_path}")

if __name__ == "__main__":
    population_based_training()