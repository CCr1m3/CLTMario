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
import itertools

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, prev_action, action, reward, next_state, done, last_delta_x, last_delta_y, delta_x, delta_y):
        self.buffer.append((state, prev_action, action, reward, next_state, done, last_delta_x, last_delta_y, delta_x, delta_y))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, prev_actions, actions, rewards, next_states, dones, last_delta_xs, last_delta_ys, delta_xs, delta_ys = zip(*batch)
        states = torch.stack(states)
        prev_actions = torch.tensor(prev_actions)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        last_delta_xs = torch.tensor(delta_xs, dtype=torch.float32)
        last_delta_ys = torch.tensor(delta_ys, dtype=torch.float32)
        delta_xs = torch.tensor(delta_xs, dtype=torch.float32)
        delta_ys = torch.tensor(delta_ys, dtype=torch.float32)
        return states, prev_actions, actions, rewards, next_states, dones, last_delta_xs, last_delta_ys, delta_xs, delta_ys

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

def calculate_shaped_reward(info, done, prev_x, prev_y, prev_w, max_x, subarea, time_subarea):
    x_pos = info["x_pos"]
    y_pos = info["y_pos"]
    if subarea:
        x_pos_min = (time_subarea - info["time"]) * 15
    else:
        x_pos_min = (400 - info["time"]) * 30
    delta_x = x_pos - prev_x
    delta_y = y_pos - prev_y
    max_x = max(max_x, x_pos)
    shaped_reward = 0
    # Max distance reached
    if max_x == x_pos and x_pos != prev_x and not subarea:
        shaped_reward += delta_x * 0.01
    # check entering subarea
    if np.abs(delta_x) > 15 and x_pos < 100:
        shaped_reward += 1
        subarea = not subarea
        time_subarea = info["time"]
    elif np.abs(delta_x) > 15 and subarea:
        shaped_reward += 1
        subarea = not subarea
    # maintaining sprint
    if abs(delta_x) > 7 and abs(delta_x) < 15:
        shaped_reward += 0.01
    #elif abs(delta_x) > 3 and abs(delta_x) < 7:
    #    shaped_reward += 0.005
    # flag reached
    if info.get("flag_get", False):
        shaped_reward += 5
    # warpzone
    cur_world = info["world"]
    if cur_world != prev_w:
        shaped_reward += (cur_world - prev_w) * 120
    # died
    if done and not info["flag_get"]:
        shaped_reward -= 3
    # not moving
    if x_pos < x_pos_min:
        shaped_reward -= 0.01
    # prior rewards
    #if delta_x == 0 and delta_y == 0:
    #    shaped_reward -= 0.01
    #if shaped_reward == 0:
    #    shaped_reward -= 0.01
    return shaped_reward, max_x, subarea, time_subarea

def bc_phase(agent_id, bc_epochs, optimizer, writer, device, model, expert_dataset, batch_size, global_step):
    for bc_epoch in range(1, bc_epochs+1):
        expert_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
        for batch_states, batch_prev_actions, batch_actions, batch_delta_x, batch_delta_y in expert_loader:
            batch_states = batch_states.to(device)
            batch_prev_actions = batch_prev_actions.to(device)
            batch_actions = batch_actions.to(device)
            batch_extra = torch.stack([batch_delta_x, batch_delta_y], dim=1).to(device)
            batch_prev_actions_onehot = F.one_hot(batch_prev_actions, num_classes=len(CUSTOM_MOVEMENT)).float().to(device)
            batch_extra = torch.cat([batch_extra, batch_prev_actions_onehot], dim=1)
            logits = model(batch_states, extra=batch_extra)
            bc_loss = F.cross_entropy(logits, batch_actions)
            optimizer.zero_grad()
            bc_loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/BC", bc_loss.item(), global_step)
            global_step += 1
            # Debugging
            """
            with torch.no_grad():
                logits = model(batch_states, extra=batch_extra)
                predicted_actions = torch.argmax(logits, dim=1)
                accuracy = (predicted_actions == batch_actions).float().mean().item()
                print(f"BC accuracy: {accuracy:.2%}")
                """
        if bc_epoch % 5 == 0:
            print(f"Agent {agent_id}: BC_Epoch {bc_epoch} done.")
    return model, global_step

def agent_worker(agent_id, config, shared_best_weights, shared_best_score, agent_hyperparams, result_queue):
    # Set up agent-specific objects
    data_path = config["data"]["expert_data_path"]
    batch_size = agent_hyperparams["batch_size"]
    learning_rate = agent_hyperparams["lr"]
    lambda_bc = agent_hyperparams["lambda_bc"]
    lambda_bc_decay = agent_hyperparams["lambda_bc_decay"]
    replay_buffer_size = config["training"]["replaybuffersize"]
    gamma = config["training"]["gamma"]
    explorations_per_epoch = config["training"]["exploration_episodes_per_epoch"]
    dqn_steps_per_epoch = config["training"]["dqn_steps_per_epoch"]
    epsilon = config["training"]["epsilon_start"]
    epsilon_end = config["training"]["epsilon_end"]
    epsilon_decay = config["training"]["epsilon_decay"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNPolicy(len(CUSTOM_MOVEMENT)).to(device)
    target_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    writer = SummaryWriter(log_dir=f"runs/mario_training_agent_{agent_id}")
    global_step = 0

    num_epochs = config["training"]["epochs"]
    bc_epochs = config["training"]["bc_epochs"]
    best_score = -float('inf')
    best_weights = None

    all_files = glob.glob(os.path.join(data_path, "*", "*.pt"))
    if not all_files:
        print(f"[Agent {agent_id}] No expert data found.")
        return

    try:
        # --- Behaviour Cloning Phase ---
        print(f"Agent {agent_id}: Starting Behavior Cloning phase")

        states = []
        prev_actions = []
        actions = []
        delta_x = []
        delta_y = []
        for file in all_files:
            dataset = torch.load(file)
            states.append(dataset["states"])
            prev_actions.append(dataset["prev_actions"])
            actions.append(dataset["actions"])
            delta_x.append(dataset["delta_x"])
            delta_y.append(dataset["delta_y"])
        expert_dataset = TensorDataset(
            torch.cat(states, dim=0),
            torch.cat(prev_actions, dim=0),
            torch.cat(actions, dim=0),
            torch.cat(delta_x, dim=0),
            torch.cat(delta_y, dim=0)
        )
        model, global_step = bc_phase(
            agent_id, bc_epochs, optimizer, writer, device, model, expert_dataset, batch_size, global_step
        )
        expert_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
        expert_iter = itertools.cycle(expert_loader)

        for epoch in range(1, num_epochs+1):
            print(f"[==========AGENT {agent_id}: EPOCH {epoch}==========]")
            # --- Load expert data for BC ---
            selected = random.choice(all_files)
            stage_folder = os.path.basename(os.path.dirname(selected))
            try:
                world, level = map(int, stage_folder.split("-"))
                stage = (world, level)
            except ValueError:
                print(f"[Agent {agent_id}] Invalid stage format in folder name: {stage_folder}")
                return
                
            lambda_bc *= lambda_bc_decay ** (epoch - 1)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            agent_hyperparams["lambda_bc"] = lambda_bc
            writer.add_scalar("Hyperparams/LearningRate", agent_hyperparams["lr"], epoch)
            writer.add_scalar("Hyperparams/LambdaBC", agent_hyperparams["lambda_bc"], epoch)
            writer.add_scalar("Hyperparams/Epsilon", epsilon, epoch)

            print(f"Agent {agent_id}: Starting Exploration phase on {stage[0]}-{stage[1]}")
            episode_rewards_this_epoch = []
            env = make_env(stage, config["env"])
            for episode in range(1, explorations_per_epoch + 1):
                state = env.reset()
                noop_action = 0 
                prev_action = noop_action
                next_state_img, _, done, info = env.step(noop_action)
                next_state_img = np.array(next_state_img)
                if next_state_img.shape[-1] == 1:
                    next_state_img = np.squeeze(next_state_img, axis=-1)
                state = torch.tensor(next_state_img).unsqueeze(0).float().to(device) / 255.0
                done = False
                prev_x = info["x_pos"]
                max_x = prev_x
                prev_y = info["y_pos"]
                prev_w = info["world"]
                last_delta_x = 0
                last_delta_y = 0
                extra = torch.tensor([[0, 0]], dtype=torch.float32).to(device)
                prev_action_onehot = F.one_hot(torch.tensor([prev_action]), num_classes=len(CUSTOM_MOVEMENT)).float().to(device)
                extra = torch.cat([extra, prev_action_onehot], dim=1)
                subarea = False
                time_subarea = 0
                episode_reward = 0
                max_steps = config["training"].get("max_steps_per_episode", 1000)

                steps = 0
                while not done and steps < max_steps:
                    with torch.no_grad():
                        logits = model(state, extra=extra)
                        if random.random() < lambda_bc:
                            probs = torch.softmax(logits, dim=1)
                            action = torch.multinomial(probs, num_samples=1).item()
                        else:
                            if random.random() < epsilon:
                                action = random.randint(0, logits.shape[1] - 1)
                            else:
                                action = torch.argmax(logits, dim=1).item()
                    next_state_img, _, done, info = env.step(action)
                    next_state_img = np.array(next_state_img)
                    if next_state_img.shape[-1] == 1:
                        next_state_img = np.squeeze(next_state_img, axis=-1)
                    next_state = torch.tensor(next_state_img).unsqueeze(0).float().to(device) / 255.0
                    delta_x_val = info["x_pos"] - prev_x
                    delta_y_val = info["y_pos"] - prev_y
                    extra = torch.tensor([[delta_x_val, delta_y_val]], dtype=torch.float32).to(device)
                    prev_action_onehot = F.one_hot(torch.tensor([prev_action]), num_classes=len(CUSTOM_MOVEMENT)).float().to(device)
                    extra = torch.cat([extra, prev_action_onehot], dim=1)

                    # Rewards
                    shaped_reward, max_x, subarea, time_subarea = calculate_shaped_reward(
                        info, done, prev_x, prev_y, prev_w, max_x, subarea, time_subarea
                    )
                    #print(f"Agent {agent_id}: Shaped Reward: {shaped_reward}")
                    prev_x = info["x_pos"]
                    prev_y = info["y_pos"]
                    prev_w = info["world"]

                    episode_reward += shaped_reward
                    replay_buffer.push(
                        state.squeeze(0).cpu().clone().detach(),
                        prev_action,
                        action,
                        shaped_reward,
                        next_state.squeeze(0).cpu().clone().detach(),
                        done,
                        last_delta_x,
                        last_delta_y,
                        delta_x_val,
                        delta_y_val
                    )
                    state = next_state
                    last_delta_x = delta_x_val
                    last_delta_y = delta_y_val
                    prev_action = action
                    steps += 1
                    global_step += 1


                    # --- DQN Phase ---
                    for i in range(dqn_steps_per_epoch):
                        if len(replay_buffer) < batch_size:
                            #print("not enough replay_buffer, jumping to next exploration step")
                            continue
                        states, prev_actions, actions, rewards, next_states, dones, last_delta_xs, last_delta_ys, delta_xs, delta_ys = replay_buffer.sample(batch_size)
                        states = states.to(device)
                        prev_actions = prev_actions.to(device)
                        actions = actions.to(device)
                        rewards = rewards.to(device)
                        next_states = next_states.to(device)
                        dones = dones.to(device)
                        batch_extra = torch.stack([last_delta_xs, last_delta_ys], dim=1).to(device)
                        prev_actions_onehot = F.one_hot(prev_actions, num_classes=len(CUSTOM_MOVEMENT)).float().to(device)
                        batch_extra = torch.cat([batch_extra, prev_actions_onehot], dim=1)
                        next_batch_extra = torch.stack([delta_xs, delta_ys], dim=1).to(device)
                        next_actions_onehot = F.one_hot(actions, num_classes=len(CUSTOM_MOVEMENT)).float().to(device)
                        next_batch_extra = torch.cat([next_batch_extra, next_actions_onehot], dim=1)

                        q_values = model(states, extra=batch_extra)
                        with torch.no_grad():
                            next_q_online = model(next_states, extra=next_batch_extra)
                            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                            next_q_target = target_model(next_states, extra=next_batch_extra)
                            max_next_q = next_q_target.gather(1, next_actions).squeeze(1)
                            q_target = rewards + gamma * (1 - dones) * max_next_q

                        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                        dqn_loss = F.mse_loss(q_pred, q_target)
                        
                        batch_states, batch_prev_actions, batch_actions, batch_delta_x, batch_delta_y = next(expert_iter)
                        batch_states = batch_states.to(device)
                        batch_prev_actions = batch_prev_actions.to(device)
                        batch_actions = batch_actions.to(device)
                        batch_extra = torch.stack([batch_delta_x, batch_delta_y], dim=1).to(device)
                        batch_prev_actions_onehot = F.one_hot(batch_prev_actions, num_classes=len(CUSTOM_MOVEMENT)).float().to(device)
                        batch_extra = torch.cat([batch_extra, batch_prev_actions_onehot], dim=1)
                        logits = model(batch_states, extra=batch_extra)
                        bc_loss = F.cross_entropy(logits, batch_actions)

                        total_loss = (1 - lambda_bc) * dqn_loss + lambda_bc * bc_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        writer.add_scalar("Debug/MeanQValue", q_values.mean().item(), global_step)
                        writer.add_scalar("Debug/MaxQValue", q_values.max().item(), global_step)
                        writer.add_scalar("Debug/MinQValue", q_values.min().item(), global_step)
                        writer.add_scalar("Debug/StdQValue", q_values.std().item(), global_step)
                        writer.add_scalar("Loss/DQN", dqn_loss.item(), global_step)
                        writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                        global_step += 1

                        if global_step % 100 == 0:
                            target_model.load_state_dict(model.state_dict())
                        
                        #if i % 250 == 0:
                        #    print(f"Agent {agent_id}: DQN Step {i} done.")
                        
                episode_rewards_this_epoch.append(episode_reward)
                if episode % 5 == 0:
                    print(f"Agent {agent_id}: Episode {episode} done.")

            if episode_rewards_this_epoch:
                avg_reward = np.mean(episode_rewards_this_epoch)
                writer.add_scalar("Train/AverageEpisodeReward", avg_reward, epoch)
                
            # --- Evaluation ---
            print(f"Agent {agent_id}: Starting Evaluation phase on {stage[0]}-{stage[1]}")
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
            time_subarea = 0
            eval_reward = 0
            eval_prev_action = noop_action
            while not eval_done:
                delta_x_val = eval_info["x_pos"] - prev_x
                delta_y_val = eval_info["y_pos"] - prev_y
                eval_extra = torch.tensor([[delta_x_val, delta_y_val]], dtype=torch.float32).to(device)
                eval_prev_action_onehot = F.one_hot(torch.tensor([eval_prev_action]), num_classes=len(CUSTOM_MOVEMENT)).float().to(device)
                eval_extra = torch.cat([eval_extra, eval_prev_action_onehot], dim=1)
                with torch.no_grad():
                    eval_logits = model(eval_state, extra=eval_extra)
                    eval_action = torch.argmax(eval_logits, dim=1).item()
                    #eval_probs = torch.softmax(eval_logits, dim=1)
                    #eval_action = torch.multinomial(eval_probs, num_samples=1).item()
                eval_next_state_img, _, eval_done, eval_info = eval_env.step(eval_action)
                eval_next_state_img = np.array(eval_next_state_img)
                if eval_next_state_img.shape[-1] == 1:
                    eval_next_state_img = np.squeeze(eval_next_state_img, axis=-1)
                eval_next_state = torch.tensor(eval_next_state_img).unsqueeze(0).float().to(device) / 255.0
                shaped_reward, max_x, subarea, time_subarea = calculate_shaped_reward(
                    eval_info, eval_done, prev_x, prev_y, prev_w, max_x, subarea, time_subarea
                )
                eval_reward += shaped_reward
                prev_x = eval_info["x_pos"]
                prev_y = eval_info["y_pos"]
                prev_w = eval_info["world"]
                eval_prev_action = eval_action
                eval_state = eval_next_state

            writer.add_scalar("Eval/Reward", eval_reward, epoch)

            # --- PBT: Share best agent ---
            if eval_reward > best_score:
                best_score = eval_reward
                best_weights = copy.deepcopy(model.state_dict())
                cpu_state_dict = {k: v.cpu() for k, v in best_weights.items()}
                shared_best_weights[agent_id] = cpu_state_dict
                shared_best_score[agent_id] = best_score

            # --- PBT: Exploit/Explore ---
            if epoch > 0 and epoch % config["training"]["pbt_exploit_interval"] == 0:
                all_scores = list(shared_best_score.values())
                max_score = max(all_scores)
                if best_score < max_score:
                    best_agent_id = all_scores.index(max_score)
                    model.load_state_dict(shared_best_weights[best_agent_id])
                    agent_hyperparams["lr"] *= np.random.uniform(0.9, 1.1)
                    agent_hyperparams["lambda_bc"] *= np.random.uniform(1, 1.5)
                    agent_hyperparams["lr"] = float(np.clip(agent_hyperparams["lr"], 1e-5, 1e-3))
                    agent_hyperparams["lambda_bc"] = float(np.clip(agent_hyperparams["lambda_bc"], 0.001, 0.99))
                    optimizer = torch.optim.Adam(model.parameters(), lr=agent_hyperparams["lr"])
                    print(f"[Agent {agent_id}] Exploited best agent {best_agent_id} and mutated hyperparams: lr={agent_hyperparams['lr']:.5f}, lambda_bc={agent_hyperparams['lambda_bc']:.3f}")
                    replay_buffer = ReplayBuffer(replay_buffer_size)

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
            "lr": np.random.uniform(1e-5, 5e-4),
            "lambda_bc": np.random.uniform(0.8, 0.99),
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