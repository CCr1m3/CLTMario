import torch
import yaml
import time

from utils.env_wrapper import make_env
from utils.model import MarioPolicyNet

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def test():
    config = load_config()
    device = torch.device(config["device"])

    env = make_env(config["env"])
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = MarioPolicyNet(obs_shape, n_actions).to(device)
    model.load_state_dict(torch.load(config["train"]["save_path"], map_location=device))
    model.eval()

    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            logits = model(state)
            action = torch.argmax(logits, dim=1).item()

        next_state, reward, done, _ = env.step(action)
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward += reward
        env.render()
        time.sleep(0.02)

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test()