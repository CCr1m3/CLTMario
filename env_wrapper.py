import gym
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

def make_env(env_config):
    env = gym.make(env_config["name"])

    if env_config.get("gray_scale", False):
        env = GrayScaleObservation(env, keep_dim=True)

    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, env_config.get("frame_stack", 4))

    return env