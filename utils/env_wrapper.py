import gym_super_mario_bros
import gym
import numpy as np
from gym.wrappers import FrameStack
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from utils.actions import CUSTOM_MOVEMENT
from utils.preprocessing import preprocess_frame
from collections import deque
import cv2

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(128, 120), grayscale=True):
        super().__init__(env)
        self.shape = shape
        self.grayscale = grayscale
        self.observation_space = Box(
            low=0, high=255, shape=(shape[1], shape[0], 1 if grayscale else 3), dtype=np.uint8
        )

    def observation(self, obs):
        return preprocess_frame(obs, resize_shape=self.shape, gray=self.grayscale)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

def make_env(env_config):
    env = gym.make(env_config["name"])
    if env_config.get("gray_scale", False):
        env = PreprocessFrame(env, shape=tuple(env_config["resize_shape"]), grayscale=True)
    if env_config.get("frame_skip", 1) > 1:
        env = SkipFrame(env, skip=env_config["frame_skip"])
    env = FrameStack(env, env_config.get("frame_stack", 4))
    return env

def make_env_human(env_config):
    env = gym.make(env_config["name"])
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    if env_config.get("frame_skip", 1) > 1:
        env = SkipFrame(env, skip=env_config["frame_skip"])
    env = FrameStack(env, env_config.get("frame_stack", 4))
    return env