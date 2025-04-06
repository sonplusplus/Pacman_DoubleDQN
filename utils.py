import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import cv2
import gymnasium as gym
from datetime import datetime

class FrameStack(gym.ObservationWrapper):
    """
    Stack n_frames last frames.
    """
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(n_frames, shp[1], shp[2]),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def observation(self, observation):
        self.frames.append(observation)
        return self._get_obs()

    def _get_obs(self):
        return np.array(list(self.frames))

class SkipFrame(gym.Wrapper):
    """
    Skip frames by repeating actions
    """
    def __init__(self, env, skip):
        super().__init__(env) 
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info

class PreprocessFrame(gym.ObservationWrapper):
    """
    Preprocess frames (grayscale, resize)
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255,
            shape=(1, self.height, self.width),
            dtype=np.uint8
        )

    def observation(self, observation):
        # Convert to grayscale
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        # Resize
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # Add channel dimension
        frame = np.expand_dims(frame, axis=0)
        
        # Apply max pooling (can improve feature extraction)
        # frame = np.maximum(frame, frame_prev)
        
        return frame

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Convert image to PyTorch tensor format (C, H, W)
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_shape[0], obs_shape[1], obs_shape[2]),
            dtype=np.float32
        )

    def observation(self, observation):
        return np.float32(observation / 255.0)

class RewardProcessor(gym.Wrapper):
    """
    Process rewards to make learning more stable
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Clip rewards to be between -1 and 1
        reward = np.clip(reward, -1, 1)
        
        return obs, reward, terminated, truncated, info

def make_env(env_name, frame_skip=4, frame_stack=4, render_mode=None):
    """
    Create environment with all necessary wrappers
    
    Args:
        env_name: Name of the environment
        frame_skip: Number of frames to skip
        frame_stack: Number of frames to stack
        render_mode: Render mode for visualization
        
    Returns:
        Wrapped environment
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = SkipFrame(env, frame_skip)
    env = PreprocessFrame(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, frame_stack)
    env = RewardProcessor(env)
    return env

class Logger:
    """
    Logger for tracking metrics during training
    """
    def __init__(self, log_dir, metrics):
        self.log_dir = log_dir
        self.metrics = {metric: [] for metric in metrics}
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for unique file names
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log(self, metric, value):
        """
        Log a metric value
        
        Args:
            metric: Name of the metric
            value: Value to log
        """
        if metric in self.metrics:
            self.metrics[metric].append(value)
    
    def save_metrics(self):
        """
        Save metrics to file
        """
        for metric, values in self.metrics.items():
            if values:
                # Save to CSV
                np.savetxt(
                    os.path.join(self.log_dir, f"{metric}_{self.timestamp}.csv"),
                    np.array(values),
                    delimiter=','
                )
    
    def plot_metrics(self, save=True, show=False):
        """
        Plot metrics
        
        Args:
            save: Whether to save the plots
            show: Whether to show the plots
        """
        for metric, values in self.metrics.items():
            if values:
                plt.figure(figsize=(10, 5))
                plt.plot(values)
                plt.title(f"{metric.replace('_', ' ').capitalize()}")
                plt.xlabel("Episode")
                plt.ylabel("Value")
                plt.grid(True)
                
                if save:
                    plt.savefig(os.path.join(self.log_dir, f"{metric}_{self.timestamp}.png"))
                
                if show:
                    plt.show()
                else:
                    plt.close()

def moving_average(values, window_size):
    """
    Calculate moving average
    
    Args:
        values: List of values
        window_size: Window size for moving average
        
    Returns:
        Moving average values
    """
    if len(values) < window_size:
        return values
    
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(values, weights, 'valid')