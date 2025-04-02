import gym
import numpy as np
import cv2
from gym.wrappers import atari_preprocessing, frame_stack

class pacman_env:
    def __init__(self, render_mode=None):
        #ori env
        self.env = gym.make('MsPacman-v0', render_mode=render_mode)
        
        #preprocess
        self.env = atari_preprocessing.AtariPreprocessing(
            self.env,
            frame_skip=4,
            grayscale_obs=True, #change rgb to gray (3ch -> 1ch)
            scale_obs=True,     #change 0~255 to 0~1
            terminal_on_life_loss=True, #end game = die
        )

        self.env = frame_stack.FrameStack(
            self.env,
            num_stack=4,
        )
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self):
        return self.env.reset()
            
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
        
    @staticmethod
    def preprocess_observe(observe):
        if isinstance(observe, np.ndarray):
            return observe/255.0
        else:
            return np.array(observe)/255.0        
        
