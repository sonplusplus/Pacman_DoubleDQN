# environment.py
import gymnasium as gym
import numpy as np
import cv2
from gymnasium.wrappers import AtariPreprocessing
from collections import deque

class FrameProcessor:
    """
    Class xử lý frame từ môi trường Atari
    """
    def __init__(self, frame_size=(84, 84), frame_skip=4):
        """
        Khởi tạo frame processor
        
        Args:
            frame_size: kích thước frame sau khi resize
            frame_skip: số frame bỏ qua
        """
        self.frame_size = frame_size
        self.frame_skip = frame_skip
    
    def process(self, frame):
        """
        Xử lý một frame
        
        Args:
            frame: array RGB
            
        Returns:
            numpy array: frame đã được xử lý
        """
        # Chuyển thành grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        
        return frame

class FrameStack:
    """
    Class quản lý stack các frame liên tiếp
    """
    def __init__(self, num_frames=4):
        """
        Khởi tạo frame stack
        
        Args:
            num_frames: số frame để stack
        """
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        """
        Reset frame stack với frame mới
        
        Args:
            frame: frame đầu tiên
            
        Returns:
            numpy array: state gồm các frame đã stack
        """
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self._get_state()
    
    def add(self, frame):
        """
        Thêm frame mới vào stack
        
        Args:
            frame: frame mới
            
        Returns:
            numpy array: state mới
        """
        self.frames.append(frame)
        return self._get_state()
    
    def _get_state(self):
        """
        Lấy state từ các frame đã stack
        
        Returns:
            numpy array: state với shape (num_frames, height, width)
        """
        return np.array(self.frames)

class PacmanEnv:
    """
    Môi trường wrapper cho Pacman (Atari)
    """
    def __init__(self, env_name="MsPacmanNoFrameskip-v4", render_mode=None, stack_frames=4):
        """
        Khởi tạo môi trường Pacman
        
        Args:
            env_name: tên môi trường
            render_mode: chế độ render (None, 'human', 'rgb_array')
            stack_frames: số frame để stack
        """
        # Khởi tạo môi trường gốc
        self.env = gym.make(env_name, render_mode=render_mode)
        
        # Wrap môi trường với AtariPreprocessing
        self.env = AtariPreprocessing(
            self.env,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            scale_obs=False,  # Giữ giá trị pixel trong khoảng [0, 255]
            noop_max=30
        )
        
        # Frame processor và stacker
        self.processor = FrameProcessor()
        self.stacker = FrameStack(num_frames=stack_frames)
        
        # Lưu trữ không gian hành động và quan sát
        self.action_space = self.env.action_space
        
        # Định nghĩa không gian quan sát mới sau khi stack frame
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255,
            shape=(stack_frames, 84, 84),
            dtype=np.uint8
        )
    
    def reset(self):
        """
        Reset môi trường
        
        Returns:
            numpy array: state ban đầu
            dict: thông tin bổ sung
        """
        obs, info = self.env.reset()
        # Reset frame stacker với frame đầu tiên
        stacked_state = self.stacker.reset(obs)
        return stacked_state, info
    
    def step(self, action):
        """
        Thực hiện một hành động trong môi trường
        
        Args:
            action: hành động để thực hiện
            
        Returns:
            tuple: (state mới, reward, done, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Thêm frame mới vào stack
        stacked_state = self.stacker.add(obs)
        return stacked_state, reward, terminated, truncated, info
    
    def render(self):
        """
        Render môi trường
        
        Returns:
            numpy array hoặc None: frame đã render
        """
        return self.env.render()
    
    def close(self):
        """Đóng môi trường"""
        self.env.close()