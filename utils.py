# utils.py
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import random
import cv2
from datetime import datetime

def set_seed(seed):
    """
    Đặt seed cho tất cả các nguồn ngẫu nhiên
    
    Args:
        seed: giá trị seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_dirs(dirs):
    """
    Tạo các thư mục nếu chưa tồn tại
    
    Args:
        dirs: list các đường dẫn thư mục
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_timestamp():
    """
    Lấy timestamp hiện tại
    
    Returns:
        str: timestamp theo định dạng YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_learning_curve(x, y, title, ylabel, filename):
    """
    Vẽ đồ thị học tập
    
    Args:
        x: giá trị trục x
        y: giá trị trục y
        title: tiêu đề đồ thị
        ylabel: nhãn trục y
        filename: tên file để lưu
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_rewards(episode_rewards, avg_rewards, window_size=100, filename='rewards_plot.png'):
    """
    Vẽ đồ thị phần thưởng
    
    Args:
        episode_rewards: list phần thưởng của từng episode
        avg_rewards: list phần thưởng trung bình
        window_size: kích thước cửa sổ để tính trung bình
        filename: tên file để lưu
    """
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.5, label='Episode Reward')
    plt.plot(avg_rewards, label=f'Average Reward (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def display_frames_as_gif(frames, filename='game.gif', fps=30):
    """
    Lưu một chuỗi frame dưới dạng GIF
    
    Args:
        frames: list các frame
        filename: tên file để lưu
        fps: số frame mỗi giây
    """
    from PIL import Image
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Tạo gif từ frames
    frames_pil = [Image.fromarray(frame) for frame in frames]
    frames_pil[0].save(
        filename,
        save_all=True,
        append_images=frames_pil[1:],
        duration=1000/fps,
        loop=0
    )

def visualize_q_values(model, state, n_actions, device, filename='q_values.png'):
    """
    Trực quan hóa Q-values cho một state
    
    Args:
        model: DQN model
        state: state để đánh giá
        n_actions: số lượng action
        device: thiết bị (cpu/cuda)
        filename: tên file để lưu
    """
    # Chuyển state thành tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # Lấy Q-values
    with torch.no_grad():
        q_values = model(state_tensor).cpu().numpy()[0]
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    actions = np.arange(n_actions)
    plt.bar(actions, q_values)
    plt.xlabel('Action')
    plt.ylabel('Q-value')
    plt.title('Q-values for each action')
    plt.xticks(actions)
    plt.grid(True, axis='y')
    plt.savefig(filename)
    plt.close()