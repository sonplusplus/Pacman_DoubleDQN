# test.py
import argparse
import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from environment import PacmanEnv
from agent import DoubleDQNAgent
from utils import set_seed, create_dirs, get_timestamp, display_frames_as_gif, visualize_q_values
from config import TestConfig, Config

def parse_args():
    """Parse các tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description="Test trained Double DQN for Pacman")
    
    # Môi trường
    parser.add_argument("--env", type=str, default=Config.ENV_NAME, 
                        help="Atari environment name")
    parser.add_argument("--stack-frames", type=int, default=Config.STACK_FRAMES,
                        help="Number of frames to stack")
    
    # Model
    parser.add_argument("--model-path", type=str, required=True if TestConfig.MODEL_PATH is None else False,
                        default=TestConfig.MODEL_PATH,
                        help="Path to saved model")
    
    # Testing parameters
    parser.add_argument("--episodes", type=int, default=TestConfig.TEST_EPISODES,
                        help="Number of episodes to test")
    parser.add_argument("--render", action="store_true", default=TestConfig.RENDER,
                        help="Render the environment")
    parser.add_argument("--record", action="store_true", default=TestConfig.RECORD,
                        help="Record video of gameplay")
    parser.add_argument("--save-dir", type=str, default=TestConfig.VIDEO_DIR,
                        help="Directory to save videos")
    
    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Random seed")
    
    return parser.parse_args()

def test(env, agent, episodes, render=False, record=False, save_dir='videos'):
    """
    Test agent trên một số episode
    
    Args:
        env: môi trường
        agent: agent cần test
        episodes: số episode để test
        render: có render môi trường không
        record: có ghi lại frames không
        save_dir: thư mục lưu video
        
    Returns:
        rewards: list các phần thưởng
    """
    rewards = []
    all_frames = []
    
    for i in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_frames = []
        
        print(f"Episode {i+1}/{episodes}")
        
        # Chơi một episode
        while not (done or truncated):
            # Nếu record, lưu frame hiện tại (trước khi thực hiện action)
            if record and hasattr(env, 'render') and hasattr(env.env, 'render'):
                frame = env.env.render()
                if frame is not None:
                    episode_frames.append(frame)
            
            # Chọn action
            action = agent.select_action(state, training=False)
            
            # Thực hiện action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Cập nhật metrics
            total_reward += reward
            steps += 1
            
            # Render nếu cần
            if render:
                env.render()
                time.sleep(0.02)  # Delay để dễ nhìn
            
            # Chuyển sang state mới
            state = next_state
        
        print(f"Episode {i+1} finished with reward {total_reward:.1f} after {steps} steps")
        rewards.append(total_reward)
        
        # Nếu record, lưu episode dưới dạng GIF
        if record and episode_frames:
            # Tạo tên file
            filename = os.path.join(save_dir, f"episode_{i+1}_reward_{int(total_reward)}.gif")
            # Lưu GIF
            display_frames_as_gif(episode_frames, filename=filename)
            print(f"Saved episode recording to {filename}")
            
            # Thêm vào danh sách tất cả frames
            all_frames.extend(episode_frames)
    
    # Nếu record, lưu tất cả episodes dưới dạng một GIF
    if record and all_frames:
        filename = os.path.join(save_dir, f"all_episodes.gif")
        display_frames_as_gif(all_frames[::5], filename=filename, fps=20)  # Lấy 1 frame mỗi 5 frame để giảm kích thước
        print(f"Saved all episodes recording to {filename}")
    
    return rewards

def main():
    # Parse tham số
    args = parse_args()
    
    # Đặt seed
    set_seed(args.seed)
    
    # Tạo thư mục nếu cần record
    if args.record:
        create_dirs([args.save_dir])
    
    # Tạo môi trường
    env_name = args.env
    
    # Nếu cần render, tạo môi trường với render_mode
    render_mode = "human" if args.render else "rgb_array" if args.record else None
    
    # Tạo môi trường
    env = PacmanEnv(env_name=env_name, render_mode=render_mode, stack_frames=args.stack_frames)
    
    # Thông tin về state và action space
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"State shape: {state_shape}, Action space: {n_actions}")
    
    # Tạo agent
    agent = DoubleDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=args.device,
        epsilon_start=0.01,  # Bắt đầu với epsilon thấp vì chỉ cần exploitation
        epsilon_final=0.01,
        epsilon_decay=1  # Không cần decay vì không train
    )
    
    # Kiểm tra đường dẫn model
    if args.model_path is None:
        raise ValueError("--model-path must be specified")
        
    # Load model
    print(f"Loading model from {args.model_path}")
    agent.load(args.model_path)
    
    # Đánh giá agent
    print(f"Testing agent for {args.episodes} episodes")
    rewards = test(env, agent, args.episodes, render=args.render, record=args.record, save_dir=args.save_dir)
    
    # In thống kê
    print("\nTest Results:")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Standard deviation: {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    
    # Vẽ biểu đồ phần thưởng
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, 'b-')
    plt.axhline(y=np.mean(rewards), color='r', linestyle='-', label=f"Average: {np.mean(rewards):.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Test Rewards ({args.episodes} episodes)")
    plt.legend()
    
    # Lưu biểu đồ
    result_dir = args.save_dir if args.record else "."
    plt.savefig(os.path.join(result_dir, "test_rewards.png"))
    
    if args.render:
        plt.show()
    
    # Đóng môi trường
    env.close()

if __name__ == "__main__":
    main()