# train.py
import argparse
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque

from env import PacmanEnv
from agent import DoubleDQNAgent
from utils import set_seed, create_dirs, get_timestamp, plot_rewards
from config import Config

def parse_args():
    """Parse các tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description="Train Double DQN for Pacman")
    
    # Môi trường
    parser.add_argument("--env", type=str, default=Config.ENV_NAME, 
                        help="Atari environment name")
    parser.add_argument("--stack-frames", type=int, default=Config.STACK_FRAMES,
                        help="Number of frames to stack")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, 
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=Config.GAMMA, 
                        help="Discount factor")
    parser.add_argument("--eps-start", type=float, default=Config.EPSILON_START,
                        help="Starting value of epsilon")
    parser.add_argument("--eps-final", type=float, default=Config.EPSILON_FINAL,
                        help="Final value of epsilon")
    parser.add_argument("--eps-decay", type=int, default=Config.EPSILON_DECAY,
                        help="Number of steps for epsilon decay")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--buffer-size", type=int, default=Config.BUFFER_SIZE,
                        help="Size of replay buffer")
    parser.add_argument("--target-update", type=int, default=Config.TARGET_UPDATE,
                        help="Number of steps between target network updates")
    parser.add_argument("--train-steps", type=int, default=Config.TRAIN_STEPS,
                        help="Total number of training steps")
    parser.add_argument("--eval-interval", type=int, default=Config.EVAL_INTERVAL,
                        help="Interval between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=Config.EVAL_EPISODES,
                        help="Number of episodes for evaluation")
    
    # Misc
    parser.add_argument("--save-dir", type=str, default=Config.SAVE_DIR,
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default=Config.LOG_DIR,
                        help="Directory for tensorboard logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Random seed")
    
    return parser.parse_args()

def evaluate(env, agent, num_episodes):
    """
    Đánh giá agent trên một số episodes
    
    Args:
        env: môi trường
        agent: agent cần đánh giá
        num_episodes: số episodes để đánh giá
        
    Returns:
        float: phần thưởng trung bình
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        # Chơi một episode
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            
            total_reward += reward
            state = next_state
        
        total_rewards.append(total_reward)
    
    # Tính phần thưởng trung bình
    average_reward = np.mean(total_rewards)
    return average_reward

def main():
    # Parse tham số
    args = parse_args()
    
    # Đặt seed
    set_seed(args.seed)
    
    # Tạo thư mục
    timestamp = get_timestamp()
    save_dir = os.path.join(args.save_dir, timestamp)
    log_dir = os.path.join(args.log_dir, timestamp)
    create_dirs([save_dir, log_dir])
    
    # Thiết lập Tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Tạo môi trường
    env = PacmanEnv(env_name=args.env, stack_frames=args.stack_frames)
    eval_env = PacmanEnv(env_name=args.env, stack_frames=args.stack_frames)
    
    # Thông tin về state và action space
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"State shape: {state_shape}, Action space: {n_actions}")
    
    # Tạo agent
    agent = DoubleDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=args.device,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_final=args.eps_final,
        epsilon_decay=args.eps_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )
    
    # Lưu thông tin cấu hình
    config_file = os.path.join(log_dir, "config.txt")
    with open(config_file, "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_rewards = []
    avg_rewards = []
    rewards_window = deque(maxlen=100)  # Để tính phần thưởng trung bình của 100 episode gần nhất
    
    episode = 0
    best_eval_reward = float('-inf')
    
    progress_bar = tqdm(range(1, args.train_steps + 1), desc="Training")
    
    for step in progress_bar:
        # Chọn và thực hiện action
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Lưu transition vào replay buffer
        agent.buffer.add(state, action, reward, next_state, done or truncated)
        
        # Cập nhật phần thưởng của episode
        episode_reward += reward
        
        # Chuyển sang state mới
        state = next_state
        
        # Huấn luyện agent
        loss = agent.optimize()
        
        # Cập nhật target network
        if step % args.target_update == 0:
            agent.update_target_network()
        
        # Log metrics
        if loss is not None:
            writer.add_scalar("train/loss", loss, step)
        
        writer.add_scalar("train/epsilon", agent.epsilon, step)
        
        # Episode kết thúc
        if done or truncated:
            # Log kết thúc episode
            writer.add_scalar("train/episode_reward", episode_reward, episode)
            episode_rewards.append(episode_reward)
            rewards_window.append(episode_reward)
            
            if len(rewards_window) > 0:
                avg_reward = np.mean(rewards_window)
                avg_rewards.append(avg_reward)
                writer.add_scalar("train/avg_reward_100", avg_reward, episode)
                
                # Cập nhật progress bar
                progress_bar.set_postfix({
                    "episode": episode,
                    "reward": f"{episode_reward:.1f}",
                    "avg_100": f"{avg_reward:.1f}"
                })
            
            # Reset môi trường
            state, _ = env.reset()
            episode_reward = 0
            episode += 1
        
        # Đánh giá định kỳ
        if step % args.eval_interval == 0:
            eval_reward = evaluate(eval_env, agent, args.eval_episodes)
            writer.add_scalar("eval/avg_reward", eval_reward, step)
            
            print(f"\nEvaluation at step {step}: Average reward = {eval_reward:.2f}")
            
            # Lưu model tốt nhất
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(save_dir, "model_best.pt"))
                print(f"New best model saved with reward {best_eval_reward:.2f}")
            
            # Lưu checkpoint định kỳ
            agent.save(os.path.join(save_dir, f"model_step_{step}.pt"))
            
            # Vẽ đồ thị phần thưởng
            if len(episode_rewards) > 0:
                plot_rewards(
                    episode_rewards, 
                    avg_rewards,
                    window_size=100,
                    filename=os.path.join(log_dir, f"rewards_step_{step}.png")
                )
    
    # Lưu model cuối cùng
    agent.save(os.path.join(save_dir, "model_final.pt"))
    
    # Vẽ đồ thị phần thưởng cuối cùng
    if len(episode_rewards) > 0:
        plot_rewards(
            episode_rewards, 
            avg_rewards,
            window_size=100,
            filename=os.path.join(log_dir, "rewards_final.png")
        )
    
    # Đóng môi trường
    env.close()
    eval_env.close()
    
    # Đóng Tensorboard writer
    writer.close()
    
    print(f"\nTraining completed. Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Models saved in {save_dir}")
    print(f"Logs saved in {log_dir}")

if __name__ == "__main__":
    main()