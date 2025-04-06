import torch

config = {
    # Environment
    'env_name': 'ALE/Pacman-v5',
    'frame_skip': 4,
    'render_mode': None,  # Set to 'human' for visualization
    
    # Training parameters
    'num_episodes': 2000,
    'max_steps_per_episode': 100000,
    'num_eval_episodes': 5,
    'eval_interval': 50,  # Evaluate every 50 episodes
    
    # Agent hyperparameters
    'gamma': 0.99,  # Discount factor
    'learning_rate': 0.0001,
    'batch_size': 32,
    'buffer_size': 50000,
    'target_update': 1000,  # Update target network every 1000 steps
    
    # Exploration parameters
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    
    # Preprocessing
    'frame_stack': 4,  # Stack 4 frames as input
    'image_size': (84, 84),  # Resize frames to 84x84
    
    # Hardware
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # Saving & Loading
    'save_dir': './checkpoints',
    'log_dir': './logs',
    'save_interval': 100,  # Save model every 100 episodes
    
    # Metrics to track
    'metrics': ['episode_reward', 'episode_length', 'epsilon', 'loss', 'eval_reward']
}