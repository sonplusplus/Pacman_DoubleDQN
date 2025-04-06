# config.py
"""
Cấu hình cho dự án Pacman Double DQN
"""

class Config:
    # Cấu hình môi trường
    ENV_NAME = "ALE/MsPacman-v5"  # Thay thế MsPacmanNoFrameskip-v4
    STACK_FRAMES = 4                     # Số frame để stack
    FRAME_SIZE = (84, 84)                # Kích thước frame sau khi resize
    
    # Cấu hình agent
    LEARNING_RATE = 1e-4                 # Tốc độ học
    GAMMA = 0.99                         # Hệ số discount
    EPSILON_START = 1.0                  # Epsilon ban đầu
    EPSILON_FINAL = 0.01                 # Epsilon cuối cùng
    EPSILON_DECAY = 100000               # Tốc độ giảm epsilon
    
    # Cấu hình replay buffer
    BUFFER_SIZE = 100000                 # Kích thước replay buffer
    BATCH_SIZE = 32                      # Kích thước batch
    
    # Cấu hình huấn luyện
    TARGET_UPDATE = 1000                 # Số bước giữa mỗi lần cập nhật target network
    TRAIN_STEPS = 1000000                # Tổng số bước huấn luyện
    EVAL_INTERVAL = 10000                # Khoảng cách giữa các lần đánh giá
    EVAL_EPISODES = 5                    # Số episode để đánh giá
    
    # Cấu hình lưu trữ
    SAVE_DIR = "checkpoints"             # Thư mục lưu model
    LOG_DIR = "logs"                     # Thư mục log
    SEED = 42                            # Seed cho random

class TestConfig:
    # Cấu hình kiểm thử
    MODEL_PATH = None                    # Đường dẫn đến model đã huấn luyện
    TEST_EPISODES = 10                   # Số episode để kiểm thử
    RENDER = False                       # Có render môi trường không
    RECORD = False                       # Có ghi lại video không
    VIDEO_DIR = "videos"                 # Thư mục lưu video