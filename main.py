# main.py
import argparse
import os
import sys
import torch

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Pacman Double DQN")
    
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                        help="Mode to run: 'train' or 'test'")
    
    # Arguments to pass to the selected mode
    parser.add_argument("--args", nargs=argparse.REMAINDER, 
                        help="Arguments to pass to the selected mode")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    print(f"Running in {args.mode} mode")
    
    if args.mode == "train":
        # Import và chạy script train
        from train import main as train_main
        
        # Khởi tạo lại sys.argv cho script train
        sys.argv = [sys.argv[0]] + (args.args if args.args else [])
        
        # Chạy script train
        train_main()
    elif args.mode == "test":
        # Import và chạy script test
        from test import main as test_main
        
        # Khởi tạo lại sys.argv cho script test
        sys.argv = [sys.argv[0]] + (args.args if args.args else [])
        
        # Chạy script test
        test_main()
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()