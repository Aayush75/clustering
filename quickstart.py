"""
Quick start script for TEMI clustering.

This script provides a simplified interface to run the clustering experiment
with common presets.
"""

import subprocess
import sys
import argparse


def run_command(cmd):
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run as a list
    """
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed successfully!")


def run_training(resume=False, force_recompute=False):
    """
    Run the training script.
    
    Args:
        resume: Whether to resume from checkpoint
        force_recompute: Whether to force recompute embeddings
    """
    cmd = [sys.executable, "train.py"]
    
    if resume:
        cmd.extend(["--resume", "checkpoints/checkpoint_latest.pth"])
    
    if force_recompute:
        cmd.append("--force-recompute")
    
    run_command(cmd)


def visualize_results():
    """Visualize training results."""
    run_command([sys.executable, "visualize_results.py"])


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quick start script for TEMI clustering"
    )
    
    parser.add_argument(
        "action",
        choices=["install", "train", "resume", "visualize", "all"],
        help="Action to perform"
    )
    
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recompute embeddings"
    )
    
    args = parser.parse_args()
    
    if args.action == "install":
        install_dependencies()
    
    elif args.action == "train":
        run_training(resume=False, force_recompute=args.force_recompute)
    
    elif args.action == "resume":
        run_training(resume=True, force_recompute=args.force_recompute)
    
    elif args.action == "visualize":
        visualize_results()
    
    elif args.action == "all":
        install_dependencies()
        run_training(resume=False, force_recompute=args.force_recompute)
        visualize_results()


if __name__ == "__main__":
    main()
