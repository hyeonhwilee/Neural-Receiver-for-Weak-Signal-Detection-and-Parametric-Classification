#!/usr/bin/env python3
"""
Main training script for Neural Receiver
"""

import argparse
import yaml
from pathlib import Path

from src.training import create_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural Receiver')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model_config=config['model'],
        training_config=config['training'],
        data_config=config['data']
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_every=10
    )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
