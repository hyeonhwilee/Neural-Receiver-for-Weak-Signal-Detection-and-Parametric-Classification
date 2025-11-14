#!/usr/bin/env python3
"""
Evaluation script for Neural Receiver
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.models import NeuralReceiver
from src.data import SignalDataset
from src.evaluation import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Neural Receiver')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation'
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

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = SignalDataset(
        n_samples=config['data'].get('test_samples', 2000),
        sequence_length=config['data']['sequence_length'],
        snr_range=config['data']['snr_range'],
        no_signal_prob=config['data']['no_signal_prob'],
        seed=config['data']['seed'] + 2000,  # Different seed for test set
        pregenerate=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=SignalDataset.collate_fn
    )

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = NeuralReceiver(
        input_channels=config['model']['input_channels'],
        base_channels=config['model']['base_channels'],
        num_blocks=config['model']['num_blocks'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create evaluator
    evaluator = Evaluator(model, device=args.device)

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluator.evaluate(test_loader)

    # Save results
    output_dir = Path(args.output)
    evaluator.save_results(metrics, str(output_dir))

    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
