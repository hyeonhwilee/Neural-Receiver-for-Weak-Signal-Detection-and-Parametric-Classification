#!/usr/bin/env python3
"""
Inference script for Neural Receiver
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.models import NeuralReceiver
from src.data import ModulationType


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with Neural Receiver')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input IQ data (.npy file with complex values)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Detection threshold'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (.npy file)'
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_iq_data(file_path: str) -> np.ndarray:
    """Load IQ data from file"""
    data = np.load(file_path)

    # Ensure complex dtype
    if not np.iscomplexobj(data):
        raise ValueError("Input data must be complex (IQ samples)")

    return data


def preprocess_iq_data(iq_data: np.ndarray, sequence_length: int) -> torch.Tensor:
    """
    Preprocess IQ data for model input

    Args:
        iq_data: Complex IQ data (n_samples,) or (batch_size, n_samples)
        sequence_length: Expected sequence length

    Returns:
        Preprocessed tensor (batch_size, 2, sequence_length)
    """
    # Handle single sequence
    if iq_data.ndim == 1:
        iq_data = iq_data.reshape(1, -1)

    batch_size, n_samples = iq_data.shape

    # Trim or pad to sequence_length
    if n_samples > sequence_length:
        iq_data = iq_data[:, :sequence_length]
    elif n_samples < sequence_length:
        # Zero-pad
        padding = np.zeros((batch_size, sequence_length - n_samples), dtype=iq_data.dtype)
        iq_data = np.concatenate([iq_data, padding], axis=1)

    # Convert to tensor [batch_size, 2, sequence_length]
    iq_tensor = torch.zeros(batch_size, 2, sequence_length)
    iq_tensor[:, 0, :] = torch.from_numpy(iq_data.real).float()
    iq_tensor[:, 1, :] = torch.from_numpy(iq_data.imag).float()

    return iq_tensor


def main():
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load IQ data
    print(f"Loading IQ data from {args.input}")
    iq_data = load_iq_data(args.input)
    print(f"Loaded {len(iq_data)} IQ samples")

    # Preprocess
    sequence_length = config['data']['sequence_length']
    iq_tensor = preprocess_iq_data(iq_data, sequence_length)
    print(f"Preprocessed to shape: {iq_tensor.shape}")

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
    model.to(args.device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Run inference
    print("\nRunning inference...")
    iq_tensor = iq_tensor.to(args.device)

    with torch.no_grad():
        predictions = model.predict(iq_tensor, threshold=args.threshold)

    # Get class names
    class_names = ModulationType.get_class_names()

    # Print results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)

    for i in range(len(iq_tensor)):
        print(f"\nSample {i}:")
        print(f"  Signal Detected:    {bool(predictions['signal_detected'][i])}")
        print(f"  Detection Prob:     {predictions['detection_prob'][i]:.4f}")

        if predictions['signal_detected'][i]:
            mod_class = predictions['modulation_class'][i].item()
            mod_prob = predictions['modulation_prob'][i, mod_class].item()

            print(f"  Modulation Type:    {class_names[mod_class]} ({mod_prob:.4f})")
            print(f"  Center Frequency:   {predictions['center_freq'][i]:.4f}")
            print(f"  Bandwidth:          {predictions['bandwidth'][i]:.4f}")
            print(f"  Estimated SNR:      {predictions['snr_db'][i]:.2f} dB")
            print(f"  Symbol Rate:        {predictions['symbol_rate'][i]:.2f}")

    print("="*60 + "\n")

    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert predictions to numpy
        predictions_np = {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in predictions.items()
        }

        np.save(output_path, predictions_np)
        print(f"Predictions saved to {output_path}")


if __name__ == '__main__':
    main()
