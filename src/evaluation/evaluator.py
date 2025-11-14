"""
Model evaluator for neural receiver
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from ..models import NeuralReceiver
from .metrics import MetricsCalculator


class Evaluator:
    """Evaluate trained neural receiver model"""

    def __init__(
        self,
        model: NeuralReceiver,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        class_names: list = None
    ):
        self.model = model.to(device)
        self.device = device

        if class_names is None:
            from ..data import ModulationType
            class_names = ModulationType.get_class_names()

        self.class_names = class_names
        self.metrics_calculator = MetricsCalculator(class_names)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model on dataset

        Args:
            dataloader: DataLoader for evaluation dataset

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        self.metrics_calculator.reset()

        print("Evaluating model...")

        for iq_data, labels in tqdm(dataloader, desc='Evaluation'):
            # Move data to device
            iq_data = iq_data.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            outputs = self.model(iq_data)

            # Get predictions
            detection_prob = outputs['detection_prob'].cpu().numpy()
            regression_pred = outputs['regression_params'].cpu().numpy()

            classification_logits = outputs['classification_logits']
            classification_prob = F.softmax(classification_logits, dim=1).cpu().numpy()

            # Get labels
            detection_label = labels['signal_present'].cpu().numpy()

            regression_label = torch.stack([
                labels['center_freq'],
                labels['bandwidth'],
                labels['snr_db'],
                labels['symbol_rate']
            ], dim=1).cpu().numpy()

            classification_label = labels['modulation_class'].cpu().numpy()

            # Update metrics
            self.metrics_calculator.update(
                detection_prob=detection_prob,
                detection_label=detection_label,
                regression_pred=regression_pred,
                regression_label=regression_label,
                classification_prob=classification_prob,
                classification_label=classification_label
            )

        # Compute all metrics
        metrics = self.metrics_calculator.compute_all_metrics()

        return metrics

    def save_results(self, metrics: Dict, output_dir: str):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Print summary
        self.metrics_calculator.print_summary(metrics)

        # Save plots
        from .metrics import plot_all_metrics
        plot_all_metrics(metrics, self.class_names, str(output_path))

        # Save metrics as JSON
        import json

        # Convert numpy arrays to lists for JSON serialization
        metrics_json = self._prepare_metrics_for_json(metrics)

        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=2)

        print(f"Results saved to {output_dir}")

    def _prepare_metrics_for_json(self, metrics: Dict) -> Dict:
        """Convert numpy arrays to lists for JSON serialization"""
        json_metrics = {}

        # Detection metrics
        json_metrics['detection'] = {
            'auc': float(metrics['detection']['auc']),
            'avg_precision': float(metrics['detection']['avg_precision']),
            'accuracy': float(metrics['detection']['accuracy']),
            'best_threshold': float(metrics['detection']['best_threshold']),
            'best_tpr': float(metrics['detection']['best_tpr']),
            'best_fpr': float(metrics['detection']['best_fpr'])
        }

        # Regression metrics
        json_metrics['regression'] = {
            'mse': float(metrics['regression']['mse']),
            'mae': float(metrics['regression']['mae']),
            'rmse': float(metrics['regression']['rmse']),
            'param_errors': {}
        }

        for param_name, errors in metrics['regression']['param_errors'].items():
            json_metrics['regression']['param_errors'][param_name] = {
                k: float(v) for k, v in errors.items()
            }

        # Classification metrics
        json_metrics['classification'] = {
            'accuracy': float(metrics['classification']['accuracy']),
            'f1_macro': float(metrics['classification']['f1_macro']),
            'f1_weighted': float(metrics['classification']['f1_weighted']),
            'classification_report': metrics['classification']['classification_report']
        }

        return json_metrics


def load_and_evaluate(
    checkpoint_path: str,
    dataloader: DataLoader,
    output_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Load model from checkpoint and evaluate

    Args:
        checkpoint_path: Path to model checkpoint
        dataloader: DataLoader for evaluation
        output_dir: Directory to save results
        device: Device to use for evaluation
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (assumes default config - modify as needed)
    model = NeuralReceiver()
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Evaluate
    metrics = evaluator.evaluate(dataloader)

    # Save results
    evaluator.save_results(metrics, output_dir)

    return metrics
