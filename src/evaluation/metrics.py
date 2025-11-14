"""
Evaluation metrics for multi-task neural receiver
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, accuracy_score
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate comprehensive metrics for multi-task evaluation"""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.detection_probs = []
        self.detection_labels = []

        self.regression_preds = []
        self.regression_labels = []

        self.classification_probs = []
        self.classification_labels = []

    def update(
        self,
        detection_prob: np.ndarray,
        detection_label: np.ndarray,
        regression_pred: np.ndarray,
        regression_label: np.ndarray,
        classification_prob: np.ndarray,
        classification_label: np.ndarray
    ):
        """
        Update metrics with batch results

        Args:
            detection_prob: Detection probabilities (batch_size,)
            detection_label: Ground truth detection labels (batch_size,)
            regression_pred: Regression predictions (batch_size, 4)
            regression_label: Ground truth regression labels (batch_size, 4)
            classification_prob: Classification probabilities (batch_size, num_classes)
            classification_label: Ground truth classification labels (batch_size,)
        """
        self.detection_probs.extend(detection_prob.tolist())
        self.detection_labels.extend(detection_label.tolist())

        self.regression_preds.append(regression_pred)
        self.regression_labels.append(regression_label)

        self.classification_probs.append(classification_prob)
        self.classification_labels.extend(classification_label.tolist())

    def compute_detection_metrics(self) -> Dict:
        """Compute detection metrics (ROC, AUC, etc.)"""
        y_true = np.array(self.detection_labels)
        y_prob = np.array(self.detection_probs)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        # Best threshold (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        best_tpr = tpr[best_idx]
        best_fpr = fpr[best_idx]

        # Accuracy at best threshold
        y_pred = (y_prob >= best_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)

        return {
            'auc': auc,
            'avg_precision': avg_precision,
            'best_threshold': best_threshold,
            'best_tpr': best_tpr,
            'best_fpr': best_fpr,
            'accuracy': accuracy,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': thresholds,
            'precision': precision,
            'recall': recall,
            'pr_thresholds': pr_thresholds
        }

    def compute_regression_metrics(self) -> Dict:
        """Compute regression metrics (MSE, MAE, etc.)"""
        preds = np.concatenate(self.regression_preds, axis=0)
        labels = np.concatenate(self.regression_labels, axis=0)

        # Filter only signal-present samples
        # Assume label[:,3] (SNR) > -50 indicates signal present
        signal_mask = labels[:, 2] > -50
        preds_signal = preds[signal_mask]
        labels_signal = labels[signal_mask]

        if len(preds_signal) == 0:
            return {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'param_errors': {}
            }

        # Overall metrics
        mse = np.mean((preds_signal - labels_signal) ** 2)
        mae = np.mean(np.abs(preds_signal - labels_signal))
        rmse = np.sqrt(mse)

        # Per-parameter metrics
        param_names = ['center_freq', 'bandwidth', 'snr_db', 'symbol_rate']
        param_errors = {}

        for i, param_name in enumerate(param_names):
            pred = preds_signal[:, i]
            true = labels_signal[:, i]

            param_errors[param_name] = {
                'mae': np.mean(np.abs(pred - true)),
                'mse': np.mean((pred - true) ** 2),
                'rmse': np.sqrt(np.mean((pred - true) ** 2)),
                'mean_error': np.mean(pred - true),
                'std_error': np.std(pred - true)
            }

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'param_errors': param_errors
        }

    def compute_classification_metrics(self) -> Dict:
        """Compute classification metrics (accuracy, F1, confusion matrix)"""
        y_true = np.array(self.classification_labels)
        y_prob = np.concatenate(self.classification_probs, axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }

    def compute_all_metrics(self) -> Dict:
        """Compute all metrics"""
        metrics = {
            'detection': self.compute_detection_metrics(),
            'regression': self.compute_regression_metrics(),
            'classification': self.compute_classification_metrics()
        }

        return metrics

    def print_summary(self, metrics: Dict):
        """Print metrics summary"""
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)

        # Detection metrics
        det = metrics['detection']
        print("\nSignal Detection:")
        print(f"  AUC-ROC:           {det['auc']:.4f}")
        print(f"  Average Precision: {det['avg_precision']:.4f}")
        print(f"  Accuracy:          {det['accuracy']:.4f}")
        print(f"  Best Threshold:    {det['best_threshold']:.4f}")
        print(f"  TPR @ Best:        {det['best_tpr']:.4f}")
        print(f"  FPR @ Best:        {det['best_fpr']:.4f}")

        # Regression metrics
        reg = metrics['regression']
        print("\nParameter Estimation:")
        print(f"  Overall MAE:       {reg['mae']:.4f}")
        print(f"  Overall RMSE:      {reg['rmse']:.4f}")

        if reg['param_errors']:
            print("\n  Per-Parameter Errors:")
            for param_name, errors in reg['param_errors'].items():
                print(f"    {param_name:15s}: MAE={errors['mae']:.4f}, RMSE={errors['rmse']:.4f}")

        # Classification metrics
        cls = metrics['classification']
        print("\nModulation Classification:")
        print(f"  Accuracy:          {cls['accuracy']:.4f}")
        print(f"  F1 (macro):        {cls['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {cls['f1_weighted']:.4f}")

        print("\n  Per-Class Metrics:")
        for class_name in self.class_names:
            if class_name in cls['classification_report']:
                report = cls['classification_report'][class_name]
                print(f"    {class_name:25s}: Precision={report['precision']:.3f}, "
                      f"Recall={report['recall']:.3f}, F1={report['f1-score']:.3f}")

        print("="*60 + "\n")


def plot_detection_metrics(metrics: Dict, save_path: str = None):
    """Plot ROC curve and Precision-Recall curve"""
    det = metrics['detection']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(det['fpr'], det['tpr'], 'b-', linewidth=2, label=f"AUC = {det['auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    axes[0].plot(det['best_fpr'], det['best_tpr'], 'go', markersize=10,
                 label=f"Best (thresh={det['best_threshold']:.3f})")
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve - Signal Detection', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall curve
    axes[1].plot(det['recall'], det['precision'], 'b-', linewidth=2,
                 label=f"AP = {det['avg_precision']:.4f}")
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(metrics: Dict, class_names: List[str], save_path: str = None):
    """Plot confusion matrix"""
    conf_matrix = metrics['classification']['confusion_matrix']

    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        conf_matrix_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )

    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Confusion Matrix - Modulation Classification', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_regression_errors(metrics: Dict, save_path: str = None):
    """Plot regression error distributions"""
    reg = metrics['regression']
    param_errors = reg['param_errors']

    if not param_errors:
        print("No regression errors to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    param_names = ['center_freq', 'bandwidth', 'snr_db', 'symbol_rate']
    display_names = ['Center Frequency', 'Bandwidth', 'SNR (dB)', 'Symbol Rate']

    for i, (param_name, display_name) in enumerate(zip(param_names, display_names)):
        if param_name in param_errors:
            errors = param_errors[param_name]

            # Create bar plot of error metrics
            metrics_names = ['MAE', 'RMSE']
            metrics_values = [errors['mae'], errors['rmse']]

            axes[i].bar(metrics_names, metrics_values, color=['#3498db', '#e74c3c'])
            axes[i].set_ylabel('Error', fontsize=11)
            axes[i].set_title(f'{display_name}\nMAE={errors["mae"]:.4f}, RMSE={errors["rmse"]:.4f}',
                            fontsize=11)
            axes[i].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_all_metrics(metrics: Dict, class_names: List[str], output_dir: str):
    """Plot all evaluation metrics"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Detection metrics
    plot_detection_metrics(metrics, save_path=os.path.join(output_dir, 'detection_metrics.png'))

    # Confusion matrix
    plot_confusion_matrix(metrics, class_names, save_path=os.path.join(output_dir, 'confusion_matrix.png'))

    # Regression errors
    plot_regression_errors(metrics, save_path=os.path.join(output_dir, 'regression_errors.png'))

    print(f"Plots saved to {output_dir}")
