"""
Training script for Multi-Task Neural Receiver
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json

from ..models import NeuralReceiver, MultiTaskLoss
from ..data import SignalDataset


class Trainer:
    """Trainer for multi-task neural receiver"""

    def __init__(
        self,
        model: NeuralReceiver,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'experiments',
        experiment_name: str = 'neural_receiver',
        use_tensorboard: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Output directories
        self.output_dir = Path(output_dir) / experiment_name
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'detection_loss': 0.0,
            'regression_loss': 0.0,
            'classification_loss': 0.0
        }

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')

        for batch_idx, (iq_data, labels) in enumerate(progress_bar):
            # Move data to device
            iq_data = iq_data.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(iq_data)

            # Compute loss
            loss, loss_dict = self.criterion(predictions, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'det': f"{loss_dict['detection_loss']:.4f}",
                'reg': f"{loss_dict['regression_loss']:.4f}",
                'cls': f"{loss_dict['classification_loss']:.4f}"
            })

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'detection_loss': 0.0,
            'regression_loss': 0.0,
            'classification_loss': 0.0
        }

        # Additional metrics
        detection_correct = 0
        classification_correct = 0
        total_samples = 0
        regression_errors = []

        for iq_data, labels in tqdm(self.val_loader, desc='Validation'):
            # Move data to device
            iq_data = iq_data.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            predictions = self.model(iq_data)

            # Compute loss
            loss, loss_dict = self.criterion(predictions, labels)

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

            # Compute metrics
            batch_size = iq_data.size(0)
            total_samples += batch_size

            # Detection accuracy
            det_pred = (predictions['detection_prob'] > 0.5).float()
            detection_correct += (det_pred == labels['signal_present']).sum().item()

            # Classification accuracy
            cls_pred = torch.argmax(predictions['classification_logits'], dim=1)
            classification_correct += (cls_pred == labels['modulation_class']).sum().item()

            # Regression error (for signal-present samples)
            signal_mask = labels['signal_present'] > 0.5
            if signal_mask.sum() > 0:
                reg_targets = torch.stack([
                    labels['center_freq'],
                    labels['bandwidth'],
                    labels['snr_db'],
                    labels['symbol_rate']
                ], dim=1)

                reg_pred = predictions['regression_params'][signal_mask]
                reg_true = reg_targets[signal_mask]
                reg_error = torch.abs(reg_pred - reg_true).mean(dim=1)
                regression_errors.extend(reg_error.cpu().numpy().tolist())

        # Average losses and metrics
        num_batches = len(self.val_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        epoch_losses['detection_accuracy'] = detection_correct / total_samples
        epoch_losses['classification_accuracy'] = classification_correct / total_samples
        epoch_losses['regression_mae'] = np.mean(regression_errors) if regression_errors else 0.0

        return epoch_losses

    def train(self, num_epochs: int, save_every: int = 10):
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Training on device: {self.device}")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_losses['total_loss'])

            # Log to console
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_losses['total_loss']:.4f} | Val Loss: {val_losses['total_loss']:.4f}")
            print(f"Detection Acc: {val_losses['detection_accuracy']:.4f} | Classification Acc: {val_losses['classification_accuracy']:.4f}")
            print(f"Regression MAE: {val_losses['regression_mae']:.4f}")

            # Log to TensorBoard
            if self.use_tensorboard:
                for key, value in train_losses.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)

                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', current_lr, epoch)

            # Save training history
            self.training_history['train_loss'].append(train_losses['total_loss'])
            self.training_history['val_loss'].append(val_losses['total_loss'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pt', val_losses)
                print(f"Saved best model (val_loss: {self.best_val_loss:.4f})")

            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', val_losses)

        # Save final model
        self.save_checkpoint('final_model.pt', val_losses)

        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")

        if self.use_tensorboard:
            self.writer.close()

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Loaded checkpoint from epoch {self.epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def create_trainer(
    model_config: Dict,
    training_config: Dict,
    data_config: Dict
) -> Trainer:
    """
    Factory function to create trainer

    Args:
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration

    Returns:
        Configured trainer
    """
    # Create datasets
    train_dataset = SignalDataset(
        n_samples=data_config.get('train_samples', 10000),
        sequence_length=data_config.get('sequence_length', 1024),
        snr_range=data_config.get('snr_range', (-10, 0)),
        no_signal_prob=data_config.get('no_signal_prob', 0.2),
        seed=data_config.get('seed', 42),
        pregenerate=data_config.get('pregenerate', False)
    )

    val_dataset = SignalDataset(
        n_samples=data_config.get('val_samples', 2000),
        sequence_length=data_config.get('sequence_length', 1024),
        snr_range=data_config.get('snr_range', (-10, 0)),
        no_signal_prob=data_config.get('no_signal_prob', 0.2),
        seed=data_config.get('seed', 42) + 1000,
        pregenerate=data_config.get('pregenerate', False)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 32),
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        collate_fn=SignalDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 32),
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        collate_fn=SignalDataset.collate_fn
    )

    # Create model
    model = NeuralReceiver(
        input_channels=model_config.get('input_channels', 2),
        base_channels=model_config.get('base_channels', 64),
        num_blocks=model_config.get('num_blocks', 4),
        num_classes=model_config.get('num_classes', 12),
        dropout=model_config.get('dropout', 0.2)
    )

    # Create loss function
    criterion = MultiTaskLoss(
        detection_weight=training_config.get('detection_weight', 1.0),
        regression_weight=training_config.get('regression_weight', 1.0),
        classification_weight=training_config.get('classification_weight', 1.0),
        use_uncertainty_weighting=training_config.get('use_uncertainty_weighting', False)
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.get('num_epochs', 100),
        eta_min=training_config.get('min_lr', 1e-6)
    ) if training_config.get('use_scheduler', True) else None

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        output_dir=training_config.get('output_dir', 'experiments'),
        experiment_name=training_config.get('experiment_name', 'neural_receiver'),
        use_tensorboard=training_config.get('use_tensorboard', True)
    )

    return trainer
