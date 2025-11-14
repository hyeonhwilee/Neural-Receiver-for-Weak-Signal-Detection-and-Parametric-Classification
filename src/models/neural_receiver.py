"""
Multi-Task Neural Receiver Architecture
Shared backbone with task-specific heads for detection, estimation, and classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class SharedBackbone(nn.Module):
    """
    Shared feature extractor for IQ data
    Uses 1D CNNs to extract features from time-series IQ sequences
    """

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []

        # Initial convolution
        layers.append(nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3))
        layers.append(nn.BatchNorm1d(base_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Residual blocks with downsampling
        current_channels = base_channels
        for i in range(num_blocks):
            # Residual block
            layers.append(ResidualBlock(current_channels))

            # Downsample (except last block)
            if i < num_blocks - 1:
                next_channels = current_channels * 2
                layers.append(nn.Conv1d(current_channels, next_channels, kernel_size=3, stride=2, padding=1))
                layers.append(nn.BatchNorm1d(next_channels))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_channels = next_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = current_channels

    def forward(self, x):
        """
        Args:
            x: Input IQ data (batch_size, 2, sequence_length)
        Returns:
            features: Extracted features (batch_size, feature_dim, reduced_length)
        """
        return self.feature_extractor(x)


class DetectionHead(nn.Module):
    """Binary classification head for signal detection"""

    def __init__(self, feature_dim: int, dropout: float = 0.3):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: (batch_size, feature_dim, length)
        Returns:
            detection_prob: (batch_size,) - probability of signal present
        """
        pooled = self.pooling(features).squeeze(-1)  # (batch_size, feature_dim)
        return self.fc(pooled).squeeze(-1)  # (batch_size,)


class RegressionHead(nn.Module):
    """Regression head for parameter estimation"""

    def __init__(self, feature_dim: int, num_params: int = 4, dropout: float = 0.3):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_params)
        )

    def forward(self, features):
        """
        Args:
            features: (batch_size, feature_dim, length)
        Returns:
            params: (batch_size, num_params) - [center_freq, bandwidth, snr_db, symbol_rate]
        """
        pooled = self.pooling(features).squeeze(-1)
        return self.fc(pooled)


class ClassificationHead(nn.Module):
    """Multi-class classification head for modulation type"""

    def __init__(self, feature_dim: int, num_classes: int = 12, dropout: float = 0.3):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        """
        Args:
            features: (batch_size, feature_dim, length)
        Returns:
            logits: (batch_size, num_classes)
        """
        pooled = self.pooling(features).squeeze(-1)
        return self.fc(pooled)


class NeuralReceiver(nn.Module):
    """
    Multi-Task Neural Receiver

    Architecture:
    - Shared backbone: Extracts features from IQ data
    - Detection head: Binary classification (signal present/absent)
    - Regression head: Parameter estimation (fc, bandwidth, SNR, symbol rate)
    - Classification head: Modulation type classification
    """

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 4,
        num_classes: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()

        # Shared feature extractor
        self.backbone = SharedBackbone(
            input_channels=input_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            dropout=dropout
        )

        feature_dim = self.backbone.feature_dim

        # Task-specific heads
        self.detection_head = DetectionHead(feature_dim, dropout=dropout)
        self.regression_head = RegressionHead(feature_dim, num_params=4, dropout=dropout)
        self.classification_head = ClassificationHead(feature_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input IQ data (batch_size, 2, sequence_length)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - detection_prob: Signal detection probability (batch_size,)
            - regression_params: Estimated parameters (batch_size, 4)
                [center_freq, bandwidth, snr_db, symbol_rate]
            - classification_logits: Modulation classification logits (batch_size, num_classes)
            - features: (optional) Intermediate features
        """
        # Extract features
        features = self.backbone(x)

        # Task-specific predictions
        detection_prob = self.detection_head(features)
        regression_params = self.regression_head(features)
        classification_logits = self.classification_head(features)

        outputs = {
            'detection_prob': detection_prob,
            'regression_params': regression_params,
            'classification_logits': classification_logits
        }

        if return_features:
            outputs['features'] = features

        return outputs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions with post-processing

        Args:
            x: Input IQ data (batch_size, 2, sequence_length)
            threshold: Detection threshold

        Returns:
            Dictionary containing:
            - signal_detected: Binary detection (batch_size,)
            - center_freq: Estimated center frequency (batch_size,)
            - bandwidth: Estimated bandwidth (batch_size,)
            - snr_db: Estimated SNR (batch_size,)
            - symbol_rate: Estimated symbol rate (batch_size,)
            - modulation_class: Predicted modulation class (batch_size,)
            - modulation_prob: Class probabilities (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

            # Binary detection
            signal_detected = (outputs['detection_prob'] > threshold).long()

            # Regression parameters
            center_freq = outputs['regression_params'][:, 0]
            bandwidth = outputs['regression_params'][:, 1]
            snr_db = outputs['regression_params'][:, 2]
            symbol_rate = outputs['regression_params'][:, 3]

            # Classification
            modulation_prob = F.softmax(outputs['classification_logits'], dim=1)
            modulation_class = torch.argmax(modulation_prob, dim=1)

            return {
                'signal_detected': signal_detected,
                'detection_prob': outputs['detection_prob'],
                'center_freq': center_freq,
                'bandwidth': bandwidth,
                'snr_db': snr_db,
                'symbol_rate': symbol_rate,
                'modulation_class': modulation_class,
                'modulation_prob': modulation_prob
            }


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    Balances detection, regression, and classification objectives
    """

    def __init__(
        self,
        detection_weight: float = 1.0,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        use_uncertainty_weighting: bool = False
    ):
        """
        Args:
            detection_weight: Weight for detection loss
            regression_weight: Weight for regression loss
            classification_weight: Weight for classification loss
            use_uncertainty_weighting: Use learned uncertainty weighting (Kendall et al. 2018)
        """
        super().__init__()

        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Learnable log-variance parameters for uncertainty weighting
            self.log_var_detection = nn.Parameter(torch.zeros(1))
            self.log_var_regression = nn.Parameter(torch.zeros(1))
            self.log_var_classification = nn.Parameter(torch.zeros(1))
        else:
            self.detection_weight = detection_weight
            self.regression_weight = regression_weight
            self.classification_weight = classification_weight

        # Individual loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss

        Args:
            predictions: Model predictions
            labels: Ground truth labels

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Detection loss (binary cross-entropy)
        detection_loss = self.bce_loss(
            predictions['detection_prob'],
            labels['signal_present']
        )

        # Regression loss (only for samples with signal present)
        signal_mask = labels['signal_present'] > 0.5

        if signal_mask.sum() > 0:
            # Extract regression targets
            reg_targets = torch.stack([
                labels['center_freq'],
                labels['bandwidth'],
                labels['snr_db'],
                labels['symbol_rate']
            ], dim=1)  # (batch_size, 4)

            # Only compute loss for signal-present samples
            reg_pred = predictions['regression_params'][signal_mask]
            reg_true = reg_targets[signal_mask]

            regression_loss = self.mse_loss(reg_pred, reg_true)
        else:
            regression_loss = torch.tensor(0.0, device=predictions['detection_prob'].device)

        # Classification loss
        classification_loss = self.ce_loss(
            predictions['classification_logits'],
            labels['modulation_class']
        )

        # Combine losses
        if self.use_uncertainty_weighting:
            # Uncertainty weighting (Kendall et al. 2018)
            total_loss = (
                torch.exp(-self.log_var_detection) * detection_loss + self.log_var_detection +
                torch.exp(-self.log_var_regression) * regression_loss + self.log_var_regression +
                torch.exp(-self.log_var_classification) * classification_loss + self.log_var_classification
            )
        else:
            total_loss = (
                self.detection_weight * detection_loss +
                self.regression_weight * regression_loss +
                self.classification_weight * classification_loss
            )

        loss_dict = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'regression_loss': regression_loss.item() if isinstance(regression_loss, torch.Tensor) else 0.0,
            'classification_loss': classification_loss.item()
        }

        return total_loss, loss_dict
