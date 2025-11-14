"""
PyTorch Dataset for Multi-Task Signal Detection and Classification
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple, Optional

from .signal_generator import SignalGenerator, SignalParams, ModulationType


class SignalDataset(Dataset):
    """
    Dataset for multi-task learning:
    - Signal detection (binary classification)
    - Parameter estimation (regression)
    - Modulation classification (multi-class)
    """

    def __init__(
        self,
        n_samples: int,
        sequence_length: int = 1024,
        sample_rate: float = 1.0,
        snr_range: Tuple[float, float] = (-10, 0),
        no_signal_prob: float = 0.2,
        seed: Optional[int] = None,
        pregenerate: bool = False
    ):
        """
        Args:
            n_samples: Number of samples in the dataset
            sequence_length: Length of each IQ sequence
            sample_rate: Sampling rate (normalized)
            snr_range: Range of SNR values in dB
            no_signal_prob: Probability of generating no-signal samples
            seed: Random seed for reproducibility
            pregenerate: Whether to pregenerate all samples (uses more memory but faster)
        """
        self.n_samples = n_samples
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.no_signal_prob = no_signal_prob
        self.seed = seed

        # Initialize generator
        self.generator = SignalGenerator(sample_rate=sample_rate, seed=seed)

        # Pregenerate data if requested
        self.pregenerate = pregenerate
        if pregenerate:
            self.data = self._pregenerate_data()
        else:
            self.data = None

        # Set random seed for dataset sampling
        if seed is not None:
            np.random.seed(seed)

    def _pregenerate_data(self) -> list:
        """Pregenerate all samples"""
        print(f"Pregenerating {self.n_samples} samples...")
        data = []
        for idx in range(self.n_samples):
            # Temporarily set seed for reproducibility
            if self.seed is not None:
                np.random.seed(self.seed + idx)

            sample = self._generate_sample()
            data.append(sample)

        return data

    def _generate_sample(self) -> Dict:
        """Generate a single sample"""
        # Decide if this is a no-signal sample
        is_no_signal = np.random.rand() < self.no_signal_prob

        # Generate random parameters
        params = self.generator.generate_random_params(
            include_no_signal=True,
            snr_range=self.snr_range
        )

        # Force no-signal if decided
        if is_no_signal:
            params.modulation_type = 'No-signal'
            params.power = 0.0
            params.snr_db = -np.inf

        # Generate signal
        signal, noisy_signal = self.generator.generate_signal(
            self.sequence_length,
            params,
            add_noise=True
        )

        # Create labels for multi-task learning
        labels = self._create_labels(params)

        return {
            'iq_data': noisy_signal,
            'labels': labels,
            'params': params
        }

    def _create_labels(self, params: SignalParams) -> Dict:
        """Create multi-task labels"""
        # Signal detection (binary)
        signal_present = 1.0 if params.modulation_type != 'No-signal' else 0.0

        # Parameter estimation (only valid when signal present)
        # For no-signal, we set parameters to sentinel values
        if params.modulation_type == 'No-signal':
            center_freq = 0.0
            bandwidth = 0.0
            power = 0.0
            snr_db = -100.0  # Very low value
            symbol_rate = 0.0
        else:
            center_freq = params.center_freq
            bandwidth = params.bandwidth
            power = params.power
            snr_db = params.snr_db
            symbol_rate = params.symbol_rate

        # Modulation classification
        modulation_class = ModulationType.name_to_idx(params.modulation_type)

        return {
            'signal_present': signal_present,
            'center_freq': center_freq,
            'bandwidth': bandwidth,
            'power': power,
            'snr_db': snr_db,
            'symbol_rate': symbol_rate,
            'modulation_class': modulation_class
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample

        Returns:
            iq_data: Complex IQ data (2, sequence_length) - [real, imag]
            labels: Dictionary of labels for multi-task learning
        """
        if self.pregenerate:
            sample = self.data[idx]
        else:
            # Set seed for reproducibility
            if self.seed is not None:
                np.random.seed(self.seed + idx)
            sample = self._generate_sample()

        # Convert IQ data to tensor (real and imaginary parts)
        iq_data = sample['iq_data']
        iq_tensor = torch.stack([
            torch.from_numpy(iq_data.real).float(),
            torch.from_numpy(iq_data.imag).float()
        ])

        # Convert labels to tensors
        labels = sample['labels']
        label_tensors = {
            'signal_present': torch.tensor(labels['signal_present'], dtype=torch.float32),
            'center_freq': torch.tensor(labels['center_freq'], dtype=torch.float32),
            'bandwidth': torch.tensor(labels['bandwidth'], dtype=torch.float32),
            'power': torch.tensor(labels['power'], dtype=torch.float32),
            'snr_db': torch.tensor(labels['snr_db'], dtype=torch.float32),
            'symbol_rate': torch.tensor(labels['symbol_rate'], dtype=torch.float32),
            'modulation_class': torch.tensor(labels['modulation_class'], dtype=torch.long)
        }

        return iq_tensor, label_tensors

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        iq_data = torch.stack([item[0] for item in batch])

        labels = {
            key: torch.stack([item[1][key] for item in batch])
            for key in batch[0][1].keys()
        }

        return iq_data, labels


class InferenceDataset(Dataset):
    """Dataset for inference on pre-recorded IQ data"""

    def __init__(self, iq_data: np.ndarray):
        """
        Args:
            iq_data: Complex IQ data array of shape (n_samples, sequence_length)
        """
        self.iq_data = iq_data

    def __len__(self) -> int:
        return len(self.iq_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single IQ sequence"""
        iq_sequence = self.iq_data[idx]

        # Convert to tensor
        iq_tensor = torch.stack([
            torch.from_numpy(iq_sequence.real).float(),
            torch.from_numpy(iq_sequence.imag).float()
        ])

        return iq_tensor
