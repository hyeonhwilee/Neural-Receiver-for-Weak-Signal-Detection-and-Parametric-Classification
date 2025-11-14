"""
Visualization utilities for signal analysis and model interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import torch


def plot_iq_data(
    iq_data: np.ndarray,
    title: str = "IQ Data",
    sample_rate: float = 1.0,
    save_path: Optional[str] = None
):
    """
    Plot IQ data in time domain

    Args:
        iq_data: Complex IQ data (n_samples,)
        title: Plot title
        sample_rate: Sample rate in Hz
        save_path: Path to save figure
    """
    n_samples = len(iq_data)
    time = np.arange(n_samples) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # I/Q components
    axes[0].plot(time, iq_data.real, 'b-', alpha=0.7, label='I (Real)')
    axes[0].plot(time, iq_data.imag, 'r-', alpha=0.7, label='Q (Imaginary)')
    axes[0].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'{title} - Time Domain', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Magnitude
    magnitude = np.abs(iq_data)
    axes[1].plot(time, magnitude, 'g-', alpha=0.7)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Magnitude', fontsize=11)
    axes[1].set_title('Magnitude', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Phase
    phase = np.angle(iq_data)
    axes[2].plot(time, phase, 'm-', alpha=0.7)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Phase (radians)', fontsize=11)
    axes[2].set_title('Phase', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_spectrogram(
    iq_data: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 256,
    title: str = "Spectrogram",
    save_path: Optional[str] = None
):
    """
    Plot spectrogram of IQ data

    Args:
        iq_data: Complex IQ data (n_samples,)
        sample_rate: Sample rate in Hz
        nperseg: Length of each segment for STFT
        title: Plot title
        save_path: Path to save figure
    """
    from scipy import signal

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(
        iq_data,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        mode='magnitude'
    )

    # Shift zero frequency to center
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f)

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_psd(
    iq_data: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 1024,
    title: str = "Power Spectral Density",
    save_path: Optional[str] = None
):
    """
    Plot power spectral density

    Args:
        iq_data: Complex IQ data (n_samples,)
        sample_rate: Sample rate in Hz
        nperseg: Length of each segment for Welch's method
        title: Plot title
        save_path: Path to save figure
    """
    from scipy import signal

    # Compute PSD using Welch's method
    f, psd = signal.welch(
        iq_data,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False
    )

    # Shift zero frequency to center
    psd = np.fft.fftshift(psd)
    f = np.fft.fftshift(f)

    # Convert to dB
    psd_db = 10 * np.log10(psd + 1e-10)

    plt.figure(figsize=(12, 6))
    plt.plot(f, psd_db, 'b-', linewidth=2)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power/Frequency (dB/Hz)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_constellation(
    iq_data: np.ndarray,
    title: str = "Constellation Diagram",
    max_points: int = 10000,
    save_path: Optional[str] = None
):
    """
    Plot constellation diagram

    Args:
        iq_data: Complex IQ data (n_samples,)
        title: Plot title
        max_points: Maximum number of points to plot
        save_path: Path to save figure
    """
    # Subsample if necessary
    if len(iq_data) > max_points:
        indices = np.random.choice(len(iq_data), max_points, replace=False)
        iq_data = iq_data[indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(iq_data.real, iq_data.imag, alpha=0.3, s=10)
    plt.xlabel('In-Phase (I)', fontsize=12)
    plt.ylabel('Quadrature (Q)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_signal_examples(
    dataset,
    n_examples: int = 4,
    save_path: Optional[str] = None
):
    """
    Plot examples from dataset

    Args:
        dataset: SignalDataset instance
        n_examples: Number of examples to plot
        save_path: Path to save figure
    """
    from ..data import ModulationType

    class_names = ModulationType.get_class_names()

    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))

    for i in range(n_examples):
        iq_tensor, labels = dataset[i]

        # Convert to numpy
        iq_data = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()

        # Get labels
        modulation_class = labels['modulation_class'].item()
        signal_present = labels['signal_present'].item()
        snr_db = labels['snr_db'].item()

        class_name = class_names[modulation_class]

        # Time domain
        time = np.arange(len(iq_data))
        axes[i, 0].plot(time, iq_data.real, 'b-', alpha=0.7, label='I')
        axes[i, 0].plot(time, iq_data.imag, 'r-', alpha=0.7, label='Q')
        axes[i, 0].set_xlabel('Sample', fontsize=10)
        axes[i, 0].set_ylabel('Amplitude', fontsize=10)
        axes[i, 0].set_title(f'{class_name} (SNR={snr_db:.1f} dB)', fontsize=11)
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        # Magnitude spectrum
        fft = np.fft.fftshift(np.fft.fft(iq_data))
        freq = np.fft.fftshift(np.fft.fftfreq(len(iq_data)))
        mag_db = 20 * np.log10(np.abs(fft) + 1e-10)

        axes[i, 1].plot(freq, mag_db, 'g-', linewidth=1)
        axes[i, 1].set_xlabel('Normalized Frequency', fontsize=10)
        axes[i, 1].set_ylabel('Magnitude (dB)', fontsize=10)
        axes[i, 1].set_title('Spectrum', fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)

        # Constellation
        axes[i, 2].scatter(iq_data.real, iq_data.imag, alpha=0.3, s=5)
        axes[i, 2].set_xlabel('I', fontsize=10)
        axes[i, 2].set_ylabel('Q', fontsize=10)
        axes[i, 2].set_title('Constellation', fontsize=11)
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].axis('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot training history

    Args:
        history: Dictionary with training history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    axes[1].plot(history['learning_rate'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_snr_performance(
    model,
    dataset,
    snr_range: Tuple[float, float] = (-20, 10),
    n_snr_points: int = 15,
    n_samples_per_snr: int = 100,
    device: str = 'cpu',
    save_path: Optional[str] = None
):
    """
    Plot model performance vs SNR

    Args:
        model: Trained model
        dataset: Dataset class (not instance)
        snr_range: Range of SNR values to test
        n_snr_points: Number of SNR points
        n_samples_per_snr: Number of samples per SNR point
        device: Device to use
        save_path: Path to save figure
    """
    from torch.utils.data import DataLoader
    from ..data import SignalDataset

    model.eval()
    model.to(device)

    snr_values = np.linspace(snr_range[0], snr_range[1], n_snr_points)

    detection_acc = []
    classification_acc = []
    regression_mae = []

    for snr_db in snr_values:
        # Create dataset with specific SNR
        test_dataset = SignalDataset(
            n_samples=n_samples_per_snr,
            sequence_length=1024,
            snr_range=(snr_db, snr_db),  # Fixed SNR
            no_signal_prob=0.0,  # Only signals
            seed=42
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=SignalDataset.collate_fn
        )

        # Evaluate
        det_correct = 0
        cls_correct = 0
        reg_errors = []
        total = 0

        with torch.no_grad():
            for iq_data, labels in test_loader:
                iq_data = iq_data.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(iq_data)

                # Detection
                det_pred = (outputs['detection_prob'] > 0.5).float()
                det_correct += (det_pred == labels['signal_present']).sum().item()

                # Classification
                cls_pred = torch.argmax(outputs['classification_logits'], dim=1)
                cls_correct += (cls_pred == labels['modulation_class']).sum().item()

                # Regression
                reg_targets = torch.stack([
                    labels['center_freq'],
                    labels['bandwidth'],
                    labels['snr_db'],
                    labels['symbol_rate']
                ], dim=1)

                reg_error = torch.abs(outputs['regression_params'] - reg_targets).mean(dim=1)
                reg_errors.extend(reg_error.cpu().numpy())

                total += iq_data.size(0)

        detection_acc.append(det_correct / total)
        classification_acc.append(cls_correct / total)
        regression_mae.append(np.mean(reg_errors))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(snr_values, detection_acc, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Detection Accuracy', fontsize=12)
    axes[0].set_title('Detection Performance vs SNR', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(snr_values, classification_acc, 'r-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('Classification Accuracy', fontsize=12)
    axes[1].set_title('Classification Performance vs SNR', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    axes[2].plot(snr_values, regression_mae, 'g-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('SNR (dB)', fontsize=12)
    axes[2].set_ylabel('Regression MAE', fontsize=12)
    axes[2].set_title('Parameter Estimation Error vs SNR', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
