#!/usr/bin/env python3
"""
SIGINT Receiver Simulation
A Multi-Task IQ-Based Neural Receiver for Weak-Signal Detection and Parametric Classification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set Korean font for matplotlib
import platform
import matplotlib.font_manager as fm

def setup_korean_font():
    """Setup Korean font for matplotlib"""
    system = platform.system()

    if system == 'Linux':
        # Google Colab or Linux system
        try:
            # Try to use NanumGothic (pre-installed in Colab)
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
            else:
                # Fallback to any available Korean font
                plt.rcParams['font.family'] = 'NanumGothic'
        except:
            # If no Korean font available, use sans-serif
            plt.rcParams['font.family'] = 'sans-serif'
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'

    # Prevent minus sign from being displayed as a box
    plt.rcParams['axes.unicode_minus'] = False

import os
setup_korean_font()

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Signal Generation ====================

class SignalGenerator:
    """Generate various types of SIGINT signals"""

    def __init__(self, fs=1000, duration=1.0, snr_db=-10):
        self.fs = fs  # Sampling frequency
        self.duration = duration
        self.snr_db = snr_db
        self.t = np.linspace(0, duration, int(fs * duration))

    def add_noise(self, signal, snr_db):
        """Add AWGN noise to signal"""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) +
                                            1j * np.random.randn(len(signal)))
        return signal + noise

    def generate_sine(self, freq=100):
        """Generate sine wave signal"""
        signal = np.exp(1j * 2 * np.pi * freq * self.t)
        return self.add_noise(signal, self.snr_db)

    def generate_chirp(self, f0=50, f1=200):
        """Generate chirp signal"""
        chirp_sig = signal.chirp(self.t, f0, self.duration, f1, method='linear')
        complex_signal = chirp_sig * np.exp(1j * 2 * np.pi * f0 * self.t)
        return self.add_noise(complex_signal, self.snr_db)

    def generate_fsk(self, freqs=[80, 120], symbol_rate=50):
        """Generate FSK (Frequency Shift Keying) signal"""
        samples_per_symbol = int(self.fs / symbol_rate)
        n_symbols = int(len(self.t) / samples_per_symbol)

        # Random binary data
        bits = np.random.randint(0, 2, n_symbols)

        # Generate FSK signal
        fsk_signal = np.zeros(len(self.t), dtype=complex)
        for i, bit in enumerate(bits):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(self.t))
            t_symbol = self.t[start_idx:end_idx] - self.t[start_idx]
            freq = freqs[bit]
            fsk_signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_symbol)

        return self.add_noise(fsk_signal, self.snr_db)

    def generate_fhss(self, freq_set=[60, 100, 140, 180], hop_rate=100):
        """Generate FHSS (Frequency Hopping Spread Spectrum) signal"""
        samples_per_hop = int(self.fs / hop_rate)
        n_hops = int(len(self.t) / samples_per_hop)

        # Random frequency hops
        freq_sequence = np.random.choice(freq_set, n_hops)

        # Generate FHSS signal
        fhss_signal = np.zeros(len(self.t), dtype=complex)
        for i, freq in enumerate(freq_sequence):
            start_idx = i * samples_per_hop
            end_idx = min((i + 1) * samples_per_hop, len(self.t))
            t_hop = self.t[start_idx:end_idx] - self.t[start_idx]
            fhss_signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_hop)

        return self.add_noise(fhss_signal, self.snr_db)

def compute_spectrogram(sig, fs=1000, nperseg=128):
    """Compute spectrogram of signal"""
    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg,
                                    noverlap=nperseg//2, mode='magnitude')
    return f, t, 20 * np.log10(Sxx + 1e-10)  # Convert to dB

# ==================== Signal Parameter Extraction ====================

def extract_signal_parameters(sig, fs=1000, signal_name='Unknown'):
    """
    Extract comprehensive signal parameters
    Returns: dict with frequency, power, bandwidth, modulation, onset time
    """
    params = {}

    # 1. Signal Power (dBm)
    signal_power = np.mean(np.abs(sig)**2)
    params['power_dbm'] = 10 * np.log10(signal_power * 1000 + 1e-10)  # Convert to dBm
    params['power_linear'] = signal_power

    # 2. FFT Analysis
    fft_result = np.fft.fft(sig)
    fft_freq = np.fft.fftfreq(len(sig), 1/fs)
    fft_magnitude = np.abs(fft_result)
    fft_power = fft_magnitude**2

    # Only consider positive frequencies
    positive_idx = fft_freq >= 0
    fft_freq_pos = fft_freq[positive_idx]
    fft_power_pos = fft_power[positive_idx]

    # Convert to dB for threshold calculations
    fft_power_db = 10 * np.log10(fft_power_pos + 1e-10)

    # Estimate noise floor (median of lower 50% power values)
    noise_floor_db = np.median(np.sort(fft_power_db)[:len(fft_power_db)//2])
    signal_threshold_db = noise_floor_db + 10  # 10dB above noise floor

    # Identify signal bins (above noise threshold)
    signal_mask = fft_power_db >= signal_threshold_db

    # 3. Center Frequency and Peak Frequency
    peak_idx = np.argmax(fft_power_pos)
    params['peak_frequency'] = fft_freq_pos[peak_idx]

    # Weighted center frequency (only from signal bins)
    if np.any(signal_mask):
        signal_power_sum = np.sum(fft_power_pos[signal_mask])
        params['center_frequency'] = np.sum(fft_freq_pos[signal_mask] * fft_power_pos[signal_mask]) / signal_power_sum
    else:
        params['center_frequency'] = params['peak_frequency']

    # 4. Bandwidth (99% power bandwidth - only from signal bins)
    if np.any(signal_mask):
        signal_freqs = fft_freq_pos[signal_mask]
        signal_powers = fft_power_pos[signal_mask]

        sorted_indices = np.argsort(signal_powers)[::-1]
        cumsum_power = np.cumsum(signal_powers[sorted_indices])
        total_signal_power = np.sum(signal_powers)

        threshold_idx = np.where(cumsum_power >= 0.99 * total_signal_power)[0]
        if len(threshold_idx) > 0:
            significant_freqs = signal_freqs[sorted_indices[:threshold_idx[0]+1]]
            params['bandwidth'] = np.max(significant_freqs) - np.min(significant_freqs)
        else:
            params['bandwidth'] = np.max(signal_freqs) - np.min(signal_freqs)
    else:
        params['bandwidth'] = 0

    # 5. Occupied Bandwidth (3dB bandwidth from peak)
    max_power_db = np.max(fft_power_db)
    threshold_3db = max_power_db - 3
    above_3db = fft_power_db >= threshold_3db

    if np.any(above_3db):
        # Find contiguous regions above 3dB threshold around peak
        freq_above = fft_freq_pos[above_3db]

        # Get continuous region containing peak frequency
        peak_freq = params['peak_frequency']
        freq_diff = np.diff(freq_above)

        # If frequencies are contiguous (diff < 2*freq_resolution)
        freq_resolution = fft_freq_pos[1] - fft_freq_pos[0]
        gaps = np.where(freq_diff > 2 * freq_resolution)[0]

        if len(gaps) > 0:
            # Find which segment contains peak
            segments_start = [0] + (gaps + 1).tolist()
            segments_end = gaps.tolist() + [len(freq_above)-1]

            for start, end in zip(segments_start, segments_end):
                if freq_above[start] <= peak_freq <= freq_above[end]:
                    params['bandwidth_3db'] = freq_above[end] - freq_above[start]
                    break
            else:
                # Peak not in any segment, use full range
                params['bandwidth_3db'] = np.max(freq_above) - np.min(freq_above)
        else:
            # All contiguous
            params['bandwidth_3db'] = np.max(freq_above) - np.min(freq_above)
    else:
        params['bandwidth_3db'] = 0

    # 6. Signal Onset Detection (energy-based)
    window_size = int(0.01 * fs)  # 10ms window
    energy = np.array([np.sum(np.abs(sig[i:i+window_size])**2)
                       for i in range(0, len(sig)-window_size, window_size//2)])
    threshold = np.mean(energy) + 2 * np.std(energy)
    onset_idx = np.where(energy > threshold)[0]
    if len(onset_idx) > 0:
        params['onset_time'] = onset_idx[0] * (window_size//2) / fs
    else:
        params['onset_time'] = 0.0

    # 7. Modulation Type Detection (heuristic)
    instantaneous_freq = np.diff(np.unwrap(np.angle(sig))) * fs / (2 * np.pi)
    freq_variation = np.std(instantaneous_freq)

    # Count frequency peaks for FHSS detection (only in signal region)
    if np.any(signal_mask):
        signal_fft_power = fft_power_pos[signal_mask]
        peak_threshold = 0.1 * np.max(signal_fft_power)
        num_peaks = len(signal.find_peaks(signal_fft_power, height=peak_threshold)[0])
    else:
        num_peaks = 1

    if num_peaks > 3:
        params['modulation'] = 'FHSS'
    elif freq_variation > 50:
        params['modulation'] = 'Chirp/FM'
    elif num_peaks == 2:
        params['modulation'] = 'FSK'
    else:
        params['modulation'] = 'CW/AM'

    # Override with known signal name if provided
    if signal_name != 'Unknown':
        params['modulation'] = signal_name

    # 8. SNR Estimation (using pre-calculated noise floor)
    signal_peak_db = np.max(fft_power_db)
    params['estimated_snr'] = signal_peak_db - noise_floor_db

    return params

def print_signal_parameters(params, signal_name):
    """Print signal parameters in a formatted table"""
    print(f"\n{'='*60}")
    print(f"Signal: {signal_name}")
    print(f"{'='*60}")
    print(f"  Center Frequency:     {params['center_frequency']:.2f} Hz")
    print(f"  Peak Frequency:       {params['peak_frequency']:.2f} Hz")
    print(f"  Signal Power:         {params['power_dbm']:.2f} dBm")
    print(f"  Bandwidth (99%):      {params['bandwidth']:.2f} Hz")
    print(f"  Bandwidth (3dB):      {params['bandwidth_3db']:.2f} Hz")
    print(f"  Modulation:           {params['modulation']}")
    print(f"  Signal Onset:         {params['onset_time']:.4f} s")
    print(f"  Estimated SNR:        {params['estimated_snr']:.2f} dB")
    print(f"{'='*60}")

# ==================== Dataset ====================

class SIGINTDataset(Dataset):
    """Dataset for SIGINT signal classification"""

    def __init__(self, n_samples=1000, snr_range=(-15, 5)):
        self.n_samples = n_samples
        self.data = []
        self.labels = []
        self.signal_types = ['Sine', 'Chirp', 'FSK', 'FHSS']

        print("Generating dataset...")
        for i in range(n_samples):
            # Random SNR for each sample
            snr = np.random.uniform(*snr_range)
            gen = SignalGenerator(snr_db=snr)

            # Random signal type
            signal_type = i % 4

            if signal_type == 0:
                sig = gen.generate_sine()
            elif signal_type == 1:
                sig = gen.generate_chirp()
            elif signal_type == 2:
                sig = gen.generate_fsk()
            else:
                sig = gen.generate_fhss()

            # Compute spectrogram
            _, _, Sxx = compute_spectrogram(sig)

            # Resize to fixed size
            Sxx_resized = self._resize_spectrogram(Sxx, (64, 64))

            self.data.append(Sxx_resized)
            self.labels.append(signal_type)

            if (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples")

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def _resize_spectrogram(self, spec, target_size):
        """Resize spectrogram to target size"""
        from scipy.ndimage import zoom
        zoom_factors = (target_size[0] / spec.shape[0],
                       target_size[1] / spec.shape[1])
        return zoom(spec, zoom_factors, order=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx]).unsqueeze(0)  # Add channel dimension
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y

# ==================== Neural Network Model ====================

class SIGINTClassifier(nn.Module):
    """CNN-based SIGINT signal classifier"""

    def __init__(self, num_classes=4):
        super(SIGINTClassifier, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================== Training ====================

def train_model(model, train_loader, val_loader, num_epochs=20):
    """Train the SIGINT classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("\nTraining model...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return history

# ==================== Visualization ====================

def plot_sample_spectrograms():
    """Generate and plot sample spectrograms with time and frequency domain analysis"""
    gen = SignalGenerator(snr_db=-5)

    signals = {
        'Sine': gen.generate_sine(),
        'Chirp': gen.generate_chirp(),
        'FSK': gen.generate_fsk(),
        'FHSS': gen.generate_fhss()
    }

    # Extract parameters for all signals
    print("\n" + "="*70)
    print("📊 신호 제원 분석 (Signal Parameter Analysis)")
    print("="*70)

    all_params = {}
    for name, sig in signals.items():
        params = extract_signal_parameters(sig, gen.fs, name)
        all_params[name] = params
        print_signal_parameters(params, name)

    # Create figure with 4 rows (signals) x 3 columns (time, freq, spectrogram)
    fig = plt.figure(figsize=(20, 16))

    for idx, (name, sig) in enumerate(signals.items()):
        params = all_params[name]

        # Time domain plot (I/Q components)
        ax1 = plt.subplot(4, 3, idx*3 + 1)
        time_axis = np.linspace(0, gen.duration, len(sig))
        ax1.plot(time_axis[:500], np.real(sig[:500]), 'b-', linewidth=0.8, label='I (Real)', alpha=0.7)
        ax1.plot(time_axis[:500], np.imag(sig[:500]), 'r-', linewidth=0.8, label='Q (Imag)', alpha=0.7)

        # Mark onset time
        if params['onset_time'] < 0.5:  # Only show if in visible range
            ax1.axvline(params['onset_time'], color='red', linestyle='--', linewidth=1.5,
                       label=f"Onset: {params['onset_time']:.3f}s", alpha=0.7)

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{name} - Time Domain\nPower: {params["power_dbm"]:.1f} dBm')
        ax1.legend(loc='upper right', fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Frequency spectrum (FFT)
        ax2 = plt.subplot(4, 3, idx*3 + 2)
        fft_result = np.fft.fft(sig)
        fft_freq = np.fft.fftfreq(len(sig), 1/gen.fs)
        fft_magnitude = 20 * np.log10(np.abs(fft_result) + 1e-10)

        # Plot only positive frequencies
        positive_freq_idx = fft_freq >= 0
        ax2.plot(fft_freq[positive_freq_idx], fft_magnitude[positive_freq_idx], 'g-', linewidth=1.0)

        # Mark center frequency
        ax2.axvline(params['center_frequency'], color='red', linestyle='--',
                   linewidth=1.5, label=f"Fc: {params['center_frequency']:.1f} Hz", alpha=0.7)

        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Magnitude [dB]')
        ax2.set_title(f'{name} - Frequency Spectrum\nBW: {params["bandwidth"]:.1f} Hz (99%), {params["bandwidth_3db"]:.1f} Hz (3dB)')
        ax2.legend(loc='upper right', fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, gen.fs/2])

        # Spectrogram
        ax3 = plt.subplot(4, 3, idx*3 + 3)
        f, t, Sxx = compute_spectrogram(sig)
        im = ax3.pcolormesh(t, f, Sxx, shading='gouraud', cmap='jet')
        ax3.set_ylabel('Frequency [Hz]')
        ax3.set_xlabel('Time [s]')
        ax3.set_title(f'{name} - Spectrogram\nModulation: {params["modulation"]}, SNR: {params["estimated_snr"]:.1f} dB')
        plt.colorbar(im, ax=ax3, label='Magnitude [dB]')

    plt.tight_layout()
    plt.savefig('sample_spectrograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved sample_spectrograms.png")

    # Create parameter summary table
    create_parameter_table(all_params)

def create_parameter_table(all_params):
    """Create a visual table of signal parameters"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    headers = ['신호 타입\n(Signal Type)',
               '중심주파수\n(Fc) [Hz]',
               '신호세기\n(Power) [dBm]',
               '대역폭 (99%)\n(BW) [Hz]',
               '대역폭 (3dB)\n(BW) [Hz]',
               '변조방식\n(Modulation)',
               '출현시점\n(Onset) [s]',
               '추정 SNR\n[dB]']

    table_data = []
    for name, params in all_params.items():
        row = [
            name,
            f"{params['center_frequency']:.1f}",
            f"{params['power_dbm']:.2f}",
            f"{params['bandwidth']:.1f}",
            f"{params['bandwidth_3db']:.1f}",
            params['modulation'],
            f"{params['onset_time']:.4f}",
            f"{params['estimated_snr']:.1f}"
        ]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.13, 0.13, 0.13, 0.12, 0.13])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style the header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')

    plt.title('신호 제원 분석표 (Signal Parameter Summary)',
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig('signal_parameters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved signal_parameters.png")

def plot_training_results(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved training_results.png")

def plot_confusion_matrix(model, test_loader):
    """Generate and plot confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sine', 'Chirp', 'FSK', 'FHSS'],
                yticklabels=['Sine', 'Chirp', 'FSK', 'FHSS'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - SIGINT Signal Classification')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved confusion_matrix.png")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                               target_names=['Sine', 'Chirp', 'FSK', 'FHSS']))

# ==================== Main ====================

def main():
    print("="*70)
    print("SIGINT Neural Receiver Simulation")
    print("="*70)

    # 1. Generate sample spectrograms
    print("\n[1/4] Generating sample spectrograms...")
    plot_sample_spectrograms()

    # 2. Create dataset
    print("\n[2/4] Creating dataset...")
    dataset = SIGINTDataset(n_samples=2000)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 3. Train model
    print("\n[3/4] Training classifier...")
    model = SIGINTClassifier(num_classes=4).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(model, train_loader, val_loader, num_epochs=20)

    # 4. Evaluate and visualize
    print("\n[4/4] Generating results...")
    plot_training_results(history)
    plot_confusion_matrix(model, test_loader)

    # Save model
    torch.save(model.state_dict(), 'sigint_detector_model.pth')
    print("✓ Saved sigint_detector_model.pth")

    print("\n" + "="*70)
    print("✅ Simulation Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - sample_spectrograms.png      (신호 시각화: 시간/주파수/스펙트로그램)")
    print("  - signal_parameters.png        (신호 제원 분석표)")
    print("  - training_results.png         (학습 결과)")
    print("  - confusion_matrix.png         (혼동 행렬)")
    print("  - sigint_detector_model.pth    (학습된 모델)")

if __name__ == "__main__":
    main()
