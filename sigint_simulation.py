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

# Set matplotlib to use standard fonts (no Korean required)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
        """Generate chirp signal (linear FM sweep)"""
        # Generate instantaneous frequency sweep
        instantaneous_phase = 2 * np.pi * (f0 * self.t + (f1 - f0) / (2 * self.duration) * self.t**2)
        complex_signal = np.exp(1j * instantaneous_phase)
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

    def generate_fhss(self, freq_set=[60, 100, 140, 180], hop_rate=20):
        """Generate FHSS (Frequency Hopping Spread Spectrum) signal"""
        # Lower hop rate (20 Hz) = longer hop duration (0.05s) for better visibility
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

def compute_spectrogram(sig, fs=1000, nperseg=64):
    """Compute spectrogram of signal with better time resolution"""
    # Reduced nperseg from 128 to 64 for better time resolution (0.064s window)
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

    # 3. Center Frequency and Peak Frequency
    peak_idx = np.argmax(fft_power_pos)
    params['peak_frequency'] = fft_freq_pos[peak_idx]
    max_power_db = fft_power_db[peak_idx]

    # More robust signal threshold: 10dB below peak, but at least 15dB above noise floor
    signal_threshold_db = max(max_power_db - 10, noise_floor_db + 15)
    signal_mask = fft_power_db >= signal_threshold_db

    # Weighted center frequency (only from signal bins)
    if np.any(signal_mask):
        signal_power_sum = np.sum(fft_power_pos[signal_mask])
        params['center_frequency'] = np.sum(fft_freq_pos[signal_mask] * fft_power_pos[signal_mask]) / signal_power_sum
    else:
        params['center_frequency'] = params['peak_frequency']

    # 4. Bandwidth calculations using multiple methods
    freq_resolution = fft_freq_pos[1] - fft_freq_pos[0]

    # Method 1: 3dB Bandwidth (half-power bandwidth)
    threshold_3db = max_power_db - 3
    above_3db_mask = fft_power_db >= threshold_3db

    if np.any(above_3db_mask):
        # Find indices above 3dB threshold
        indices_3db = np.where(above_3db_mask)[0]

        # Find largest contiguous region containing peak
        peak_in_region = False
        max_region_size = 0
        best_region = (indices_3db[0], indices_3db[0])

        # Split into contiguous regions
        region_starts = [indices_3db[0]]
        for i in range(1, len(indices_3db)):
            if indices_3db[i] - indices_3db[i-1] > 1:  # Gap detected
                region_starts.append(indices_3db[i])

        # Find region containing peak
        for i, start_idx in enumerate(region_starts):
            if i < len(region_starts) - 1:
                end_idx = region_starts[i+1] - 1
                region_indices = indices_3db[(indices_3db >= start_idx) & (indices_3db < region_starts[i+1])]
            else:
                region_indices = indices_3db[indices_3db >= start_idx]

            if peak_idx in region_indices:
                best_region = (region_indices[0], region_indices[-1])
                peak_in_region = True
                break

        if peak_in_region:
            params['bandwidth_3db'] = fft_freq_pos[best_region[1]] - fft_freq_pos[best_region[0]]
        else:
            params['bandwidth_3db'] = fft_freq_pos[indices_3db[-1]] - fft_freq_pos[indices_3db[0]]
    else:
        params['bandwidth_3db'] = freq_resolution

    # Method 2: 99% Power Bandwidth (using cumulative power in signal region only)
    if np.any(signal_mask):
        signal_indices = np.where(signal_mask)[0]
        signal_freqs = fft_freq_pos[signal_indices]
        signal_powers = fft_power_pos[signal_indices]

        # Sort by power
        sorted_power_indices = np.argsort(signal_powers)[::-1]
        cumsum_power = np.cumsum(signal_powers[sorted_power_indices])
        total_signal_power = cumsum_power[-1]

        # Find frequencies that contain 99% of power
        power_99_idx = np.searchsorted(cumsum_power, 0.99 * total_signal_power)
        significant_freq_indices = sorted_power_indices[:power_99_idx+1]
        significant_freqs = signal_freqs[significant_freq_indices]

        params['bandwidth'] = np.max(significant_freqs) - np.min(significant_freqs)
    else:
        params['bandwidth'] = freq_resolution

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
    """Multi-task dataset for SIGINT signal classification and parameter estimation"""

    def __init__(self, n_samples=1000, snr_range=(-15, 5)):
        self.n_samples = n_samples
        self.data = []
        self.labels = []
        self.parameters = []  # Store ground truth parameters
        self.signal_types = ['Sine', 'Chirp', 'FSK', 'FHSS']

        # Ground truth parameters for each signal type
        self.gt_params = {
            0: {'fc': 100.0, 'bw': 0.0},      # Sine
            1: {'fc': 125.0, 'bw': 150.0},    # Chirp
            2: {'fc': 100.0, 'bw': 40.0},     # FSK
            3: {'fc': 120.0, 'bw': 120.0}     # FHSS
        }

        print("Generating dataset with ground truth parameters...")
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

            # Calculate actual signal power
            signal_power = np.mean(np.abs(sig)**2)
            power_dbm = 10 * np.log10(signal_power * 1000 + 1e-10)

            # Store data, label, and ground truth parameters
            self.data.append(Sxx_resized)
            self.labels.append(signal_type)
            self.parameters.append({
                'center_freq': self.gt_params[signal_type]['fc'],
                'bandwidth': self.gt_params[signal_type]['bw'],
                'power_dbm': power_dbm,
                'snr_db': snr
            })

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
        y_class = torch.LongTensor([self.labels[idx]])[0]

        # Ground truth parameters as regression targets
        params = self.parameters[idx]
        y_params = torch.FloatTensor([
            params['center_freq'] / 200.0,  # Normalize to [0, 1] range (max 200 Hz)
            params['bandwidth'] / 200.0,     # Normalize to [0, 1] range
            (params['power_dbm'] + 50) / 100.0,  # Normalize: [-50, 50] dBm -> [0, 1]
            (params['snr_db'] + 15) / 20.0    # Normalize: [-15, 5] dB -> [0, 1]
        ])

        return x, y_class, y_params

# ==================== Neural Network Model ====================

class SIGINTClassifier(nn.Module):
    """Multi-task CNN for SIGINT signal classification and parameter estimation"""

    def __init__(self, num_classes=4):
        super(SIGINTClassifier, self).__init__()

        # Shared feature extractor
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

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

        # Parameter estimation head (4 outputs: fc, bw, power, snr)
        self.param_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Output in [0, 1] range (normalized)
        )

    def forward(self, x):
        # Shared feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)

        # Task-specific heads
        class_output = self.classifier(x)
        param_output = self.param_estimator(x)

        return class_output, param_output

# ==================== Training ====================

def train_model(model, train_loader, val_loader, num_epochs=20):
    """Train the multi-task SIGINT classifier"""
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_params = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    history = {
        'train_loss': [], 'train_acc': [], 'train_param_loss': [],
        'val_loss': [], 'val_acc': [], 'val_param_loss': []
    }

    print("\nTraining multi-task model...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_class_loss = 0
        train_param_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels, params in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            params = params.to(device)

            optimizer.zero_grad()

            # Forward pass
            class_outputs, param_outputs = model(inputs)

            # Multi-task loss
            loss_class = criterion_class(class_outputs, labels)
            loss_params = criterion_params(param_outputs, params)
            loss = loss_class + 0.5 * loss_params  # Weight parameter loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_class_loss += loss_class.item()
            train_param_loss += loss_params.item()

            _, predicted = class_outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_class_loss = train_class_loss / len(train_loader)
        train_param_loss = train_param_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_class_loss = 0
        val_param_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels, params in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                params = params.to(device)

                class_outputs, param_outputs = model(inputs)

                loss_class = criterion_class(class_outputs, labels)
                loss_params = criterion_params(param_outputs, params)
                loss = loss_class + 0.5 * loss_params

                val_loss += loss.item()
                val_class_loss += loss_class.item()
                val_param_loss += loss_params.item()

                _, predicted = class_outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_class_loss = val_class_loss / len(val_loader)
        val_param_loss = val_param_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_param_loss'].append(train_param_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_param_loss'].append(val_param_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Param: {train_param_loss:.4f}), "
                  f"Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    return history

# ==================== Visualization ====================

def plot_sample_spectrograms(model):
    """Generate and plot sample spectrograms with AI-based parameter estimation"""
    gen = SignalGenerator(snr_db=5)  # Higher SNR for accurate parameter extraction

    signals = {
        'Sine': gen.generate_sine(),
        'Chirp': gen.generate_chirp(),
        'FSK': gen.generate_fsk(),
        'FHSS': gen.generate_fhss()
    }

    signal_types = ['Sine', 'Chirp', 'FSK', 'FHSS']

    # Define ground truth parameters for comparison
    ground_truth = {
        'Sine': {
            'center_frequency': 100.0,
            'bandwidth': 0.0,
            'modulation': 'Sine'
        },
        'Chirp': {
            'center_frequency': 125.0,  # (50 + 200) / 2
            'bandwidth': 150.0,  # 200 - 50
            'modulation': 'Chirp'
        },
        'FSK': {
            'center_frequency': 100.0,  # (80 + 120) / 2
            'bandwidth': 40.0,  # 120 - 80
            'modulation': 'FSK'
        },
        'FHSS': {
            'center_frequency': 120.0,  # (60 + 100 + 140 + 180) / 4
            'bandwidth': 120.0,  # 180 - 60
            'modulation': 'FHSS'
        }
    }

    # Use AI model to estimate parameters
    print("\n" + "="*70)
    print("AI-Based Signal Parameter Analysis")
    print("="*70)

    model.eval()
    all_params = {}

    with torch.no_grad():
        for idx, (name, sig) in enumerate(signals.items()):
            # Compute spectrogram
            _, _, Sxx = compute_spectrogram(sig)

            # Resize to model input size
            from scipy.ndimage import zoom
            zoom_factors = (64 / Sxx.shape[0], 64 / Sxx.shape[1])
            Sxx_resized = zoom(Sxx, zoom_factors, order=1)

            # Prepare input tensor
            x = torch.FloatTensor(Sxx_resized).unsqueeze(0).unsqueeze(0).to(device)

            # Get AI prediction
            class_output, param_output = model(x)

            # Get predicted class
            _, predicted_class = class_output.max(1)
            predicted_signal = signal_types[predicted_class.item()]

            # Denormalize predicted parameters
            params_norm = param_output.cpu().numpy()[0]
            center_freq = params_norm[0] * 200.0
            bandwidth = params_norm[1] * 200.0
            power_dbm = params_norm[2] * 100.0 - 50
            snr_db = params_norm[3] * 20.0 - 15

            all_params[name] = {
                'center_frequency': center_freq,
                'bandwidth': bandwidth,
                'power_dbm': power_dbm,
                'estimated_snr': snr_db,
                'modulation': predicted_signal
            }

            print(f"\n{'='*60}")
            print(f"Signal: {name}")
            print(f"{'='*60}")
            print(f"  AI Predicted Type:    {predicted_signal}")
            print(f"  Center Frequency:     {center_freq:.2f} Hz")
            print(f"  Bandwidth:            {bandwidth:.2f} Hz")
            print(f"  Signal Power:         {power_dbm:.2f} dBm")
            print(f"  Estimated SNR:        {snr_db:.2f} dB")
            print(f"{'='*60}")

    # Create figure with 4 rows (signals) x 3 columns (time, freq, spectrogram)
    fig = plt.figure(figsize=(20, 16))

    for idx, (name, sig) in enumerate(signals.items()):
        params = all_params[name]

        # Time domain plot (I/Q components)
        ax1 = plt.subplot(4, 3, idx*3 + 1)
        time_axis = np.linspace(0, gen.duration, len(sig))
        ax1.plot(time_axis[:500], np.real(sig[:500]), 'b-', linewidth=0.8, label='I (Real)', alpha=0.7)
        ax1.plot(time_axis[:500], np.imag(sig[:500]), 'r-', linewidth=0.8, label='Q (Imag)', alpha=0.7)

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{name} - Time Domain\nAI Predicted: {params["modulation"]}, Power: {params["power_dbm"]:.1f} dBm')
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

        # Mark AI-predicted center frequency
        ax2.axvline(params['center_frequency'], color='red', linestyle='--',
                   linewidth=1.5, label=f"AI Fc: {params['center_frequency']:.1f} Hz", alpha=0.7)

        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Magnitude [dB]')
        ax2.set_title(f'{name} - Frequency Spectrum\nAI BW: {params["bandwidth"]:.1f} Hz')
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

    # Create parameter summary table with ground truth comparison
    create_parameter_table(all_params, ground_truth)

def create_parameter_table(all_params, ground_truth):
    """Create a visual table comparing extracted vs ground truth parameters"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    ax1.axis('tight')
    ax1.axis('off')
    ax2.axis('tight')
    ax2.axis('off')

    # Table 1: AI-Estimated Parameters
    headers1 = ['Signal\nType',
                'AI Predicted\nType',
                'Center Freq\n(Fc) [Hz]',
                'Bandwidth\n[Hz]',
                'Power\n[dBm]',
                'Estimated\nSNR [dB]']

    table_data1 = []
    for name, params in all_params.items():
        row = [
            name,
            params['modulation'],
            f"{params['center_frequency']:.1f}",
            f"{params['bandwidth']:.1f}",
            f"{params['power_dbm']:.2f}",
            f"{params['estimated_snr']:.1f}"
        ]
        table_data1.append(row)

    # Create extracted parameters table
    table1 = ax1.table(cellText=table_data1, colLabels=headers1,
                       cellLoc='center', loc='center',
                       colWidths=[0.15, 0.2, 0.2, 0.15, 0.15, 0.15])

    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2.2)

    # Style the header for table 1
    for i in range(len(headers1)):
        cell = table1[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors for table 1
    for i in range(1, len(table_data1) + 1):
        for j in range(len(headers1)):
            cell = table1[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')

    ax1.set_title('AI-Estimated Signal Parameters', fontsize=12, fontweight='bold', pad=10)

    # Table 2: Ground Truth vs AI Estimation Comparison
    headers2 = ['Signal\nType',
                'Parameter',
                'Ground Truth\n[Hz]',
                'AI Predicted\n[Hz]',
                'Error\n[Hz]',
                'Error\n[%]']

    table_data2 = []
    for name in all_params.keys():
        gt = ground_truth[name]
        ex = all_params[name]

        # Center Frequency comparison
        fc_error = ex['center_frequency'] - gt['center_frequency']
        fc_error_pct = (fc_error / gt['center_frequency'] * 100) if gt['center_frequency'] != 0 else 0
        table_data2.append([
            name,
            'Center Freq',
            f"{gt['center_frequency']:.1f}",
            f"{ex['center_frequency']:.1f}",
            f"{fc_error:+.1f}",
            f"{fc_error_pct:+.1f}"
        ])

        # Bandwidth comparison
        bw_error = ex['bandwidth'] - gt['bandwidth']
        bw_error_pct = (bw_error / gt['bandwidth'] * 100) if gt['bandwidth'] != 0 else 0
        table_data2.append([
            '',
            'Bandwidth',
            f"{gt['bandwidth']:.1f}",
            f"{ex['bandwidth']:.1f}",
            f"{bw_error:+.1f}",
            f"{bw_error_pct:+.1f}" if gt['bandwidth'] != 0 else 'N/A'
        ])

    # Create comparison table
    table2 = ax2.table(cellText=table_data2, colLabels=headers2,
                       cellLoc='center', loc='center',
                       colWidths=[0.15, 0.18, 0.18, 0.18, 0.15, 0.16])

    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.8)

    # Style the header for table 2
    for i in range(len(headers2)):
        cell = table2[(0, i)]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(weight='bold', color='white')

    # Color code rows based on signal type and error magnitude
    signal_colors = {'Sine': '#E8F5E9', 'Chirp': '#FFF3E0', 'FSK': '#E3F2FD', 'FHSS': '#F3E5F5'}
    current_signal = None
    for i in range(1, len(table_data2) + 1):
        # Check if this is a new signal (first column not empty)
        if table_data2[i-1][0] != '':
            current_signal = table_data2[i-1][0]

        for j in range(len(headers2)):
            cell = table2[(i, j)]
            if current_signal:
                cell.set_facecolor(signal_colors.get(current_signal, 'white'))

            # Highlight large errors in red
            if j == 5 and table_data2[i-1][5] not in ['N/A', '']:  # Error % column
                try:
                    error_pct = float(table_data2[i-1][5])
                    if abs(error_pct) > 20:  # More than 20% error
                        cell.set_text_props(weight='bold', color='red')
                except:
                    pass

    ax2.set_title('Ground Truth vs AI-Estimated Parameters Comparison',
                  fontsize=12, fontweight='bold', pad=10)

    plt.suptitle('AI-Based Signal Parameter Analysis Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
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
    """Generate and plot confusion matrix with parameter estimation results"""
    model.eval()
    all_preds = []
    all_labels = []
    all_param_preds = []
    all_param_gt = []

    with torch.no_grad():
        for inputs, labels, params in test_loader:
            inputs = inputs.to(device)
            class_outputs, param_outputs = model(inputs)
            _, predicted = class_outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_param_preds.extend(param_outputs.cpu().numpy())
            all_param_gt.extend(params.numpy())

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

    # Print parameter estimation results
    all_param_preds = np.array(all_param_preds)
    all_param_gt = np.array(all_param_gt)

    # Denormalize parameters
    fc_pred = all_param_preds[:, 0] * 200.0
    fc_gt = all_param_gt[:, 0] * 200.0
    bw_pred = all_param_preds[:, 1] * 200.0
    bw_gt = all_param_gt[:, 1] * 200.0
    power_pred = all_param_preds[:, 2] * 100.0 - 50
    power_gt = all_param_gt[:, 2] * 100.0 - 50
    snr_pred = all_param_preds[:, 3] * 20.0 - 15
    snr_gt = all_param_gt[:, 3] * 20.0 - 15

    print("\nParameter Estimation Performance (AI Model):")
    print("="*60)
    print(f"  Center Frequency MAE:  {np.mean(np.abs(fc_pred - fc_gt)):.2f} Hz")
    print(f"  Bandwidth MAE:         {np.mean(np.abs(bw_pred - bw_gt)):.2f} Hz")
    print(f"  Power MAE:             {np.mean(np.abs(power_pred - power_gt)):.2f} dBm")
    print(f"  SNR MAE:               {np.mean(np.abs(snr_pred - snr_gt)):.2f} dB")
    print("="*60)

# ==================== Main ====================

def main():
    print("="*70)
    print("SIGINT Multi-Task Neural Receiver Simulation")
    print("="*70)

    # 1. Create dataset
    print("\n[1/4] Creating dataset...")
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

    # 2. Train model
    print("\n[2/4] Training multi-task classifier...")
    model = SIGINTClassifier(num_classes=4).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(model, train_loader, val_loader, num_epochs=20)

    # 3. Evaluate and visualize
    print("\n[3/4] Generating results...")
    plot_training_results(history)
    plot_confusion_matrix(model, test_loader)

    # 4. Generate sample spectrograms with AI-based parameter estimation
    print("\n[4/4] Generating sample spectrograms with AI parameter estimation...")
    plot_sample_spectrograms(model)

    # Save model
    torch.save(model.state_dict(), 'sigint_detector_model.pth')
    print("✓ Saved sigint_detector_model.pth")

    print("\n" + "="*70)
    print("✅ Simulation Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - sample_spectrograms.png      (Signal visualization: time/freq/spectrogram)")
    print("  - signal_parameters.png        (Signal parameter analysis table)")
    print("  - training_results.png         (Training results)")
    print("  - confusion_matrix.png         (Confusion matrix)")
    print("  - sigint_detector_model.pth    (Trained model)")

if __name__ == "__main__":
    main()
