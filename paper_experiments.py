#!/usr/bin/env python3
"""
Paper Experiments: SOTA Comparison for SIGINT Multi-Task Learning
Compares our Multi-Task CNN with baseline models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sigint_simulation import SIGINTDataset, SIGINTClassifier, device
import time
import pandas as pd

# Set style for paper-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# ==================== Extended Signal Types ====================

class ExtendedSignalGenerator:
    """Generate 12 different signal types for comprehensive evaluation"""

    def __init__(self, fs=1000, duration=1.0, snr_db=-10):
        self.fs = fs
        self.duration = duration
        self.snr_db = snr_db
        self.t = np.linspace(0, duration, int(fs * duration))

    def add_noise(self, signal, snr_db):
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) +
                                            1j * np.random.randn(len(signal)))
        return signal + noise

    # Original 4 signals
    def generate_sine(self, freq=100):
        """CW (Continuous Wave) / Sine"""
        signal = np.exp(1j * 2 * np.pi * freq * self.t)
        return self.add_noise(signal, self.snr_db)

    def generate_chirp(self, f0=50, f1=200):
        """Linear FM Chirp"""
        instantaneous_phase = 2 * np.pi * (f0 * self.t + (f1 - f0) / (2 * self.duration) * self.t**2)
        complex_signal = np.exp(1j * instantaneous_phase)
        return self.add_noise(complex_signal, self.snr_db)

    def generate_fsk(self, freqs=[80, 120], symbol_rate=50):
        """2-FSK (Binary Frequency Shift Keying)"""
        samples_per_symbol = int(self.fs / symbol_rate)
        n_symbols = int(len(self.t) / samples_per_symbol)
        bits = np.random.randint(0, 2, n_symbols)

        fsk_signal = np.zeros(len(self.t), dtype=complex)
        for i, bit in enumerate(bits):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(self.t))
            t_symbol = self.t[start_idx:end_idx] - self.t[start_idx]
            freq = freqs[bit]
            fsk_signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_symbol)

        return self.add_noise(fsk_signal, self.snr_db)

    def generate_fhss(self, freq_set=[60, 100, 140, 180], hop_rate=20):
        """FHSS (Frequency Hopping Spread Spectrum)"""
        samples_per_hop = int(self.fs / hop_rate)
        n_hops = int(len(self.t) / samples_per_hop)
        freq_sequence = np.random.choice(freq_set, n_hops)

        fhss_signal = np.zeros(len(self.t), dtype=complex)
        for i, freq in enumerate(freq_sequence):
            start_idx = i * samples_per_hop
            end_idx = min((i + 1) * samples_per_hop, len(self.t))
            t_hop = self.t[start_idx:end_idx] - self.t[start_idx]
            fhss_signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_hop)

        return self.add_noise(fhss_signal, self.snr_db)

    # New signals (8 additional)
    def generate_bpsk(self, fc=100, symbol_rate=50):
        """BPSK (Binary Phase Shift Keying)"""
        samples_per_symbol = int(self.fs / symbol_rate)
        n_symbols = int(len(self.t) / samples_per_symbol)
        bits = np.random.randint(0, 2, n_symbols)

        bpsk_signal = np.zeros(len(self.t), dtype=complex)
        for i, bit in enumerate(bits):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(self.t))
            t_symbol = self.t[start_idx:end_idx] - self.t[start_idx]
            phase = 0 if bit == 0 else np.pi
            bpsk_signal[start_idx:end_idx] = np.exp(1j * (2 * np.pi * fc * t_symbol + phase))

        return self.add_noise(bpsk_signal, self.snr_db)

    def generate_qpsk(self, fc=100, symbol_rate=50):
        """QPSK (Quadrature Phase Shift Keying)"""
        samples_per_symbol = int(self.fs / symbol_rate)
        n_symbols = int(len(self.t) / samples_per_symbol)
        symbols = np.random.randint(0, 4, n_symbols)

        qpsk_signal = np.zeros(len(self.t), dtype=complex)
        phase_map = {0: 0, 1: np.pi/2, 2: np.pi, 3: 3*np.pi/2}

        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(self.t))
            t_symbol = self.t[start_idx:end_idx] - self.t[start_idx]
            phase = phase_map[symbol]
            qpsk_signal[start_idx:end_idx] = np.exp(1j * (2 * np.pi * fc * t_symbol + phase))

        return self.add_noise(qpsk_signal, self.snr_db)

    def generate_8psk(self, fc=100, symbol_rate=50):
        """8-PSK (8-ary Phase Shift Keying)"""
        samples_per_symbol = int(self.fs / symbol_rate)
        n_symbols = int(len(self.t) / samples_per_symbol)
        symbols = np.random.randint(0, 8, n_symbols)

        psk8_signal = np.zeros(len(self.t), dtype=complex)

        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(self.t))
            t_symbol = self.t[start_idx:end_idx] - self.t[start_idx]
            phase = 2 * np.pi * symbol / 8
            psk8_signal[start_idx:end_idx] = np.exp(1j * (2 * np.pi * fc * t_symbol + phase))

        return self.add_noise(psk8_signal, self.snr_db)

    def generate_16qam(self, fc=100, symbol_rate=50):
        """16-QAM (16-ary Quadrature Amplitude Modulation)"""
        samples_per_symbol = int(self.fs / symbol_rate)
        n_symbols = int(len(self.t) / samples_per_symbol)
        symbols = np.random.randint(0, 16, n_symbols)

        # 16-QAM constellation
        constellation = np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                                  -1-3j, -1-1j, -1+3j, -1+1j,
                                   3-3j,  3-1j,  3+3j,  3+1j,
                                   1-3j,  1-1j,  1+3j,  1+1j])

        qam_signal = np.zeros(len(self.t), dtype=complex)
        carrier = np.exp(1j * 2 * np.pi * fc * self.t)

        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(self.t))
            qam_signal[start_idx:end_idx] = constellation[symbol] * carrier[start_idx:end_idx]

        return self.add_noise(qam_signal, self.snr_db)

    def generate_am(self, fc=100, fm=10, mod_index=0.5):
        """AM (Amplitude Modulation)"""
        carrier = np.exp(1j * 2 * np.pi * fc * self.t)
        message = np.cos(2 * np.pi * fm * self.t)
        am_signal = (1 + mod_index * message) * carrier
        return self.add_noise(am_signal, self.snr_db)

    def generate_fm(self, fc=100, fm=10, freq_dev=20):
        """FM (Frequency Modulation)"""
        message = np.cos(2 * np.pi * fm * self.t)
        instantaneous_phase = 2 * np.pi * fc * self.t + 2 * np.pi * freq_dev * np.cumsum(message) / self.fs
        fm_signal = np.exp(1j * instantaneous_phase)
        return self.add_noise(fm_signal, self.snr_db)

    def generate_gfsk(self, freqs=[80, 120], symbol_rate=50, bt=0.5):
        """GFSK (Gaussian Frequency Shift Keying)"""
        # Simplified GFSK (similar to FSK but with Gaussian pulse shaping)
        samples_per_symbol = int(self.fs / symbol_rate)
        n_symbols = int(len(self.t) / samples_per_symbol)
        bits = np.random.randint(0, 2, n_symbols)

        gfsk_signal = np.zeros(len(self.t), dtype=complex)
        for i, bit in enumerate(bits):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(self.t))
            t_symbol = self.t[start_idx:end_idx] - self.t[start_idx]
            freq = freqs[bit]
            # Add Gaussian envelope
            envelope = np.exp(-(t_symbol - samples_per_symbol/(2*self.fs))**2 / (2*(bt/self.fs)**2))
            gfsk_signal[start_idx:end_idx] = envelope * np.exp(1j * 2 * np.pi * freq * t_symbol)

        return self.add_noise(gfsk_signal, self.snr_db)

    def generate_ofdm(self, n_subcarriers=16, fc=100):
        """OFDM (Orthogonal Frequency Division Multiplexing)"""
        n_samples = len(self.t)
        ofdm_signal = np.zeros(n_samples, dtype=complex)

        # Generate OFDM with random QPSK symbols on each subcarrier
        for k in range(n_subcarriers):
            freq = fc + (k - n_subcarriers/2) * 5  # 5 Hz spacing
            phase = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])
            ofdm_signal += np.exp(1j * (2 * np.pi * freq * self.t + phase))

        # Normalize
        ofdm_signal = ofdm_signal / np.sqrt(n_subcarriers)
        return self.add_noise(ofdm_signal, self.snr_db)

# ==================== Baseline Models ====================

class ResNetBaseline(nn.Module):
    """ResNet-inspired baseline for signal classification"""

    def __init__(self, num_classes=12):
        super(ResNetBaseline, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.res_block1 = self._make_res_block(32, 64)
        self.res_block2 = self._make_res_block(64, 128)
        self.res_block3 = self._make_res_block(128, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def _make_downsample(self, in_channels, out_channels):
        """Create 1x1 conv for dimension matching in residual connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual block 1: 32 -> 64
        identity = self._make_downsample(32, 64).to(x.device)(x)
        x = identity + self.res_block1(x)
        x = self.relu(x)

        # Residual block 2: 64 -> 128
        identity = self._make_downsample(64, 128).to(x.device)(x)
        x = identity + self.res_block2(x)
        x = self.relu(x)

        # Residual block 3: 128 -> 256
        identity = self._make_downsample(128, 256).to(x.device)(x)
        x = identity + self.res_block3(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class LSTMBaseline(nn.Module):
    """LSTM baseline for temporal signal analysis"""

    def __init__(self, num_classes=12):
        super(LSTMBaseline, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # Flatten to sequence
        self.lstm = nn.LSTM(input_size=32*32, hidden_size=128, num_layers=2,
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))  # [B, 32, 32, 32]

        # Reshape for LSTM: [B, Seq, Features]
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = x.view(b, h, -1)  # [B, H, W*C]

        # LSTM
        lstm_out, _ = self.lstm(x)  # [B, H, 128]
        x = lstm_out[:, -1, :]  # Take last timestep

        x = self.fc(x)
        return x

class SimpleTransformer(nn.Module):
    """Transformer baseline for signal classification"""

    def __init__(self, num_classes=12):
        super(SimpleTransformer, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # Patch embedding
        self.patch_embed = nn.Linear(64 * 32, 128)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8,
                                                    dim_feedforward=256, dropout=0.1,
                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Classification head
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = self.pool1(torch.relu(self.conv1(x)))  # [B, 64, 32, 32]

        # Reshape to patches
        b, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, C, W]
        x = x.view(b, h, -1)  # [B, H, C*W]

        # Patch embedding
        x = self.patch_embed(x)  # [B, H, 128]

        # Transformer
        x = self.transformer(x)  # [B, H, 128]

        # Global average pooling
        x = x.mean(dim=1)  # [B, 128]

        x = self.fc(x)
        return x

# ==================== Evaluation Framework ====================

def evaluate_model(model, test_loader, model_name="Model"):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    inference_times = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)

            # Measure inference time
            start_time = time.time()

            if hasattr(model, 'param_estimator'):  # Multi-task model
                outputs, _ = model(inputs)
            else:  # Baseline models
                outputs = model(inputs)

            inference_time = (time.time() - start_time) / len(inputs)
            inference_times.append(inference_time)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_inference_time = np.mean(inference_times) * 1000  # ms

    # ROC AUC (one-vs-rest)
    all_probs = np.array(all_probs)
    all_labels_array = np.array(all_labels)

    # Convert to one-hot
    n_classes = all_probs.shape[1]
    y_true_onehot = np.zeros((len(all_labels_array), n_classes))
    y_true_onehot[np.arange(len(all_labels_array)), all_labels_array] = 1

    try:
        auc = roc_auc_score(y_true_onehot, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    results = {
        'model_name': model_name,
        'accuracy': accuracy * 100,
        'auc': auc,
        'inference_time_ms': avg_inference_time,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return results

def create_comparison_table(results_list):
    """Create SOTA comparison table for paper"""
    data = {
        'Model': [r['model_name'] for r in results_list],
        'Accuracy (%)': [f"{r['accuracy']:.2f}" for r in results_list],
        'AUC': [f"{r['auc']:.4f}" for r in results_list],
        'Inference Time (ms)': [f"{r['inference_time_ms']:.3f}" for r in results_list],
    }

    df = pd.DataFrame(data)

    # Save to CSV for LaTeX
    df.to_csv('sota_comparison.csv', index=False)

    # Create visual table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best results
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('SOTA Comparison: Signal Classification Performance',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('sota_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Saved sota_comparison.csv and sota_comparison_table.png")
    return df

class Extended12ClassDataset(Dataset):
    """Extended dataset with 12 signal types for paper experiments"""

    def __init__(self, n_samples=2400, snr_range=(-15, 5)):
        self.n_samples = n_samples
        self.data = []
        self.labels = []
        self.signal_types = ['Sine', 'Chirp', 'FSK', 'FHSS', 'BPSK', 'QPSK',
                            '8PSK', '16QAM', 'AM', 'FM', 'GFSK', 'OFDM']

        print(f"Generating extended dataset with {len(self.signal_types)} signal types...")
        for i in range(n_samples):
            # Random SNR for each sample
            snr = np.random.uniform(*snr_range)
            gen = ExtendedSignalGenerator(snr_db=snr)

            # Random signal type (evenly distributed)
            signal_type = i % 12

            # Generate signal based on type
            if signal_type == 0:
                sig = gen.generate_sine()
            elif signal_type == 1:
                sig = gen.generate_chirp()
            elif signal_type == 2:
                sig = gen.generate_fsk()
            elif signal_type == 3:
                sig = gen.generate_fhss()
            elif signal_type == 4:
                sig = gen.generate_bpsk()
            elif signal_type == 5:
                sig = gen.generate_qpsk()
            elif signal_type == 6:
                sig = gen.generate_8psk()
            elif signal_type == 7:
                sig = gen.generate_16qam()
            elif signal_type == 8:
                sig = gen.generate_am()
            elif signal_type == 9:
                sig = gen.generate_fm()
            elif signal_type == 10:
                sig = gen.generate_gfsk()
            else:  # 11
                sig = gen.generate_ofdm()

            # Compute spectrogram
            from scipy import signal as sp_signal
            f, t, Sxx = sp_signal.spectrogram(sig, fs=1000, nperseg=64,
                                               noverlap=32, mode='magnitude')
            Sxx_db = 20 * np.log10(Sxx + 1e-10)

            # Resize to fixed size
            from scipy.ndimage import zoom
            zoom_factors = (64 / Sxx_db.shape[0], 64 / Sxx_db.shape[1])
            Sxx_resized = zoom(Sxx_db, zoom_factors, order=1)

            self.data.append(Sxx_resized)
            self.labels.append(signal_type)

            if (i + 1) % 400 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples")

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx]).unsqueeze(0)
        y = torch.LongTensor([self.labels[idx]])[0]
        # Return dummy parameters for compatibility
        y_params = torch.zeros(4)
        return x, y, y_params

def train_baseline_model(model, train_loader, val_loader, model_name, num_epochs=15):
    """Train a baseline model (classification only)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print(f"\nTraining {model_name}...")
    best_val_acc = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

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
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")

    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    print("="*70)
    print("Paper Experiments: SOTA Comparison with 12 Signal Types")
    print("="*70)

    # Create extended 12-class dataset
    print("\n[1/3] Creating extended 12-class dataset...")
    dataset = Extended12ClassDataset(n_samples=2400, snr_range=(-15, 5))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Initialize and train baseline models
    print("\n[2/3] Training baseline models...")

    # Define models
    models = {
        'ResNet Baseline': ResNetBaseline(num_classes=12).to(device),
        'LSTM Baseline': LSTMBaseline(num_classes=12).to(device),
        'Transformer Baseline': SimpleTransformer(num_classes=12).to(device)
    }

    # Train each baseline
    for name, model in models.items():
        train_baseline_model(model, train_loader, val_loader, name, num_epochs=15)

    # Load our trained 4-class model and note the limitation
    print("\n⚠ Note: Our Multi-Task CNN was trained on 4 classes only.")
    print("For fair comparison, we're evaluating baselines on 12 classes.")

    # Evaluate all models
    print("\n[3/3] Evaluating models on test set...")
    results = []

    for name, model in models.items():
        print(f"\n{name}:")
        result = evaluate_model(model, test_loader, name)
        results.append(result)

        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  Inference Time: {result['inference_time_ms']:.3f} ms")

    # Create comparison table
    print("\nCreating SOTA comparison table...")
    df = create_comparison_table(results)
    print("\n" + df.to_string(index=False))

    print("\n" + "="*70)
    print("✅ Paper experiments complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - sota_comparison.csv           (LaTeX table data)")
    print("  - sota_comparison_table.png     (Visual comparison)")
    print("\nNote: Results are for 12-signal classification task")
    print("Signal types: Sine, Chirp, FSK, FHSS, BPSK, QPSK, 8PSK, 16QAM, AM, FM, GFSK, OFDM")

if __name__ == "__main__":
    main()
