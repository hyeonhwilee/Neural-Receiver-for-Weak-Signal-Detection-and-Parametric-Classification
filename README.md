# Neural Receiver for Weak Signal Detection and Parametric Classification

A Multi-Task IQ-Based Neural Receiver for Weak-Signal Detection and Parametric Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Problem Definition

This project implements a machine-learning-based receiver capable of detecting and characterizing weak signals (SNR ≤ 0 dB) from single-antenna baseband IQ data. Given a complex IQ sequence `y[n] ∈ ℂ, n=0,...,N-1`, the model performs multi-task inference:

### Tasks

1. **Signal Detection** (Binary Classification)
   - Output: Probability of signal presence `p_sig = Pr[signal present]`

2. **Parameter Estimation** (Regression)
   - Estimated center frequency: `f̂_c`
   - Estimated bandwidth: `B̂`
   - Estimated signal power/SNR: `P̂`
   - Estimated symbol rate: `R̂_s`

3. **Modulation Classification** (Multi-class Classification)
   - 12 modulation types including:
     - Communication signals: No-signal, AM, FM, 2FSK
     - Radar waveforms: CW Radar, Pulsed Radar, LFM/Chirp
     - Advanced MOP types: Phase-Coded Pulse, Frequency-Coded Pulse, Unmodulated Pulse, Pulse-Compression MOP, Other MOP Types

### Objective

Assess whether a multi-task neural model operating directly on IQ-domain data can reliably detect weak signals and accurately estimate their parameters and modulation types under low-SNR conditions, surpassing traditional energy-based or spectral-analysis methods.

## 🏗️ Architecture

The system uses a multi-task learning architecture with:

- **Shared Backbone**: 1D CNN with residual blocks for feature extraction from IQ sequences
- **Detection Head**: Binary classification for signal presence/absence
- **Regression Head**: Multi-output regression for parameter estimation
- **Classification Head**: Multi-class classification for modulation type

```
                    ┌─────────────────────┐
                    │   IQ Input (I, Q)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Shared Backbone    │
                    │  (ResNet-like CNN)  │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
        │ Detection    │ │Regression│ │Classification│
        │ Head         │ │Head      │ │Head          │
        └───────┬──────┘ └────┬─────┘ └─────┬────────┘
                │             │              │
        ┌───────▼──────┐ ┌───▼──────┐ ┌────▼─────────┐
        │p_sig         │ │f̂_c, B̂,  │ │Modulation    │
        │              │ │SNR, R̂_s  │ │Type          │
        └──────────────┘ └──────────┘ └──────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration
python train.py

# Train with custom configuration
python train.py --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint experiments/neural_receiver/checkpoints/best_model.pt
```

### Inference

```bash
# Run inference on IQ data
python inference.py --checkpoint path/to/model.pt --input data/iq_samples.npy
```

## 📊 Dataset

The project includes a synthetic signal generator that creates IQ data with:

- **Modulation Types**: 12 different types including communication and radar signals
- **SNR Range**: Configurable, default -10 to 0 dB (weak signals)
- **Signal Parameters**: Randomized center frequency, bandwidth, power, and symbol rate

### Signal Types Implemented

| Category | Modulation Types |
|----------|-----------------|
| Noise | No-signal (AWGN only) |
| Analog Comm | AM, FM |
| Digital Comm | 2FSK |
| Radar (Basic) | CW Radar, Pulsed Radar |
| Radar (Advanced) | LFM/Chirp, Phase-Coded Pulse, Frequency-Coded Pulse |
| Radar (MOP) | Unmodulated Pulse, Pulse-Compression MOP, Other MOP Types |

## 📈 Performance Metrics

The system evaluates performance using:

### Detection Metrics
- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- Precision-Recall Curve
- Detection accuracy at optimal threshold

### Regression Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Per-parameter error analysis

### Classification Metrics
- Overall accuracy
- F1-score (macro and weighted)
- Confusion matrix
- Per-class precision, recall, F1

## 📁 Project Structure

```
Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/
│
├── src/
│   ├── data/
│   │   ├── signal_generator.py    # Signal generation for all modulation types
│   │   └── dataset.py             # PyTorch dataset for multi-task learning
│   │
│   ├── models/
│   │   └── neural_receiver.py     # Multi-task neural architecture
│   │
│   ├── training/
│   │   └── trainer.py             # Training loop and utilities
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── evaluator.py           # Model evaluation
│   │
│   └── utils/
│       └── visualization.py       # Plotting and visualization
│
├── configs/
│   └── default_config.yaml        # Default configuration
│
├── notebooks/
│   ├── 01_signal_generation.ipynb # Signal generation examples
│   ├── 02_training_demo.ipynb     # Training demonstration
│   └── 03_evaluation_demo.ipynb   # Evaluation and results
│
├── experiments/                    # Training outputs (checkpoints, logs)
├── train.py                       # Main training script
├── evaluate.py                    # Evaluation script
├── inference.py                   # Inference script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

Edit `configs/default_config.yaml` to customize:

- Model architecture (channels, depth, dropout)
- Training parameters (epochs, learning rate, batch size)
- Data generation (SNR range, sequence length, number of samples)
- Loss weights for multi-task learning

## 📚 Usage Examples

### Python API

```python
from src.data import SignalDataset, SignalGenerator
from src.models import NeuralReceiver
import torch

# Generate sample data
generator = SignalGenerator(sample_rate=1.0)
params = generator.generate_random_params(snr_range=(-10, 0))
signal, noisy_signal = generator.generate_signal(1024, params)

# Create model
model = NeuralReceiver(
    input_channels=2,
    base_channels=64,
    num_blocks=4,
    num_classes=12
)

# Inference
iq_tensor = torch.stack([
    torch.from_numpy(noisy_signal.real).float(),
    torch.from_numpy(noisy_signal.imag).float()
]).unsqueeze(0)

predictions = model.predict(iq_tensor)
print(f"Signal detected: {predictions['signal_detected'].item()}")
print(f"Modulation type: {predictions['modulation_class'].item()}")
```

## 🎯 Results

Expected performance on weak signals (SNR ≤ 0 dB):

- **Detection AUC**: > 0.95
- **Classification Accuracy**: > 0.85
- **Parameter Estimation MAE**: < 0.1 (normalized)

Performance improves significantly at higher SNR levels.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_receiver_2025,
  title={Neural Receiver for Weak Signal Detection and Parametric Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification}
}
```

## 🙏 Acknowledgments

- Signal processing techniques inspired by modern SDR and radar systems
- Multi-task learning architecture based on recent deep learning research
- Implementation uses PyTorch for efficient GPU acceleration

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]