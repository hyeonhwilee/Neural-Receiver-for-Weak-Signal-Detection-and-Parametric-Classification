# Neural Receiver for Weak Signal Detection and Parametric Classification

A Multi-Task IQ-Based Neural Receiver for Weak-Signal Detection and Parametric Classification

This project implements a deep learning-based system for detecting and classifying various types of SIGINT (Signals Intelligence) signals under low SNR conditions.

## Features

- **Signal Generation**: Synthetic generation of 12 signal types:
  - **Basic Signals**: Sine waves, Chirp (Linear FM)
  - **FSK Family**: FSK, GFSK, FHSS
  - **PSK Family**: BPSK, QPSK, 8PSK
  - **QAM**: 16QAM
  - **Analog Modulation**: AM, FM, OFDM

- **Multi-Task Learning**: Simultaneous signal classification and parameter estimation
  - Center frequency, bandwidth, signal power, SNR

- **Spectrogram Analysis**: Time-frequency representation using STFT for feature extraction

- **Neural Network Models**:
  - Multi-Task CNN with shared feature extractor
  - SOTA baselines: ResNet, LSTM, Transformer

- **Performance Evaluation**:
  - Comprehensive metrics (Accuracy, ROC AUC, Inference Time)
  - Confusion matrices and parameter estimation errors
  - SOTA comparison tables for paper-ready results

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, SciPy, Matplotlib, Seaborn, scikit-learn

See `requirements.txt` for full dependencies.

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

**Option 1: Use Notebook** (Recommended for beginners)

Open `run_simulation.ipynb` in Google Colab for a ready-to-run notebook with GPU support.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/blob/claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF/run_simulation.ipynb)

**Option 2: Quick Run** (Direct execution)

Create a new notebook in Google Colab and paste the following code:

```python
# Check GPU and clone repository
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
import os, shutil
os.chdir('/content')
if os.path.exists('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification'):
    shutil.rmtree('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')
!git clone -b claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
os.chdir('/content/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')

# Install dependencies and run simulation
!pip install -q numpy scipy matplotlib torch torchvision scikit-learn pandas seaborn
!python sigint_simulation.py

# Display results
from IPython.display import Image, display
display(Image('sample_spectrograms.png'))
display(Image('signal_parameters.png'))
display(Image('training_results.png'))
display(Image('confusion_matrix.png'))
```

📖 **Detailed Guide**: [COLAB_QUICK_START.md](./COLAB_QUICK_START.md)

**Note**:
- Current branch: `claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF`
- After merging to main: use `main` branch instead

### Paper Experiments (12-Signal SOTA Comparison)

For comprehensive experiments with 12 signal types and SOTA baseline comparisons:

**Colab Notebook**: Open `run_paper_experiments.ipynb` in Google Colab

This runs experiments comparing:
- ResNet baseline
- LSTM baseline
- Transformer baseline

On 12 signal types: Sine, Chirp, FSK, FHSS, BPSK, QPSK, 8PSK, 16QAM, AM, FM, GFSK, OFDM

**Command line**:
```bash
python paper_experiments.py
```

Generates:
- `sota_comparison.csv` - LaTeX-ready comparison table
- `sota_comparison_table.png` - Visual comparison chart

## Usage

### Command Line

```bash
# Run the full simulation
python sigint_simulation.py

# Or use the runner script
python run_simulation.py
```

### Jupyter Notebook

Open and run `run_simulation.ipynb` in Jupyter or Google Colab.

## Output

The simulation generates the following outputs:

1. **sample_spectrograms.png** - Visual representation of different signal types (time domain, frequency spectrum, spectrogram)
2. **signal_parameters.png** - Comprehensive signal parameter analysis table including:
   - Center frequency and peak frequency
   - Signal power (dBm)
   - Bandwidth (99% and 3dB)
   - Modulation type
   - Signal onset time
   - Estimated SNR
3. **training_results.png** - Training and validation loss/accuracy curves
4. **confusion_matrix.png** - Classification performance matrix
5. **sigint_detector_model.pth** - Trained PyTorch model weights

## Signal Types

The system supports 12 different signal modulation types:

### Basic Signals
1. **Sine Wave** - Pure sinusoidal signal at a fixed frequency (CW)
2. **Chirp** - Linear frequency-modulated signal sweeping from f0 to f1

### Frequency Shift Keying (FSK) Family
3. **FSK** - Binary frequency shift keying
4. **GFSK** - Gaussian frequency shift keying with pulse shaping
5. **FHSS** - Frequency hopping spread spectrum with pseudo-random pattern

### Phase Shift Keying (PSK) Family
6. **BPSK** - Binary phase shift keying (2 phases)
7. **QPSK** - Quadrature phase shift keying (4 phases)
8. **8PSK** - 8-ary phase shift keying (8 phases)

### Quadrature Amplitude Modulation (QAM)
9. **16QAM** - 16-ary quadrature amplitude modulation

### Analog Modulation
10. **AM** - Amplitude modulation
11. **FM** - Frequency modulation

### Multi-Carrier
12. **OFDM** - Orthogonal frequency division multiplexing

## Model Architecture

### Multi-Task CNN (Our Approach)
- **Shared Feature Extractor**: 3 convolutional blocks (32, 64, 128 filters)
- **Batch Normalization** and ReLU activation
- **Max Pooling** for spatial downsampling
- **Dual Heads**:
  - Classification head: Signal type identification
  - Parameter regression head: Fc, BW, Power, SNR estimation
- **Multi-task Loss**: CrossEntropy + 0.5 × MSE

### SOTA Baselines
- **ResNet**: Residual blocks with skip connections
- **LSTM**: CNN features + LSTM for temporal analysis
- **Transformer**: Self-attention mechanism with patch embeddings

## Performance

### 4-Signal Multi-Task Model
- **Dataset**: 2000 samples, SNR range: -15 to 5 dB
- **Classification Accuracy**: ~95%
- **Parameter Estimation (MAE)**:
  - Center Frequency: 3.66 Hz
  - Bandwidth: 7.68 Hz
  - SNR: 1.30 dB

### 12-Signal SOTA Comparison
Comprehensive evaluation on extended dataset with 12 signal types. Run `paper_experiments.py` to generate comparison tables with accuracy, AUC, and inference time metrics for multiple baseline models.

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{neural-sigint-receiver,
  title={Neural Receiver for Weak Signal Detection and Parametric Classification},
  author={Hyeonhwi Lee},
  year={2024},
  publisher={GitHub},
  url={https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
