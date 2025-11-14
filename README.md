# Neural Receiver for Weak Signal Detection and Parametric Classification

A Multi-Task IQ-Based Neural Receiver for Weak-Signal Detection and Parametric Classification

This project implements a deep learning-based system for detecting and classifying various types of SIGINT (Signals Intelligence) signals under low SNR conditions.

## Features

- **Signal Generation**: Synthetic generation of multiple signal types:
  - Sine waves
  - Chirp signals
  - FSK (Frequency Shift Keying)
  - FHSS (Frequency Hopping Spread Spectrum)

- **Spectrogram Analysis**: Time-frequency representation of signals for feature extraction

- **Neural Network Classification**: CNN-based classifier for robust signal identification

- **Performance Evaluation**: Comprehensive metrics including confusion matrices and accuracy plots

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

Open `run_simulation.ipynb` in Google Colab for a ready-to-run notebook with GPU support.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/blob/main/run_simulation.ipynb)

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

1. **sample_spectrograms.png** - Visual representation of different signal types
2. **training_results.png** - Training and validation loss/accuracy curves
3. **confusion_matrix.png** - Classification performance matrix
4. **sigint_detector_model.pth** - Trained PyTorch model weights

## Signal Types

### 1. Sine Wave
Pure sinusoidal signal at a fixed frequency.

### 2. Chirp
Linear frequency-modulated signal with frequency sweeping from f0 to f1.

### 3. FSK (Frequency Shift Keying)
Digital modulation where binary data is transmitted by switching between two frequencies.

### 4. FHSS (Frequency Hopping Spread Spectrum)
Signal that rapidly switches carrier frequencies following a pseudo-random pattern.

## Model Architecture

The classifier uses a CNN architecture with:
- 3 convolutional blocks (32, 64, 128 filters)
- Batch normalization and ReLU activation
- Max pooling for spatial downsampling
- Fully connected layers with dropout for classification

## Performance

The model is trained on 2000 synthetic samples with SNR ranging from -15 dB to 5 dB, achieving robust classification performance across all signal types.

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
