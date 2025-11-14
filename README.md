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

**방법 1: 노트북 사용** (초보자 추천)

Open `run_simulation.ipynb` in Google Colab for a ready-to-run notebook with GPU support.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/blob/claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF/run_simulation.ipynb)

**방법 2: 직접 실행** (빠른 실행)

Google Colab에서 새 노트북을 만들고 아래 코드를 셀에 붙여넣어 실행:

```python
# GPU 확인 및 저장소 클론
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
import os, shutil
os.chdir('/content')
if os.path.exists('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification'):
    shutil.rmtree('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')
!git clone -b claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
os.chdir('/content/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')

# 라이브러리 설치 및 시뮬레이션 실행
!pip install -q numpy scipy matplotlib torch torchvision scikit-learn pandas seaborn
!python sigint_simulation.py

# 결과 표시
from IPython.display import Image, display
display(Image('sample_spectrograms.png'))
display(Image('training_results.png'))
display(Image('confusion_matrix.png'))
```

📖 **상세 가이드**: [COLAB_QUICK_START.md](./COLAB_QUICK_START.md)

**참고**:
- 현재 작업 브랜치: `claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF`
- 메인 브랜치에 merge된 후: `main`으로 브랜치명 변경

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
