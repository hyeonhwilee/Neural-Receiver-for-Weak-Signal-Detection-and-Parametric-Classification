# Quick Start Guide

This guide will help you get started with the Neural Receiver project in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Train Your First Model

```bash
# Train with default configuration (runs for 100 epochs)
python train.py

# Or train with custom config
python train.py --config configs/custom_config.yaml
```

The training will:
- Generate synthetic IQ data with various modulation types
- Train a multi-task neural network
- Save checkpoints in `experiments/neural_receiver_default/checkpoints/`
- Log metrics to TensorBoard

**Monitor training:**
```bash
tensorboard --logdir experiments/neural_receiver_default/logs
```

## Evaluate the Model

```bash
# Evaluate the best model
python evaluate.py --checkpoint experiments/neural_receiver_default/checkpoints/best_model.pt

# Results will be saved to 'evaluation_results/' including:
# - ROC curves
# - Confusion matrices
# - Regression error plots
# - Detailed metrics (JSON)
```

## Run Inference

```bash
# Create sample IQ data
python -c "
import numpy as np
from src.data import SignalGenerator, SignalParams

generator = SignalGenerator(seed=42)
params = SignalParams(
    center_freq=0.1, bandwidth=0.05, power=1.0,
    snr_db=-5.0, symbol_rate=200, modulation_type='AM'
)
signal, noisy = generator.generate_signal(1024, params)
np.save('sample_iq.npy', noisy)
print('Created sample_iq.npy')
"

# Run inference
python inference.py \
    --checkpoint experiments/neural_receiver_default/checkpoints/best_model.pt \
    --input sample_iq.npy
```

## Explore with Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Open notebooks in order:
# 1. 01_signal_generation.ipynb - Learn about signal generation
# 2. 02_training_demo.ipynb - Train a model interactively
# 3. 03_evaluation_demo.ipynb - Comprehensive evaluation
```

## Understanding the Output

### Training Output
- `experiments/[experiment_name]/checkpoints/` - Model checkpoints
- `experiments/[experiment_name]/logs/` - TensorBoard logs
- `experiments/[experiment_name]/training_history.json` - Training metrics

### Evaluation Output
- `evaluation_results/detection_metrics.png` - ROC and Precision-Recall curves
- `evaluation_results/confusion_matrix.png` - Modulation classification confusion matrix
- `evaluation_results/regression_errors.png` - Parameter estimation errors
- `evaluation_results/metrics.json` - Detailed metrics

### Inference Output
```
Signal Detected:    True
Detection Prob:     0.9234
Modulation Type:    AM (0.8765)
Center Frequency:   0.1023
Bandwidth:          0.0487
Estimated SNR:      -4.87 dB
Symbol Rate:        198.45
```

## Customization

### Modify Configuration

Edit `configs/default_config.yaml`:

```yaml
# Adjust model architecture
model:
  base_channels: 128  # Increase for more capacity
  num_blocks: 6       # Deeper network

# Adjust training
training:
  num_epochs: 200
  batch_size: 64
  learning_rate: 0.0005

# Adjust data
data:
  snr_range: [-15, 5]  # Wider SNR range
  train_samples: 50000  # More training data
```

### Add Custom Modulation Types

Edit `src/data/signal_generator.py` and add your generator function:

```python
def generate_my_modulation(self, n_samples: int, params: SignalParams) -> np.ndarray:
    # Your modulation logic here
    pass
```

## Performance Tuning

### For Faster Training
- Increase `batch_size` (if you have GPU memory)
- Set `data.pregenerate: true` (uses more RAM but faster)
- Reduce `data.train_samples` for quick experiments

### For Better Performance
- Increase `training.num_epochs` to 200+
- Increase `data.train_samples` to 50000+
- Adjust `model.base_channels` and `model.num_blocks`
- Enable uncertainty weighting: `training.use_uncertainty_weighting: true`

## Common Issues

### CUDA Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 16  # or 8
```

### Training Too Slow
```yaml
# Use fewer samples for quick experiments
data:
  train_samples: 5000
  val_samples: 1000
```

### Poor Performance at Low SNR
- Train with wider SNR range: `snr_range: [-15, 5]`
- Increase model capacity: `base_channels: 128`
- Add more training data

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore the Jupyter notebooks for interactive examples
3. Modify the configuration to suit your needs
4. Experiment with different architectures and hyperparameters
5. Try on your own IQ data!

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review the example notebooks
- Open an issue on GitHub for bugs or questions

## Citation

If you use this in your research, please cite:

```bibtex
@software{neural_receiver_2025,
  title={Neural Receiver for Weak Signal Detection and Parametric Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification}
}
```
