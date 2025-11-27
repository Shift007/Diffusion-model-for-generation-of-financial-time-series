# Financial Time Series Synthesis with Diffusion Models

A deep learning framework for generating synthetic financial volume data using wavelet-based diffusion models. This project preserves key statistical properties including volatility clustering and autocorrelation patterns.

## 🎯 Overview

This repository implements a novel approach to synthetic financial data generation:
1. **Wavelet Transform**: Decomposes volume time series using Stationary Wavelet Transform (SWT)
2. **Detrending**: Removes DC components to prevent mode collapse
3. **Diffusion Model**: Learns patterns in wavelet space using a U-Net architecture
4. **Reconstruction**: Generates realistic synthetic volumes with preserved temporal dependencies

## 📁 Project Structure

```
.
├── src/
│   ├── diffusion/
│   │   ├── diffusion.py      # DDPM implementation
│   │   ├── unet.py            # U-Net architecture
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── wavelet_detrending.py  # Wavelet preprocessing pipeline
│   └── utils/
│       └── __init__.py
├── notebooks/
│   ├── volume_synthesis.ipynb      # Main pipeline notebook
│   └── experiments/
│       └── cosine_validation.ipynb # Synthetic validation
├── tests/
│   └── test_detrending.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-timeseries-synthesis.git
cd financial-timeseries-synthesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from src.diffusion import Diffusion, UNet
from src.preprocessing.wavelet_detrending import WaveletDetrendingPipeline

# Initialize pipeline
pipeline = WaveletDetrendingPipeline(window_length=64)

# Preprocess data
wavelet_images, params = pipeline.transform(volume_data)

# Train diffusion model
model = UNet(in_channels=1)
diffusion = Diffusion(model, timesteps=1000)

# Generate synthetic data
synthetic_wavelets = diffusion.sample(num_samples=1000)
synthetic_volumes = pipeline.inverse_transform(synthetic_wavelets, params)
```

## 📊 Features

- **Volatility Clustering Preservation**: Captures GARCH-like effects in financial data
- **Autocorrelation Matching**: Maintains temporal dependencies
- **Wavelet-Based Processing**: Robust multi-scale decomposition
- **Detrending Pipeline**: Prevents DC component mode collapse
- **Configurable Architecture**: Flexible model and preprocessing parameters

## 🧪 Validation

The framework includes comprehensive validation:
- ACF (Autocorrelation Function) comparison
- Distribution matching (KS test)
- Volatility clustering metrics
- Visual inspection tools

## 📝 Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `WINDOW_LENGTH` | Time series window size | 64 |
| `BATCH_SIZE` | Training batch size | 64 |
| `EPOCHS` | Training epochs | 30 |
| `TIMESTEPS` | Diffusion timesteps | 1000 |
| `TEMPERATURE` | Sampling temperature | 1.5 |

## 🔬 Research Background

This work addresses key challenges in synthetic financial data generation:
- **Mode Collapse**: Resolved through wavelet detrending
- **Temporal Patterns**: Preserved via diffusion in frequency domain
- **Scale Invariance**: Maintained through log transformation and percentile normalization

## 📚 Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- PyWavelets
- yfinance (for data download)
- statsmodels (for ACF analysis)

See `requirements.txt` for complete dependencies.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration inquiries, please open an issue on GitHub.

## 🙏 Acknowledgments

- Wavelet preprocessing inspired by financial signal processing literature
- Diffusion model architecture based on DDPM (Denoising Diffusion Probabilistic Models)
- Financial data courtesy of Yahoo Finance API

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{financial_timeseries_synthesis,
  title={Financial Time Series Synthesis with Diffusion Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/financial-timeseries-synthesis}
}
```
