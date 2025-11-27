# Examples

## Quick Start Example

```python
from src.diffusion import UNet, Diffusion
from src.preprocessing import WaveletDetrendingPipeline
import numpy as np
import yfinance as yf

# 1. Download data
data = yf.download('GOOG', start='2020-01-01', end='2024-12-31')
log_volumes = np.log1p(data['Volume'].values)

# 2. Preprocess
pipeline = WaveletDetrendingPipeline(window_length=64)
wavelet_images, params = pipeline.transform(log_volumes)

# 3. Train (simplified - see TUTORIAL.md for full training loop)
model = UNet(in_channels=1)
diffusion = Diffusion(model, timesteps=1000)
# ... training code ...

# 4. Generate
synthetic_wavelets = diffusion.sample((100, 1, 6, 64))

# 5. Reconstruct
synthetic_volumes = pipeline.inverse_transform(synthetic_wavelets, params)
```

## Multi-Ticker Training

```python
import yfinance as yf
import numpy as np

tickers = ['GOOG', 'AAPL', 'META', 'MSFT', 'AMZN']
all_data = []

for ticker in tickers:
    data = yf.download(ticker, start='2013-01-01', end='2024-12-31')
    log_vol = np.log1p(data['Volume'].values)
    all_data.append(log_vol)

# Concatenate and create windows
combined_data = np.concatenate(all_data)
# ... proceed with preprocessing ...
```

## Custom Window Length

```python
# Use 128 instead of default 64
pipeline = WaveletDetrendingPipeline(window_length=128)

# Note: Wavelet level automatically adjusts
# 64 → level 6
# 128 → level 7
# 256 → level 8
```

## Temperature Sampling

```python
# Low temperature = less diversity, closer to training data
conservative_samples = diffusion.sample(shape, temperature=0.8)

# High temperature = more diversity, more exploration
diverse_samples = diffusion.sample(shape, temperature=2.0)
```

## ACF Validation

```python
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def validate_acf(original, synthetic, max_lag=60):
    """Compare autocorrelation functions"""
    orig_acf = acf(original.flatten(), nlags=max_lag, fft=True)
    synth_acf = acf(synthetic.flatten(), nlags=max_lag, fft=True)
    
    plt.figure(figsize=(12, 5))
    plt.plot(orig_acf, 'b-o', label='Original', alpha=0.7)
    plt.plot(synth_acf, 'r-s', label='Synthetic', alpha=0.7)
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title('Autocorrelation Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    correlation = np.corrcoef(orig_acf, synth_acf)[0, 1]
    print(f"ACF Correlation: {correlation:.4f}")
    return correlation

# Usage
validate_acf(original_log_volumes, synthetic_log_volumes)
```

## Distribution Comparison

```python
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def compare_distributions(original, synthetic):
    """Statistical comparison of distributions"""
    # KS test
    ks_stat, p_value = ks_2samp(original.flatten(), synthetic.flatten())
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.hist(original.flatten(), bins=50, alpha=0.7, 
             label='Original', density=True)
    plt.hist(synthetic.flatten(), bins=50, alpha=0.7, 
             label='Synthetic', density=True)
    plt.xlabel('Log Volume')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Distribution Comparison (KS p-value: {p_value:.4f})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Result: {'Similar' if p_value > 0.05 else 'Different'}")

# Usage
compare_distributions(original_log_volumes, synthetic_log_volumes)
```

## Custom Wavelet Family

```python
# Try different wavelets
wavelets = ['haar', 'db4', 'sym4', 'coif3']

for wav in wavelets:
    pipeline = WaveletDetrendingPipeline(
        window_length=64,
        wavelet_type=wav
    )
    # ... train and compare results ...
```

## Resume Training

```python
import torch

# Save checkpoint during training
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'step': global_step
}
torch.save(checkpoint, 'checkpoint.pt')

# Resume training
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch']
start_step = checkpoint['step']
```

## Batch Generation

```python
# Generate large number of samples in batches
all_samples = []
batch_size = 100
total_samples = 10000

for i in range(0, total_samples, batch_size):
    batch = diffusion.sample((batch_size, 1, H, W))
    all_samples.append(batch.cpu())
    print(f"Generated {i+batch_size}/{total_samples}")

all_samples = torch.cat(all_samples, dim=0)
```
