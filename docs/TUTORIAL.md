# Tutorial: Generating Synthetic Financial Volumes

This tutorial walks through the complete pipeline for generating synthetic financial volume data.

## Step 1: Data Acquisition

First, download historical volume data using yfinance:

```python
import yfinance as yf
import numpy as np

# Download data
tickers = ['GOOG', 'AAPL', 'META', 'MSFT', 'AMZN']
start_date = '2013-01-01'
end_date = '2024-12-31'

all_volumes = []
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    log_volume = np.log1p(data['Volume'].values)
    all_volumes.append(log_volume)

# Concatenate all series
volume_data = np.concatenate(all_volumes)
print(f"Total data points: {len(volume_data)}")
```

## Step 2: Create Time Windows

Split the continuous time series into fixed-length windows:

```python
WINDOW_LENGTH = 64  # Must be power of 2

windows = []
for i in range(len(volume_data) - WINDOW_LENGTH + 1):
    window = volume_data[i:i + WINDOW_LENGTH]
    windows.append(window)

volume_series = np.array(windows)
print(f"Created {len(volume_series)} windows of length {WINDOW_LENGTH}")
```

## Step 3: Wavelet Preprocessing

Transform to wavelet space with detrending:

```python
from src.preprocessing import WaveletDetrendingPipeline
import pywt

# Initialize pipeline
pipeline = WaveletDetrendingPipeline(window_length=WINDOW_LENGTH)

# Apply detrending
window_means = volume_series.mean(axis=1, keepdims=True)
volume_detrended = volume_series - window_means

# Apply SWT
wavelet_type = 'haar'
wavelet_level = int(np.log2(WINDOW_LENGTH))

wavelet_images = []
for ts in volume_detrended:
    coeffs = pywt.swt(ts, wavelet_type, level=wavelet_level, trim_approx=True)
    wavelet_images.append(np.stack(coeffs)[::-1])

wavelet_images = np.stack(wavelet_images)

# Normalize
p5 = np.percentile(wavelet_images, 5)
p95 = np.percentile(wavelet_images, 95)
wavelet_normalized = (wavelet_images - p5) / (p95 - p5 + 1e-8)
wavelet_normalized = np.clip(wavelet_normalized, 0, 1)

# Add channel dimension
wavelet_normalized = wavelet_normalized[..., np.newaxis]

print(f"Wavelet images shape: {wavelet_normalized.shape}")
```

## Step 4: Train Diffusion Model

Train the diffusion model on wavelet coefficients:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from src.diffusion import UNet, Diffusion
from tqdm import tqdm

# Dataset
class WaveletDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        img = img * 2.0 - 1.0  # Scale to [-1, 1]
        return torch.from_numpy(img.copy()).float()

# Create dataloader
dataset = WaveletDataset(wavelet_normalized)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1).to(device)
diffusion = Diffusion(model, timesteps=1000, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
EPOCHS = 30
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch in pbar:
        batch = batch.to(device)
        t = torch.randint(0, 1000, (batch.shape[0],), device=device)
        
        loss = diffusion.p_losses(batch, t, loss_type='huber')
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Save checkpoint
    torch.save({
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch + 1
    }, f'checkpoint_epoch_{epoch+1}.pt')
```

## Step 5: Generate Synthetic Data

Generate new synthetic samples:

```python
# Sample from model
model.eval()
H, W = wavelet_normalized.shape[1], wavelet_normalized.shape[2]
synthetic_wavelets = diffusion.sample((1000, 1, H, W), temperature=1.5)

# Convert back to [0, 1]
synthetic_wavelets = (synthetic_wavelets.clamp(-1, 1) + 1.0) / 2.0
synthetic_wavelets = synthetic_wavelets.cpu().numpy()
synthetic_wavelets = np.transpose(synthetic_wavelets, (0, 2, 3, 1))

print(f"Generated {len(synthetic_wavelets)} synthetic samples")
```

## Step 6: Reconstruct Time Series

Convert wavelets back to log volumes:

```python
# Denormalize
synthetic_denorm = synthetic_wavelets.squeeze(-1) * (p95 - p5 + 1e-8) + p5

# Inverse SWT
reconstructed = []
for coeffs in synthetic_denorm:
    # Reverse order and reconstruct
    coeffs_reversed = coeffs[::-1]
    reconstruction = np.zeros(WINDOW_LENGTH)
    
    for level_idx, detail in enumerate(coeffs_reversed):
        approx = reconstruction if level_idx > 0 else np.zeros_like(detail)
        reconstruction = pywt.idwt(approx, detail, wavelet_type)[:WINDOW_LENGTH]
    
    reconstructed.append(reconstruction)

reconstructed = np.array(reconstructed)

# Add back means (sample from training distribution)
sampled_means = np.random.choice(window_means.flatten(), size=len(reconstructed))
synthetic_log_volumes = reconstructed + sampled_means[:, np.newaxis]

print(f"Reconstructed shape: {synthetic_log_volumes.shape}")
```

## Step 7: Validation

Compare original and synthetic data:

```python
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Flatten series
original_flat = volume_series.flatten()
synthetic_flat = synthetic_log_volumes.flatten()

# Compute ACF
original_acf = acf(original_flat, nlags=60, fft=True)
synthetic_acf = acf(synthetic_flat, nlags=60, fft=True)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF comparison
axes[0].plot(original_acf, 'b-o', label='Original', alpha=0.7)
axes[0].plot(synthetic_acf, 'r-s', label='Synthetic', alpha=0.7)
axes[0].set_title('Autocorrelation Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution comparison
axes[1].hist(original_flat, bins=50, alpha=0.7, label='Original', density=True)
axes[1].hist(synthetic_flat, bins=50, alpha=0.7, label='Synthetic', density=True)
axes[1].set_title('Distribution Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print(f"Original - Mean: {original_flat.mean():.4f}, Std: {original_flat.std():.4f}")
print(f"Synthetic - Mean: {synthetic_flat.mean():.4f}, Std: {synthetic_flat.std():.4f}")
print(f"ACF Correlation: {np.corrcoef(original_acf, synthetic_acf)[0,1]:.4f}")
```

## Next Steps

- Experiment with different window lengths (32, 64, 128)
- Try different wavelet families ('haar', 'db4', 'sym4')
- Adjust sampling temperature for more/less diversity
- Fine-tune model architecture and training parameters
- Validate on out-of-sample tickers
