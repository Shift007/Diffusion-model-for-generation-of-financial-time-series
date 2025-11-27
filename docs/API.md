# API Documentation

## Preprocessing Module

### WaveletDetrendingPipeline

Main class for wavelet-based preprocessing with detrending.

```python
from src.preprocessing import WaveletDetrendingPipeline

pipeline = WaveletDetrendingPipeline(
    window_length=64,      # Time series window size (power of 2)
    wavelet_type='haar',   # Wavelet family
    overlap=0              # Window overlap (0 = no overlap)
)
```

#### Methods

##### `transform(data, normalize=True)`

Transform time series data into wavelet coefficients.

**Parameters:**
- `data` (array-like): Input time series data, shape (n_samples,)
- `normalize` (bool): Whether to apply percentile normalization

**Returns:**
- `wavelet_images` (ndarray): Wavelet coefficients, shape (n_windows, n_levels, window_length, 1)
- `params` (dict): Parameters for inverse transform
  - `detrending_means`: Window means for reconstruction
  - `normalization_params`: p5, p95 percentiles

**Example:**
```python
import numpy as np
volume_data = np.log1p(raw_volumes)
wavelet_images, params = pipeline.transform(volume_data)
```

##### `inverse_transform(wavelet_coeffs, params)`

Reconstruct time series from wavelet coefficients.

**Parameters:**
- `wavelet_coeffs` (ndarray): Wavelet coefficients
- `params` (dict): Parameters from transform step

**Returns:**
- `reconstructed_series` (ndarray): Reconstructed time series

**Example:**
```python
reconstructed = pipeline.inverse_transform(synthetic_wavelets, params)
```

---

## Diffusion Module

### Diffusion

DDPM (Denoising Diffusion Probabilistic Model) implementation.

```python
from src.diffusion import Diffusion, UNet

model = UNet(in_channels=1)
diffusion = Diffusion(
    model=model,
    timesteps=1000,        # Number of diffusion steps
    beta_schedule='cosine', # Noise schedule ('linear' or 'cosine')
    device='cuda'          # Device for training
)
```

#### Methods

##### `p_losses(x_start, t, loss_type='huber')`

Compute training loss for given timestep.

**Parameters:**
- `x_start` (Tensor): Original images, shape (batch, channels, height, width)
- `t` (Tensor): Timesteps, shape (batch,)
- `loss_type` (str): Loss function ('huber', 'l1', or 'l2')

**Returns:**
- `loss` (Tensor): Computed loss value

##### `sample(shape, temperature=1.0)`

Generate synthetic samples.

**Parameters:**
- `shape` (tuple): Output shape (n_samples, channels, height, width)
- `temperature` (float): Sampling temperature (>1 = more diversity)

**Returns:**
- `samples` (Tensor): Generated samples

**Example:**
```python
synthetic = diffusion.sample((1000, 1, 6, 64), temperature=1.5)
```

---

### UNet

U-Net architecture for diffusion model.

```python
from src.diffusion import UNet

model = UNet(
    in_channels=1,     # Input channels (1 for single-channel wavelets)
    base_channels=64,  # Base channel count
    time_emb_dim=256   # Time embedding dimension
)
```

**Input Shape:** (batch, channels, height, width)
**Output Shape:** Same as input

---

## Utility Functions

### visualize_detrending_effect

Visualize the effect of detrending on wavelet coefficients.

```python
from src.preprocessing import visualize_detrending_effect

visualize_detrending_effect(
    original_data,      # Original time series
    detrended_data,     # Detrended time series
    window_means        # Removed means
)
```

Creates comparison plots showing:
- Original vs detrended time series
- Wavelet coefficient distributions
- DC component histograms
