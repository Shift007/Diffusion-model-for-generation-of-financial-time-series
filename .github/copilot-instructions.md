# AI Coding Agent Instructions

## Project Overview

This codebase implements **financial time series synthesis using diffusion models** trained on wavelet-transformed data. The system generates synthetic stock volume, log returns, and volatility data that preserve key statistical properties like volatility clustering and autocorrelation patterns.

## Architecture Flow

```
Raw Financial Data (yfinance)
    ↓ Log transform (volume), Compute log returns & volatility
Z-Score Normalization (clip ±6σ)
    ↓ Sliding window (length=256, stride=1)
Stationary Wavelet Transform (SWT, Haar, 8 levels)
    ↓ Percentile normalization (p5-p95) → [0,1]
Single-Channel Wavelet Images (N, 9, 256, 1)
    ↓ Train diffusion model (DDPM + U-Net)
Synthetic Wavelet Images
    ↓ Inverse SWT
Synthetic Time Series (z-score space)
```

**Key Insight**: The model operates in **wavelet frequency domain** to avoid mode collapse and preserve multi-scale temporal patterns.

## Critical Components

### 1. Preprocessing Pipeline (`notebooks/stocks_synthesis.ipynb` cell 2)
- Downloads Close & Volume from Yahoo Finance for multiple tickers
- **Volume**: `log1p(Volume)` → z-score → windows
- **Log Returns**: `log(Close_t / Close_{t-1})` → z-score → windows  
- **Volatility**: `log_returns²` (squared returns) → z-score → windows
  - **Why squared returns**: Captures volatility clustering without rolling window artifacts
  - Avoids ACF distortion from overlapping windows
  - GARCH-like representation of instantaneous volatility
- All features aligned to same length and clipped to ±6σ

### 2. Wavelet Transform (cell 4)
- Uses **Stationary Wavelet Transform (SWT)** with `trim_approx=False`
- Preserves ALL coefficients: 1 approximation (cA) + 8 detail levels (cD1-cD8)
- Output shape: `(N_samples, 9, 256, 1)` where 9 = frequency levels
- **CRITICAL**: SWT (not DWT) ensures translation invariance and perfect reconstruction

### 3. Diffusion Model (`src/diffusion/`)
- **UNet** (`unet.py`): 
  - Base channels: 64, time embedding: 128 dims
  - 3-level encoder/decoder with skip connections
  - `SinusoidalPosEmb` for timestep injection into ResidualBlocks
- **Diffusion** (`diffusion.py`):
  - Default: 1500 timesteps, cosine beta schedule
  - Training loss: Huber loss (robust to outliers)
  - Sampling: temperature=1.5 for diversity control
  - Supports quadratic/sigmoid schedules for heavier tails

### 4. Key Data Variables (across notebook cells)
```python
# Cell 2 outputs (use these names exactly):
all_windows_combined              # (N, 256) z-score volume
all_windows_combined_log_returns  # (N, 256) z-score log returns  
all_windows_combined_volatility   # (N, 256) z-score volatility

# Cell 4 outputs:
wavelet_images                    # (N, 9, 256, 1) normalized wavelets
wavelet_normalization_params.npy  # {p5, p95, wavelet_type, level, ...}

# Cell 6 outputs:
synthetic_samples                 # (N, 1, 9, 256) generated wavelets

# Cell 7 outputs:
synthetic_zscore_log_volumes      # (N, 256) reconstructed in z-score space
original_zscore_log_volumes       # (N, 256) original for comparison
```

## Development Workflows

### Running Experiments
```bash
# Cell execution order for full pipeline:
1. Cell 1: Imports
2. Cell 2: Download & preprocess (MODIFIED - includes log returns & volatility)
3. Cell 3: Visualize distributions & ACF
4. Cell 4: Wavelet transform
5. Cell 5: Train diffusion model (saves to ./diffusion_checkpoints/)
6. Cell 6: Generate synthetic samples
7. Cell 7: Reconstruct to z-score space
8. Cell 8: Compare distributions & ACF
```

### Training Checkpoints
- Saved every 10,000 steps + end of each epoch
- Location: `notebooks/diffusion_checkpoints/ckpt_epoch_N.pt`
- Resume: Set `RESUME_FROM = './diffusion_checkpoints/ckpt_epoch_50.pt'` in cell 5

### Testing
```bash
pytest tests/test_detrending.py  # Validates wavelet pipeline
```

## Project-Specific Conventions

### Naming Patterns
- **Time series variables**: `{ticker}_{feature}` (e.g., `ticker_log_volumes`)
- **Windowed data**: `ticker_windows_{feature}` or `all_windows_combined_{feature}`
- **Z-score normalized**: suffix `_zscore` or prefix `zscore_`
- **Checkpoints**: `ckpt_epoch_{N}.pt` or `ckpt_step_{N}.pt`

### Data Shapes
- Raw series: `(time_steps,)` 1D arrays
- Windows: `(n_windows, window_length)` = `(N, 256)`
- Wavelet images: `(N, H, W, C)` = `(N, 9, 256, 1)` in NumPy
- Model input: `(B, C, H, W)` = `(batch, 1, 9, 256)` in PyTorch

### Critical Parameters (do NOT change without retraining)
```python
WINDOW_LENGTH = 256        # Must be power of 2
wavelet_type = 'haar'      # SWT wavelet family
wavelet_level = 8          # log2(256) decomposition levels
TIMESTEPS = 1500           # Diffusion timesteps
```

### Code Patterns
1. **Always check for saved files before regenerating**:
   ```python
   try:
       data = variable_name
   except NameError:
       if os.path.exists('file.npy'):
           data = np.load('file.npy')
   ```

2. **Normalize before diffusion, denormalize after**:
   - Percentile clipping for wavelets: `(x - p5) / (p95 - p5)`
   - Z-score for time series: `(x - mean) / std` with ±6σ clipping

3. **Shape transformations**:
   ```python
   # NumPy (N,H,W,C) → PyTorch (N,C,H,W)
   x_torch = np.transpose(x_numpy, (0, 3, 1, 2))
   # Scaling: [0,1] → [-1,1] for diffusion
   x_scaled = x * 2.0 - 1.0
   ```

## Integration Points

### External Dependencies
- **yfinance**: Stock data download (handles API throttling automatically)
- **PyWavelets**: `pywt.swt()` for forward, `pywt.iswt()` for inverse transform
- **PyTorch**: Model training requires CUDA-capable GPU for reasonable speed

### Cross-Component Communication
- **Preprocessing → Diffusion**: Save `wavelet_images.npy` + `wavelet_normalization_params.npy`
- **Diffusion → Reconstruction**: Load params to denormalize, then inverse SWT
- **Reconstruction → Analysis**: Compare synthetic vs original in same z-score space

## Common Pitfalls

1. **Shape mismatches**: Always verify `(N, H, W, C)` vs `(N, C, H, W)` when crossing NumPy/PyTorch boundary
2. **Missing normalization params**: Cannot reconstruct without saved `p5`, `p95` from training
3. **SWT vs DWT confusion**: This project uses SWT exclusively for translation invariance
4. **Cell execution order**: Cell 2 MUST run before cells 3-8 (creates all base variables)
5. **GPU memory**: Reduce `BATCH_SIZE` if OOM errors occur during training

## When Modifying Cell 2

If you add new features (like we just did with log returns & volatility):
1. Download/compute the raw feature
2. Apply appropriate transform (log for prices, identity for returns)
3. Compute z-score normalization **per feature** (separate mean/std)
4. Create windowed datasets with same `WINDOW_LENGTH` and `STRIDE`
5. Create both raw and z-score versions following naming convention
6. Add to concatenated arrays: `all_windows_combined_{feature_name}`
7. **DO NOT modify cells 3-8** - they operate on whatever features exist in cell 2

## Quick Reference

### File Structure Importance
- `src/diffusion/`: Core model code (importable module)
- `src/preprocessing/`: Wavelet pipeline (currently unused, notebook has inline version)
- `notebooks/`: Primary development environment
- `tests/`: Unit tests for preprocessing logic

### Key Hyperparameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| WINDOW_LENGTH | 256 | Time series window size |
| BATCH_SIZE | 32 | Training batch size |
| EPOCHS | 40 | Training epochs |
| TIMESTEPS | 1500 | Diffusion steps |
| TEMPERATURE | 1.5 | Sampling diversity |
| LR | 1e-3 | Learning rate |

### Validation Metrics
- **ACF (Autocorrelation)**: Lag 0-20, target >0.90 correlation
- **KS Test**: p-value >0.05 for distribution similarity
- **Visual**: Histogram overlay, box plots, ACF bar charts

---

**Last Updated**: December 3, 2025  
**Maintained by**: Marco (Thesis project)
