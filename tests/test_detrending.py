"""
Quick test of the wavelet detrending pipeline
Run this to verify the implementation works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from wavelet_detrending import WaveletDetrendingPipeline, visualize_detrending_effect

# Create synthetic test data similar to volume series
np.random.seed(42)
num_series = 100
window_size = 64

# Generate test data with large DC component (like real volume data)
means = np.random.uniform(15, 20, size=(num_series, 1))  # Large DC component
variations = np.random.randn(num_series, window_size) * 2  # Small variations
test_volume_series = means + variations

print("=== TEST DATA STATISTICS ===")
print(f"Shape: {test_volume_series.shape}")
print(f"Mean of means: {means.mean():.4f}")
print(f"Std of variations: {variations.std():.4f}")
print(f"Overall mean: {test_volume_series.mean():.4f}")
print(f"Overall std: {test_volume_series.std():.4f}")

# Initialize pipeline
pipeline = WaveletDetrendingPipeline(wavelet='haar', level=6)

# Test preprocessing
print("\n=== TESTING PREPROCESSING ===")
wavelet_images, stats = pipeline.preprocess(test_volume_series, return_stats=True)

print(f"Wavelet images shape: {wavelet_images.shape}")
print(f"Original mean: {stats['original_mean']:.4f}")
print(f"Detrended mean: {stats['detrended_mean']:.6f}")
print(f"Wavelet mean (before norm): {stats['wavelet_mean_before_norm']:.4f}")
print(f"Final mean (after norm): {stats['final_mean']:.4f}")
print(f"Final std: {stats['final_std']:.4f}")

# Test postprocessing
print("\n=== TESTING POSTPROCESSING ===")
reconstructed = pipeline.postprocess(wavelet_images, original_length=64)

print(f"Reconstructed shape: {reconstructed.shape}")
print(f"Reconstructed mean: {reconstructed.mean():.4f}")
print(f"Reconstructed std: {reconstructed.std():.4f}")

# Check reconstruction accuracy
reconstruction_error = np.mean(np.abs(test_volume_series - reconstructed))
print(f"\nReconstruction MAE: {reconstruction_error:.6f}")

# Detrending info
print("\n=== DETRENDING INFO ===")
info = pipeline.get_detrending_info()
for key, value in info.items():
    if isinstance(value, (int, float, np.number)):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")

# Visualization test
print("\n=== GENERATING VISUALIZATION ===")
visualize_detrending_effect(test_volume_series, pipeline, idx=0)

# Comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original time series
axes[0, 0].plot(test_volume_series[0])
axes[0, 0].set_title(f'Original Time Series (mean={test_volume_series[0].mean():.4f})')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# Reconstructed time series
axes[0, 1].plot(reconstructed[0])
axes[0, 1].set_title(f'Reconstructed Time Series (mean={reconstructed[0].mean():.4f})')
axes[0, 1].set_ylabel('Value')
axes[0, 1].grid(True, alpha=0.3)

# Distribution comparison
axes[1, 0].hist(test_volume_series.flatten(), bins=50, alpha=0.7, label='Original')
axes[1, 0].hist(reconstructed.flatten(), bins=50, alpha=0.7, label='Reconstructed')
axes[1, 0].set_title('Distribution Comparison')
axes[1, 0].set_xlabel('Value')
axes[1, 0].legend()

# Wavelet image example
axes[1, 1].imshow(wavelet_images[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
axes[1, 1].set_title('Detrended Wavelet Image (Normalized)')
axes[1, 1].set_ylabel('Frequency Level')
axes[1, 1].set_xlabel('Time')

plt.tight_layout()
plt.show()

print("\n=== TEST SUMMARY ===")
if reconstruction_error < 0.1:
    print("✅ PASS: Reconstruction error is low")
else:
    print(f"⚠️  WARNING: Reconstruction error is high: {reconstruction_error:.6f}")

if abs(reconstructed.mean() - test_volume_series.mean()) < 0.5:
    print("✅ PASS: Mean is preserved")
else:
    print(f"⚠️  WARNING: Mean not preserved: {abs(reconstructed.mean() - test_volume_series.mean()):.4f}")

if abs(stats['detrended_mean']) < 1e-5:
    print("✅ PASS: Detrending removes DC component")
else:
    print(f"⚠️  WARNING: Detrended mean not zero: {stats['detrended_mean']:.6f}")

print("\n✅ All tests completed!")
