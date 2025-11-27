"""
Wavelet Detrending Pipeline for Diffusion Model Training
Solves the DC component mode collapse problem by removing mean before SWT
"""

import numpy as np
import pywt


class WaveletDetrendingPipeline:
    """
    Pipeline to handle detrending for SWT-based diffusion model training.
    
    The problem: Volume time series have large DC components (mean ≈ 18) with small variance (±2).
    This causes the wavelet coefficients to be dominated by the DC component, leading to mode collapse.
    
    The solution: 
    1. Remove mean from each time series before SWT transform
    2. Train diffusion model on detrended wavelet coefficients
    3. Restore mean after inverse SWT reconstruction
    """
    
    def __init__(self, wavelet='haar', level=6):
        """
        Args:
            wavelet: Wavelet type (default 'haar')
            level: Decomposition level (default 6 for 64-length series)
        """
        self.wavelet = wavelet
        self.level = level
        self.means = None
        self.normalization_params = None
    
    def preprocess(self, volume_series, return_stats=False):
        """
        Convert volume time series to detrended wavelet images.
        
        Args:
            volume_series: Array of shape (num_series, window_size)
            return_stats: If True, return detrending statistics
            
        Returns:
            wavelet_images: Array of shape (num_series, level, window_size, 1)
            If return_stats=True, also returns dict with detrending info
        """
        num_series = volume_series.shape[0]
        
        # Step 1: Remove DC component (mean) from each series
        self.means = volume_series.mean(axis=1, keepdims=True)  # (num_series, 1)
        detrended_series = volume_series - self.means
        
        # Step 2: Apply SWT on detrended data
        wavelet_images_raw = []
        for ts in detrended_series:
            coeffs = pywt.swt(ts, self.wavelet, level=self.level, start_level=0, trim_approx=True)
            cDs = np.stack(coeffs)  # shape: (level, window_size)
            cDs = cDs[::-1]  # Flip for standard convention
            wavelet_images_raw.append(cDs)
        
        wavelet_images_raw = np.stack(wavelet_images_raw)  # (num_series, level, window_size)
        
        # Step 3: Robust normalization using percentiles
        p5 = np.percentile(wavelet_images_raw, 5)
        p95 = np.percentile(wavelet_images_raw, 95)
        
        # Store normalization params for inverse transform
        self.normalization_params = {'p5': p5, 'p95': p95}
        
        # Scale to [0, 1]
        wavelet_images_norm = np.clip(
            (wavelet_images_raw - p5) / (p95 - p5 + 1e-8), 
            0, 1
        )
        
        # Step 4: Add channel dimension
        wavelet_images = wavelet_images_norm[:, :, :, np.newaxis]
        
        if return_stats:
            stats = {
                'original_mean': self.means.mean(),
                'original_std': self.means.std(),
                'detrended_mean': detrended_series.mean(),
                'detrended_std': detrended_series.std(),
                'wavelet_mean_before_norm': wavelet_images_raw.mean(),
                'wavelet_std_before_norm': wavelet_images_raw.std(),
                'p5': p5,
                'p95': p95,
                'final_mean': wavelet_images.mean(),
                'final_std': wavelet_images.std()
            }
            return wavelet_images, stats
        
        return wavelet_images
    
    def postprocess(self, wavelet_images_normalized, original_length=64):
        """
        Convert normalized wavelet images back to volume time series.
        Restores DC component (mean) for realistic reconstruction.
        
        Args:
            wavelet_images_normalized: Array of shape (num_series, level, window_size, 1)
            original_length: Length of output time series (default 64)
            
        Returns:
            volume_series: Array of shape (num_series, original_length)
        """
        if self.normalization_params is None:
            raise RuntimeError("Must call preprocess() before postprocess()")
        if self.means is None:
            raise RuntimeError("Must call preprocess() before postprocess()")
        
        # Step 1: Remove channel dimension
        wavelet_images = wavelet_images_normalized.squeeze(-1)  # (num_series, level, window_size)
        
        # Step 2: Denormalize using stored percentiles
        p5 = self.normalization_params['p5']
        p95 = self.normalization_params['p95']
        wavelet_images_raw = wavelet_images * (p95 - p5 + 1e-8) + p5
        
        # Step 3: Inverse SWT
        reconstructed_detrended = []
        for i in range(wavelet_images_raw.shape[0]):
            cDs = wavelet_images_raw[i][::-1]  # Flip back
            coeffs = [cDs[j] for j in range(self.level)]
            
            # Inverse SWT
            recon = pywt.iswt(coeffs, self.wavelet)
            
            # Handle length mismatch
            if len(recon) > original_length:
                recon = recon[:original_length]
            elif len(recon) < original_length:
                recon = np.pad(recon, (0, original_length - len(recon)), mode='edge')
            
            reconstructed_detrended.append(recon)
        
        reconstructed_detrended = np.stack(reconstructed_detrended)
        
        # Step 4: CRITICAL - Restore DC component (mean)
        # Use stored means from original preprocessing
        num_generated = reconstructed_detrended.shape[0]
        num_stored_means = self.means.shape[0]
        
        # If generating more series than original, repeat means cyclically
        if num_generated > num_stored_means:
            expanded_means = np.tile(self.means, (num_generated // num_stored_means + 1, 1))
            means_to_use = expanded_means[:num_generated]
        else:
            means_to_use = self.means[:num_generated]
        
        volume_series = reconstructed_detrended + means_to_use
        
        return volume_series
    
    def get_detrending_info(self):
        """Get information about the detrending statistics"""
        if self.means is None:
            return "Pipeline not yet used. Call preprocess() first."
        
        info = {
            'num_series': len(self.means),
            'mean_of_means': self.means.mean(),
            'std_of_means': self.means.std(),
            'min_mean': self.means.min(),
            'max_mean': self.means.max(),
            'normalization_p5': self.normalization_params['p5'] if self.normalization_params else None,
            'normalization_p95': self.normalization_params['p95'] if self.normalization_params else None,
        }
        return info


def visualize_detrending_effect(volume_series, pipeline, idx=0):
    """
    Visualize the effect of detrending on a single time series.
    
    Args:
        volume_series: Array of shape (num_series, window_size)
        pipeline: WaveletDetrendingPipeline instance (can be new)
        idx: Index of series to visualize
    """
    import matplotlib.pyplot as plt
    
    # Original series
    original = volume_series[idx]
    
    # Detrended series
    mean_val = original.mean()
    detrended = original - mean_val
    
    # Create wavelets
    coeffs_original = pywt.swt(original, pipeline.wavelet, level=pipeline.level, 
                                start_level=0, trim_approx=True)
    coeffs_detrended = pywt.swt(detrended, pipeline.wavelet, level=pipeline.level, 
                                 start_level=0, trim_approx=True)
    
    wavelet_original = np.stack(coeffs_original)[::-1]
    wavelet_detrended = np.stack(coeffs_detrended)[::-1]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Original
    axes[0, 0].plot(original)
    axes[0, 0].set_title(f'Original Series (mean={mean_val:.4f})')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    im1 = axes[0, 1].imshow(wavelet_original, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title('Wavelet Transform (Original)')
    axes[0, 1].set_ylabel('Frequency Level')
    plt.colorbar(im1, ax=axes[0, 1])
    
    axes[0, 2].hist(wavelet_original.flatten(), bins=50, alpha=0.7)
    axes[0, 2].set_title('Wavelet Coefficients Distribution')
    axes[0, 2].set_xlabel('Coefficient Value')
    axes[0, 2].axvline(wavelet_original.mean(), color='r', linestyle='--', 
                       label=f'Mean={wavelet_original.mean():.4f}')
    axes[0, 2].legend()
    
    # Row 2: Detrended
    axes[1, 0].plot(detrended)
    axes[1, 0].set_title(f'Detrended Series (mean={detrended.mean():.6f})')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    im2 = axes[1, 1].imshow(wavelet_detrended, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 1].set_title('Wavelet Transform (Detrended)')
    axes[1, 1].set_ylabel('Frequency Level')
    axes[1, 1].set_xlabel('Time')
    plt.colorbar(im2, ax=axes[1, 1])
    
    axes[1, 2].hist(wavelet_detrended.flatten(), bins=50, alpha=0.7, color='orange')
    axes[1, 2].set_title('Wavelet Coefficients Distribution')
    axes[1, 2].set_xlabel('Coefficient Value')
    axes[1, 2].axvline(wavelet_detrended.mean(), color='r', linestyle='--', 
                       label=f'Mean={wavelet_detrended.mean():.4f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("=== DETRENDING STATISTICS ===")
    print(f"Original series mean: {mean_val:.4f}")
    print(f"Detrended series mean: {detrended.mean():.6f}")
    print(f"\nWavelet coefficients (Original):")
    print(f"  Mean: {wavelet_original.mean():.4f}")
    print(f"  Std: {wavelet_original.std():.4f}")
    print(f"  Range: [{wavelet_original.min():.4f}, {wavelet_original.max():.4f}]")
    print(f"\nWavelet coefficients (Detrended):")
    print(f"  Mean: {wavelet_detrended.mean():.4f}")
    print(f"  Std: {wavelet_detrended.std():.4f}")
    print(f"  Range: [{wavelet_detrended.min():.4f}, {wavelet_detrended.max():.4f}]")
    print(f"\nDC component reduction: {abs(wavelet_original.mean() / (wavelet_detrended.mean() + 1e-8)):.1f}x")
