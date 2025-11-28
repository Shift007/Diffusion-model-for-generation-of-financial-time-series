"""
Wavelet Detrending Pipeline for Diffusion Model Training
Solves the DC component mode collapse problem by removing mean before SWT
"""

import numpy as np
import pywt


import pywt
import numpy as np

class WaveletDetrendingPipeline:
    def __init__(self, wavelet='haar', level=6):
        self.wavelet = wavelet
        self.level = level
        self.means = None
        self.normalization_params = None
    
    def preprocess(self, volume_series, return_stats=False):
        """
        Convert volume time series to detrended wavelet images using DWT.
        
        DWT structure (for window_size=64, level=6):
        - Row 0 (cA6): 1 coefficient → extended to 64 (DC/constant frequency)
        - Row 1 (cD6): 1 coefficient → extended to 64 (lowest detail frequency)
        - Row 2 (cD5): 2 coefficients → extended to 64
        - Row 3 (cD4): 4 coefficients → extended to 64
        - Row 4 (cD3): 8 coefficients → extended to 64
        - Row 5 (cD2): 16 coefficients → extended to 64
        - Row 6 (cD1): 32 coefficients → extended to 64 (highest detail frequency)
        
        Each coefficient is replicated across its corresponding time window.
        """
        num_series = volume_series.shape[0]
        window_size = volume_series.shape[1]
        
        # Step 1: Remove DC component (mean)
        self.means = volume_series.mean(axis=1, keepdims=True)
        detrended_series = volume_series - self.means
        
        # Step 2: Apply DWT
        wavelet_images_raw = []
        for ts in detrended_series:
            # DWT decomposition
            coeffs = pywt.wavedec(ts, self.wavelet, level=self.level)
            # coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
            # Example for window_size=64, level=6:
            #   cA6: 1 coefficient (constant frequency component)
            #   cD6: 1 coefficient
            #   cD5: 2 coefficients
            #   cD4: 4 coefficients
            #   cD3: 8 coefficients
            #   cD2: 16 coefficients
            #   cD1: 32 coefficients (highest frequency)
            
            # Extend each coefficient to window_size by repeating each value
            coeffs_extended = []
            for coeff in coeffs:
                num_coeffs = len(coeff)
                # Each coefficient represents a time window of size (window_size / num_coeffs)
                window_per_coeff = window_size // num_coeffs
                
                # Repeat each coefficient across its time window
                extended = np.repeat(coeff, window_per_coeff)
                
                # Handle remainder if window_size is not perfectly divisible
                remainder = window_size - len(extended)
                if remainder > 0:
                    extended = np.concatenate([extended, np.repeat(coeff[-1], remainder)])
                
                coeffs_extended.append(extended)
            
            # Stack: (level+1, window_size)
            # Row 0: cA6 (constant frequency - single value replicated 64 times)
            # Row 1: cD6 (lowest detail - single value replicated 64 times)
            # Row 2: cD5 (2 values, each replicated 32 times)
            # ...
            # Row 6: cD1 (32 values, each replicated 2 times)
            wavelet_img = np.stack(coeffs_extended)
            wavelet_images_raw.append(wavelet_img)
        
        wavelet_images_raw = np.stack(wavelet_images_raw)  # (num_series, level+1, window_size)
        
        # Step 3: Normalization
        p5 = np.percentile(wavelet_images_raw, 5)
        p95 = np.percentile(wavelet_images_raw, 95)
        self.normalization_params = {'p5': p5, 'p95': p95}
        
        wavelet_images_norm = np.clip(
            (wavelet_images_raw - p5) / (p95 - p5 + 1e-8), 
            0, 1
        )
        
        # Step 4: Add channel dimension
        wavelet_images = wavelet_images_norm[:, :, :, np.newaxis]
        
        if return_stats:
            stats = {
                'original_mean': self.means.mean(),
                'detrended_mean': detrended_series.mean(),
                'detrended_std': detrended_series.std(),
                'p5': p5,
                'p95': p95,
                'final_mean': wavelet_images.mean(),
                'final_std': wavelet_images.std(),
                'num_levels': self.level + 1  # +1 for cA
            }
            return wavelet_images, stats
        
        return wavelet_images
    
    def postprocess(self, wavelet_images_normalized, original_length=64):
        """
        Convert normalized wavelet images back to volume time series using inverse DWT.
        Restores DC component (mean) for realistic reconstruction.
        
        Args:
            wavelet_images_normalized: Array of shape (num_series, level+1, window_size, 1)
            original_length: Length of output time series (default 64)
            
        Returns:
            volume_series: Array of shape (num_series, original_length)
        """
        if self.normalization_params is None:
            raise RuntimeError("Must call preprocess() before postprocess()")
        if self.means is None:
            raise RuntimeError("Must call preprocess() before postprocess()")
        
        # Step 1: Remove channel dimension
        wavelet_images = wavelet_images_normalized.squeeze(-1)  # (num_series, level+1, window_size)
        
        # Step 2: Denormalize using stored percentiles
        p5 = self.normalization_params['p5']
        p95 = self.normalization_params['p95']
        wavelet_images_raw = wavelet_images * (p95 - p5 + 1e-8) + p5
        
        # Step 3: Extract original DWT coefficients from extended images
        reconstructed_detrended = []
        for i in range(wavelet_images_raw.shape[0]):
            extended_coeffs = wavelet_images_raw[i]  # (level+1, window_size)
            
            # Extract original coefficients by averaging over each window
            coeffs = []
            for level_idx in range(len(extended_coeffs)):
                # Number of original coefficients at this level
                # Level 0 (cA): 1, Level 1 (cD6): 1, Level 2 (cD5): 2, etc.
                num_coeffs = 2 ** max(0, level_idx - 1) if level_idx > 0 else 1
                if level_idx == 0:  # cA
                    num_coeffs = 1
                else:  # cD levels
                    num_coeffs = 2 ** (level_idx - 1)
                
                # Extract coefficients by averaging windows
                window_per_coeff = original_length // num_coeffs
                coeff_values = []
                for j in range(num_coeffs):
                    start_idx = j * window_per_coeff
                    end_idx = start_idx + window_per_coeff
                    # Average the extended values to get back the original coefficient
                    avg_val = extended_coeffs[level_idx][start_idx:end_idx].mean()
                    coeff_values.append(avg_val)
                
                coeffs.append(np.array(coeff_values))
            
            # coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
            # Inverse DWT
            recon = pywt.waverec(coeffs, self.wavelet)
            
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
    
    # Create wavelets using DWT
    coeffs_original = pywt.wavedec(original, pipeline.wavelet, level=pipeline.level)
    coeffs_detrended = pywt.wavedec(detrended, pipeline.wavelet, level=pipeline.level)
    
    # Extend coefficients to visualize
    window_size = len(original)
    
    def extend_coeffs(coeffs_list):
        extended = []
        for coeff in coeffs_list:
            num_coeffs = len(coeff)
            window_per_coeff = window_size // num_coeffs
            ext = np.repeat(coeff, window_per_coeff)
            remainder = window_size - len(ext)
            if remainder > 0:
                ext = np.concatenate([ext, np.repeat(coeff[-1], remainder)])
            extended.append(ext)
        return np.stack(extended)
    
    wavelet_original = extend_coeffs(coeffs_original)
    wavelet_detrended = extend_coeffs(coeffs_detrended)
    
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
