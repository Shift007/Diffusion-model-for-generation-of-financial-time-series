# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-27

### Added
- Initial release
- Wavelet-based preprocessing pipeline with detrending
- DDPM diffusion model implementation
- U-Net architecture for wavelet coefficient generation
- Multi-stock volume data synthesis
- ACF validation and statistical comparison tools
- Jupyter notebooks for experimentation
- Comprehensive documentation

### Features
- Volatility clustering preservation
- Autocorrelation pattern matching
- Configurable window length and model parameters
- Support for multiple stock tickers
- Log-scale transformation for financial data

### Known Issues
- Model requires significant GPU memory for large batch sizes
- Long training times for convergence (30+ epochs recommended)
