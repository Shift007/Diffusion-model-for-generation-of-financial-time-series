# Contributing Guidelines

## Welcome!

Thank you for considering contributing to this project! This document outlines the process for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/financial-timeseries-synthesis.git
   cd financial-timeseries-synthesis
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Run `black` for code formatting:
  ```bash
  black src/ tests/
  ```

## Testing

- Write tests for new features
- Run tests before submitting PR:
  ```bash
  pytest tests/
  ```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git commit -m "Add: description of your changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Any relevant issue numbers
   - Screenshots/plots if applicable

## Questions?

Feel free to open an issue for any questions or discussions!
