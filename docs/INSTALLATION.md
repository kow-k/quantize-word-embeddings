# Installation Guide

Detailed installation instructions for the Adaptive Word Embedding Quantization library.

## Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization

# Install dependencies
pip install -r requirements.txt
```

That's it! You can now use the library.

---

## Requirements

### Python Version

- **Python 3.8 or higher** (tested on 3.8, 3.9, 3.10, 3.11, 3.12)

### Core Dependencies

```
numpy>=1.20.0
scipy>=1.6.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

### Optional Dependencies

For development:
```
pytest>=6.0.0
pytest-cov>=2.10.0
black>=21.0
flake8>=3.9.0
mypy>=0.910
```

---

## Installation Methods

### Method 1: Direct from GitHub (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Test installation
python -c "from adaptive_quantization import AdaptiveQuantizer; print('Success!')"
```

### Method 2: Development Install

For contributing or development:

```bash
# Clone and enter directory
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt

# Verify installation
pytest tests/
```

### Method 3: System-wide Install

Not recommended, but possible:

```bash
# Clone repository
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization

# Install system-wide (requires sudo on Linux/Mac)
pip install -r requirements.txt

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/embedding-quantization"
```

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python and pip if needed
sudo apt install python3.8 python3-pip python3-venv

# Clone and install
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS

```bash
# Install Python using Homebrew (if needed)
brew install python@3.9

# Clone and install
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
# Clone repository
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**Note for Windows users:**
- Ensure Python is added to PATH during installation
- You may need Microsoft Visual C++ Build Tools for some dependencies

---

## Verify Installation

### Quick Test

```python
from adaptive_quantization import AdaptiveQuantizer
import numpy as np

# Create test embeddings
embeddings = np.random.randn(100, 50)

# Initialize quantizer
quantizer = AdaptiveQuantizer(base_k=32, verbose=True)

# Quantize
quantized = quantizer.quantize(embeddings)

print("✓ Installation successful!")
print(f"  Original shape: {embeddings.shape}")
print(f"  Quantized shape: {quantized.shape}")
```

### Run Test Suite

```bash
# If you installed with dev dependencies
pytest tests/

# Or run examples
python examples/basic_quantization.py
```

---

## Common Issues

### Issue 1: NumPy/SciPy Build Errors

**Problem:** Compilation errors when installing NumPy or SciPy

**Solution:**
```bash
# Use pre-built wheels
pip install --only-binary :all: numpy scipy

# Or upgrade pip
pip install --upgrade pip setuptools wheel
pip install numpy scipy
```

### Issue 2: Import Errors

**Problem:** `ModuleNotFoundError` when importing

**Solution:**
```bash
# Ensure you're in the right directory
cd /path/to/embedding-quantization

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### Issue 3: scikit-learn Version Conflict

**Problem:** Incompatible scikit-learn version

**Solution:**
```bash
# Update to compatible version
pip install --upgrade scikit-learn>=0.24.0

# Or create fresh environment
python -m venv fresh_venv
source fresh_venv/bin/activate
pip install -r requirements.txt
```

### Issue 4: Matplotlib Display Issues

**Problem:** Plots don't display on headless servers

**Solution:**
```python
# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

# Now plots will save but not display
```

### Issue 5: Memory Errors with Large Embeddings

**Problem:** Out of memory when processing large embedding files

**Solution:**
```python
# Process in chunks or use smaller sample for analysis
quantizer = AdaptiveQuantizer(base_k=32, verbose=False)

# Analyze on sample
sample_size = min(50000, embeddings.shape[0])
sample = embeddings[np.random.choice(embeddings.shape[0], sample_size, replace=False)]
quantizer.analyze_dimensions(sample)

# Then quantize full embeddings
quantized = quantizer.quantize(embeddings)
```

---

## Docker Installation (Advanced)

For containerized deployment:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Run tests
RUN python -c "from adaptive_quantization import AdaptiveQuantizer; print('OK')"

# Entry point
CMD ["python", "evaluate_quantization.py", "--help"]
```

Build and run:
```bash
docker build -t embedding-quantization .
docker run -v $(pwd)/embeddings:/data embedding-quantization python evaluate_quantization.py /data/embeddings.vec
```

---

## Conda Installation

For users preferring Conda:

```bash
# Create environment
conda create -n embedding-quant python=3.9
conda activate embedding-quant

# Install dependencies
conda install numpy scipy scikit-learn matplotlib

# Clone and test
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization
python -c "from adaptive_quantization import AdaptiveQuantizer; print('Success!')"
```

---

## Upgrading

To update to the latest version:

```bash
cd embedding-quantization
git pull origin main
pip install --upgrade -r requirements.txt
```

---

## Uninstallation

To remove the library:

```bash
# If installed in virtual environment, just delete it
rm -rf venv/

# If installed system-wide
pip uninstall numpy scipy scikit-learn matplotlib

# Remove cloned repository
cd ..
rm -rf embedding-quantization/
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: https://github.com/yourusername/embedding-quantization/issues
2. **Create new issue**: Include Python version, OS, error message
3. **Email support**: kow.k@ks.kyorin-u.ac.jp

---

## Next Steps

After installation:

1. **Read README.md** - Overview and quick start
2. **Try examples/** - Basic usage examples
3. **Read EXAMPLES.md** - Comprehensive usage guide
4. **Check API_REFERENCE.md** - Detailed API documentation

---

## Minimum Working Example

Test your installation with this complete example:

```python
#!/usr/bin/env python3
"""
Minimal working example to verify installation.
"""

from adaptive_quantization import AdaptiveQuantizer
import numpy as np

# Create synthetic embeddings
print("Creating synthetic embeddings...")
embeddings = np.random.randn(1000, 100)

# Quantize with k=32 (recommended)
print("Quantizing...")
quantizer = AdaptiveQuantizer(base_k=32, verbose=False)
quantized = quantizer.quantize(embeddings)

# Verify
print("\n✓ Success!")
print(f"  Input:  {embeddings.shape}")
print(f"  Output: {quantized.shape}")
print(f"  Dimensions analyzed: {len(quantizer.dimension_profiles)}")

summary = quantizer.get_summary()
print(f"  Distribution types: {list(summary['distribution_types'].keys())}")
print(f"  k range: {summary['k_range']}")

print("\nInstallation verified! Ready to use.")
```

Save as `test_install.py` and run:
```bash
python test_install.py
```

Expected output:
```
Creating synthetic embeddings...
Quantizing...

✓ Success!
  Input:  (1000, 100)
  Output: (1000, 100)
  Dimensions analyzed: 100
  Distribution types: ['gaussian', 'right_skewed', 'left_skewed']
  k range: (32, 48)

Installation verified! Ready to use.
```

---

**Installation complete! Happy quantizing! 🎉**
