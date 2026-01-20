# Multi-Format Support Update

## Summary

I've significantly enhanced the quantization toolkit to support **five different embedding formats** with automatic format detection and seamless conversion. This makes the toolkit much more versatile and compatible with different workflows.

## New Formats Supported

### 1. **Word2Vec Text** (.vec, .txt) 
- ✅ Universal compatibility
- ✅ Human-readable
- ✅ No dependencies
- 📝 Default format

### 2. **Word2Vec Binary** (.bin)
- ✅ Compact size
- ✅ Fast loading
- ✅ gensim compatible
- ⚠️ Requires: `pip install gensim`

### 3. **PyTorch** (.pt, .pth) [NEW!]
- ✅ **Full metadata support**
- ✅ Fast and efficient
- ✅ Perfect for research
- ⚠️ Requires: `pip install torch`

### 4. **NumPy Compressed** (.npz) [NEW!]
- ✅ **Full metadata support**
- ✅ No extra dependencies (NumPy only)
- ✅ Scientific Python standard
- 📝 Great for scientific computing

### 5. **HDF5** (.h5, .hdf5) [NEW!]
- ✅ **Full metadata support**
- ✅ Language-agnostic
- ✅ Scientific data standard
- ✅ Hierarchical organization
- ⚠️ Requires: `pip install h5py`

## Key Changes

### 1. Enhanced `adaptive_quantization.py`

#### New `save_embeddings()` function:
```python
def save_embeddings(embeddings, words, filepath, format=None, quantization_info=None)
```

**Features:**
- Automatic format detection from file extension
- Manual format override with `format` parameter
- Full metadata support for .pt, .npz, .h5
- Backwards compatible with old API

**Automatic format detection:**
```python
save_embeddings(emb, words, 'output.vec')   # → Word2Vec text
save_embeddings(emb, words, 'output.bin')   # → Word2Vec binary  
save_embeddings(emb, words, 'output.pt')    # → PyTorch
save_embeddings(emb, words, 'output.npz')   # → NumPy
save_embeddings(emb, words, 'output.h5')    # → HDF5
```

#### Updated `load_embeddings()` function:
```python
def load_embeddings(filepath) -> Tuple[np.ndarray, List[str]]
```

**Features:**
- Supports all five formats
- Automatic format detection
- Loads metadata when available
- Unified interface for all formats

### 2. Enhanced `evaluate_quantization.py`

#### New command-line options:
```bash
--save-format {vec,bin,pt,npz,h5}  # Choose output format
```

**Replaces:**
```bash
--save-binary  # OLD: Binary yes/no
```

**Example:**
```bash
# Save as PyTorch with metadata
python evaluate_quantization.py --base-k 32 --save-quantized --save-format pt embeddings.vec

# Save as NumPy
python evaluate_quantization.py --base-k 32 --save-quantized --save-format npz embeddings.vec

# Save as HDF5
python evaluate_quantization.py --base-k 32 --save-quantized --save-format h5 embeddings.vec
```

### 3. Enhanced `convert_embeddings.py`

#### New `--format` option:
```bash
--format {word2vec_text,word2vec_bin,pytorch,numpy,hdf5}
```

**Auto-detection from extension:**
```bash
python convert_embeddings.py input.vec output.pt   # → PyTorch
python convert_embeddings.py input.vec output.npz  # → NumPy
python convert_embeddings.py input.vec output.h5   # → HDF5
```

**Manual format override:**
```bash
python convert_embeddings.py input.vec output.xyz --format pytorch
```

### 4. Enhanced `batch_conversion.py`

#### New `--format` option:
```bash
--format {vec,bin,pt,npz,h5}
```

**Batch convert to any format:**
```bash
# Convert all to PyTorch
python batch_conversion.py *.vec --output-dir quantized/ --format pt

# Convert all to NumPy
python batch_conversion.py *.vec --output-dir quantized/ --format npz

# Convert all to HDF5  
python batch_conversion.py *.vec --output-dir quantized/ --format h5
```

### 5. New Documentation

#### **FORMAT_GUIDE.md** - Comprehensive format documentation:
- Detailed comparison of all formats
- Use case recommendations
- Performance benchmarks
- Troubleshooting guide
- Format conversion examples
- Installation requirements

## Usage Examples

### Python API

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings, save_embeddings

# Load from any format
embeddings, words = load_embeddings('input.vec')  # or .bin, .pt, .npz, .h5

# Quantize
quantizer = AdaptiveQuantizer(base_k=32, use_lebesgue=True)
quantized = quantizer.quantize(embeddings)

# Save to different formats
save_embeddings(quantized, words, 'output.vec')   # Word2Vec text
save_embeddings(quantized, words, 'output.bin')   # Word2Vec binary
save_embeddings(quantized, words, 'output.pt')    # PyTorch
save_embeddings(quantized, words, 'output.npz')   # NumPy
save_embeddings(quantized, words, 'output.h5')    # HDF5

# With metadata (PyTorch, NumPy, HDF5)
save_embeddings(quantized, words, 'output.pt',
               quantization_info={
                   'base_k': 32,
                   'method': 'adaptive',
                   'use_lebesgue': True,
                   'experiment_id': 'exp001',
                   'date': '2026-01-20'
               })
```

### Command-Line (evaluate_quantization.py)

```bash
# Text format (default)
python evaluate_quantization.py --base-k 32 --save-quantized embeddings.vec

# PyTorch format (with metadata)
python evaluate_quantization.py --base-k 32 --save-quantized --save-format pt embeddings.vec

# NumPy format
python evaluate_quantization.py --base-k 32 --save-quantized --save-format npz embeddings.vec

# HDF5 format
python evaluate_quantization.py --base-k 32 --save-quantized --save-format h5 embeddings.vec
```

### Command-Line (convert_embeddings.py)

```bash
# Auto-detect format from extension
python convert_embeddings.py input.vec output.pt    # → PyTorch
python convert_embeddings.py input.vec output.npz   # → NumPy
python convert_embeddings.py input.vec output.h5    # → HDF5

# Force specific format
python convert_embeddings.py input.vec output.xyz --format pytorch
```

### Batch Conversion

```bash
# Convert all to PyTorch
python batch_conversion.py *.vec --output-dir quantized/ --format pt

# Convert all to HDF5
python batch_conversion.py embeddings/*.vec --output-dir compressed/ --format h5
```

## Format Comparison

| Format | Extension | Metadata | Dependencies | Best For |
|--------|-----------|----------|--------------|----------|
| Word2Vec Text | .vec | Limited | None | Compatibility |
| Word2Vec Binary | .bin | None | gensim | Compact files |
| **PyTorch** | **.pt** | **Full** | torch | **Research** |
| **NumPy** | **.npz** | **Full** | None | **Scientific** |
| **HDF5** | **.h5** | **Full** | h5py | **Large-scale** |

## Metadata Support

Only PyTorch, NumPy, and HDF5 formats support full metadata:

```python
# Save with metadata
save_embeddings(quantized, words, 'output.pt',
               quantization_info={
                   'base_k': 32,
                   'method': 'adaptive',
                   'use_lebesgue': True,
                   'corpus': 'Wikipedia 2025',
                   'experiment_id': 'exp001'
               })

# Load and access metadata
import torch
data = torch.load('output.pt')
print(data['quantization_info'])  # Full metadata available

# Or load with unified interface (metadata loaded automatically)
embeddings, words = load_embeddings('output.pt')
```

## Installation for Full Support

```bash
# Minimal (text format only)
pip install numpy scipy scikit-learn matplotlib

# Full support
pip install numpy scipy scikit-learn matplotlib gensim torch h5py

# Or individually
pip install gensim  # For .bin support
pip install torch   # For .pt support  
pip install h5py    # For .h5 support
```

## Migration from Old API

### Old API (deprecated but still works):
```python
save_embeddings(emb, words, 'output.vec', binary=False)  # Text
save_embeddings(emb, words, 'output.bin', binary=True)   # Binary
```

### New API (recommended):
```python
save_embeddings(emb, words, 'output.vec')   # Auto-detect from extension
save_embeddings(emb, words, 'output.bin')   # Auto-detect from extension
save_embeddings(emb, words, 'output.pt')    # New formats!
save_embeddings(emb, words, 'output.npz')   # New formats!
save_embeddings(emb, words, 'output.h5')    # New formats!
```

## Files Changed/Added

### Modified:
1. **adaptive_quantization.py**
   - Rewrote `save_embeddings()` with format auto-detection
   - Added `_save_word2vec_text()`, `_save_word2vec_binary()`, `_save_pytorch()`, `_save_numpy()`, `_save_hdf5()`
   - Rewrote `load_embeddings()` with format auto-detection
   - Added `_load_pytorch()`, `_load_numpy()`, `_load_hdf5()`, `_load_word2vec_binary()`, `_load_word2vec_text()`

2. **evaluate_quantization.py**
   - Changed `--save-binary` to `--save-format {vec,bin,pt,npz,h5}`
   - Updated `save_quantized_embeddings()` to use format parameter

3. **convert_embeddings.py**
   - Changed `--binary` to `--format {word2vec_text,word2vec_bin,pytorch,numpy,hdf5}`
   - Updated examples and documentation

4. **batch_conversion.py**
   - Changed `--binary` to `--format {vec,bin,pt,npz,h5}`
   - Updated conversion logic

5. **README.md**
   - Added multi-format support to features
   - Updated examples with format demonstrations
   - Added FORMAT_GUIDE.md reference
   - Updated file structure

6. **CONTRIBUTING.md**
   - Added format testing guidelines

### Added:
7. **FORMAT_GUIDE.md** (NEW!)
   - Comprehensive format documentation
   - Format comparison table
   - Use case recommendations
   - Performance benchmarks
   - Troubleshooting guide

8. **UPDATE_SUMMARY_MULTIFORMAT.md** (NEW!)
   - This file - comprehensive change summary

## Next Steps

1. **Test the new formats:**
   ```bash
   # Quick test
   python convert_embeddings.py test.vec test.pt
   python convert_embeddings.py test.vec test.npz
   python convert_embeddings.py test.vec test.h5
   ```

2. **Update repository:**
   ```bash
   git add adaptive_quantization.py evaluate_quantization.py convert_embeddings.py
   git add examples/batch_conversion.py README.md CONTRIBUTING.md FORMAT_GUIDE.md
   git commit -m "Add multi-format support: PyTorch, NumPy, HDF5"
   git push origin main
   ```

3. **Consider adding:**
   - Unit tests for each format's save/load round-trip
   - Integration tests for format conversion
   - Performance benchmarks across formats
   - Example notebooks demonstrating each format

## Benefits

✅ **Flexibility**: Choose the best format for your workflow  
✅ **Metadata**: Full quantization parameter tracking in .pt, .npz, .h5  
✅ **Compatibility**: Works with PyTorch, NumPy, HDF5 ecosystems  
✅ **Research-friendly**: Track experiments with rich metadata  
✅ **Production-ready**: Deploy in preferred format  
✅ **Backwards compatible**: Old API still works  

All updated files are ready in `/mnt/user-data/outputs/`!
