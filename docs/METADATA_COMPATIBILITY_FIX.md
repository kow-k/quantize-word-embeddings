# Metadata Compatibility Fix

## Problem

Quantized embeddings saved with metadata comments broke gensim's loader:

```python
# This file has a comment line that breaks gensim
400000 200
# Quantization: {'base_k': 32, 'method': 'adaptive'}  ← gensim tries to parse this!
the 0.418 0.24968 ...
```

**Error:**
```
ValueError: could not convert string to float: 'Quantization:'
```

## Solution: Smart Default + Opt-in Flag

### Default Behavior (Safe)
**No metadata in .vec/.bin files** - maximum compatibility with gensim and other tools.

```bash
# Default: No metadata (gensim-compatible)
python convert_embeddings.py input.vec output.vec
```

Output file:
```
400000 200
the 0.418 0.24968 ...
, 0.013441 0.23682 ...
```

### Opt-in with --add-metadata Flag

Users can explicitly add metadata if they know they'll only use our `load_embeddings()` function:

```bash
# Add metadata (breaks gensim!)
python convert_embeddings.py input.vec output.vec --add-metadata
```

Output file with warning:
```
⚠️  Warning: Metadata comment added to .vec file
   This breaks gensim's KeyedVectors.load_word2vec_format()
   Use our load_embeddings() or remove comment line manually
```

### Automatic Metadata for Modern Formats

**.pt, .npz, .h5 formats always include metadata** regardless of flag (they natively support it):

```bash
# Metadata automatically added (safe, supported format)
python convert_embeddings.py input.vec output.pt
python convert_embeddings.py input.vec output.npz
python convert_embeddings.py input.vec output.h5
```

## Implementation Details

### Smart Default Logic in `save_embeddings()`

```python
def save_embeddings(embeddings, words, filepath, format=None, 
                   quantization_info=None, add_metadata=None):
    """
    Args:
        add_metadata: Whether to add metadata. If None, auto-decide:
                     - .vec/.bin: False (gensim compatibility)
                     - .pt/.npz/.h5: True (native support)
    """
    
    # Auto-decide metadata policy
    if add_metadata is None:
        if format in ['pytorch', 'numpy', 'hdf5']:
            add_metadata = True  # Native support
        else:  # word2vec_text, word2vec_bin
            add_metadata = False  # Compatibility
```

### Warning System

When `--add-metadata` is used with .vec/.bin:

```python
if add_metadata and format in ['word2vec_text', 'word2vec_bin']:
    print("⚠️  WARNING: Adding metadata to .vec/.bin format")
    print("   This will break compatibility with gensim")
    print("   Use only if loading with our load_embeddings()")
```

## Usage Examples

### Example 1: Maximum Compatibility (Default)

```bash
# No metadata, works with any tool
python convert_embeddings.py glove.bin glove-q32.vec
```

**Use case:** Deploying to production, sharing with others, using with gensim

### Example 2: Research with Metadata Tracking

```bash
# Use PyTorch format for automatic metadata
python convert_embeddings.py glove.bin glove-q32.pt
```

**Use case:** Research experiments, need to track quantization parameters

### Example 3: Forced Metadata (Advanced)

```bash
# Add metadata to .vec (breaks gensim!)
python convert_embeddings.py glove.bin glove-q32.vec --add-metadata
```

**Use case:** Only using our `load_embeddings()`, need human-readable + metadata

### Example 4: Batch Conversion

```bash
# All files without metadata (safe)
python batch_conversion.py *.vec --output-dir quantized/

# All files with metadata (breaks gensim!)
python batch_conversion.py *.vec --output-dir quantized/ --add-metadata
```

## Loading Files

### Option 1: Our Loader (Handles Everything)

```python
from adaptive_quantization import load_embeddings

# Works with or without metadata comments
embeddings, words = load_embeddings('output.vec')
```

**Skips comment lines automatically**

### Option 2: Gensim (Requires No Metadata)

```python
from gensim.models import KeyedVectors

# Only works if --add-metadata was NOT used
kv = KeyedVectors.load_word2vec_format('output.vec', binary=False)
```

**Breaks if file has metadata comments**

### Option 3: Manual Fix

If you accidentally added metadata and need gensim:

```bash
# Remove comment line manually
sed -i '/^# Quantization:/d' output.vec
```

## Command-Line Flags

### convert_embeddings.py

```bash
--add-metadata       Add metadata to output (breaks gensim for .vec/.bin)
```

### evaluate_quantization.py

```bash
--add-metadata       Add metadata to saved embeddings (breaks gensim for .vec/.bin)
```

### batch_conversion.py

```bash
--add-metadata       Add metadata to all output files (breaks gensim for .vec/.bin)
```

## Format Behavior Matrix

| Format | Default Metadata | With --add-metadata | Compatibility |
|--------|------------------|---------------------|---------------|
| .vec   | ❌ None          | ⚠️ Comment line     | Breaks gensim |
| .bin   | ❌ None          | ⚠️ Not supported    | N/A           |
| .pt    | ✅ Full dict     | ✅ Full dict        | PyTorch only  |
| .npz   | ✅ JSON string   | ✅ JSON string      | NumPy only    |
| .h5    | ✅ Attributes    | ✅ Attributes       | h5py/HDF5     |

## Migration Guide

### If You Previously Generated Files

**Old behavior:** All .vec files had metadata comments (broke gensim)

**New behavior:** No metadata by default (gensim-compatible)

**Action needed:**
```bash
# Regenerate all .vec files without metadata
python convert_embeddings.py input.vec output.vec  # No --add-metadata

# Or keep existing files and remove comments
sed -i '/^# Quantization:/d' *.vec
```

### If You Need Metadata

**Recommended:** Use .pt, .npz, or .h5 formats
```bash
python convert_embeddings.py input.vec output.pt    # Metadata included
python convert_embeddings.py input.vec output.npz   # Metadata included
python convert_embeddings.py input.vec output.h5    # Metadata included
```

**Alternative:** Use --add-metadata flag (only if not using gensim)
```bash
python convert_embeddings.py input.vec output.vec --add-metadata
```

## Key Takeaways

1. ✅ **Default = Safe**: No metadata in .vec/.bin (gensim works)
2. ⚠️ **Opt-in = Break**: --add-metadata breaks gensim compatibility
3. 💡 **Best Practice**: Use .pt/.npz/.h5 for metadata storage
4. 🔧 **Our Loader**: Handles both cases automatically

## Files Modified

1. ✅ **adaptive_quantization.py**
   - Added `add_metadata` parameter to `save_embeddings()`
   - Smart default logic (False for .vec/.bin, True for .pt/.npz/.h5)
   - Warning when adding metadata to incompatible formats

2. ✅ **convert_embeddings.py**
   - Added `--add-metadata` flag
   - Updated examples and documentation
   - Pass flag to `save_embeddings()`

3. ✅ **evaluate_quantization.py**
   - Added `--add-metadata` flag
   - Pass to `save_quantized_embeddings()`

4. ✅ **batch_conversion.py**
   - Added `--add-metadata` flag
   - Pass to `convert_file()`

## Testing

**Test 1: Default behavior (no metadata)**
```bash
python convert_embeddings.py test.vec test-q32.vec
gensim-test test-q32.vec  # Should work!
```

**Test 2: With metadata flag**
```bash
python convert_embeddings.py test.vec test-q32.vec --add-metadata
gensim-test test-q32.vec  # Should fail with helpful error
our-loader test-q32.vec   # Should work!
```

**Test 3: Modern formats (auto-metadata)**
```bash
python convert_embeddings.py test.vec test-q32.pt
# Metadata automatically included, works perfectly
```

All updated files ready in `/mnt/user-data/outputs/`!
