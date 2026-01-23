# Embedding Format Guide

## ⚠️ Critical Understanding: File Size vs Semantic Quality

**Before reading about formats, understand this important limitation:**

All formats in this guide store values as **32-bit floats**, which means:
- ✅ Quantization **improves semantic quality** (+3-6% on benchmarks)
- ❌ Quantization **does NOT reduce file size** with standard formats
- 📊 File size stays the same: ~305 MB before and after quantization

**Why?**
- Quantization reduces values to k=32 levels (5 bits of information)
- But formats still allocate 32 bits per value
- Like writing single digits (0-9) using full 32-bit integers

**⚠️ CRITICAL: Format Conversion Can DOUBLE File Size!**

**Format overhead matters more than quantization:**

```bash
# WRONG: Binary → Text conversion DOUBLES size
input.bin  (305 MB) → output.vec (610 MB)  # 2× larger!

# RIGHT: Keep same format
input.bin  (305 MB) → output.bin (305 MB)  # ✓ Same size
input.vec  (610 MB) → output.vec (610 MB)  # ✓ Same size
```

**Format size comparison (GloVe-200, 400K words):**
- `.bin`: Binary, 4 bytes/float → **~305 MB** (smallest)
- `.pt/.npz/.h5`: Binary, compressed → **~305-320 MB** (similar)
- `.vec/.txt`: ASCII text, 8+ bytes/float → **~610 MB** (2× larger!)

**To actually reduce file size:**
- Use external compression: `gzip file.vec` (~2-3× reduction)
- Implement custom format storing k-bit indices (not yet available)
- Use specialized vector databases (FAISS, etc.)

**This guide focuses on format compatibility and metadata support, not compression.**

---

## Supported Formats

The quantization toolkit supports five different formats for saving and loading embeddings. Each format has its own advantages and use cases.

## Format Comparison Table

| Format | Extension | Dependencies | Metadata | Size | Speed | Best For |
|--------|-----------|--------------|----------|------|-------|----------|
| **Word2Vec Text** | `.vec`, `.txt` | None | Limited | **~610 MB** | Medium | Compatibility, human-readable |
| **Word2Vec Binary** | `.bin` | gensim | None | **~305 MB** | Fast | Compatibility, compact |
| **PyTorch** | `.pt`, `.pth` | torch | Full | **~305 MB** | Fast | Research, deep learning |
| **NumPy** | `.npz` | None | Full | **~315 MB** | Fast | Scientific Python |
| **HDF5** | `.h5`, `.hdf5` | h5py | Full | **~310 MB** | Medium | Scientific data, large scale |

**Note:** Sizes shown for GloVe-200 (400K words × 200 dims). Text format (.vec) is **~2× larger** than binary formats due to ASCII encoding overhead.

**⚠️ Format Conversion Warning:**
Converting from `.bin` to `.vec` will **DOUBLE your file size** (305 MB → 610 MB), even with quantization! This is format overhead, not quantization overhead. Keep the same format to maintain file size.

---

## 1. Word2Vec Text Format (.vec, .txt)

### Description
Human-readable text format used by Word2Vec and GloVe. First line contains vocabulary size and dimensions, followed by word-vector pairs.

### Advantages
✅ No dependencies (pure Python)  
✅ Human-readable and inspectable  
✅ Universal compatibility  
✅ Can include comments (quantization metadata)  

### Disadvantages
❌ Slower to load than binary formats  
❌ Larger file size than compressed formats  
❌ Limited metadata storage  

### Format Structure
```
400000 300
the 0.418 0.24968 -0.41242 ...
, 0.013441 0.23682 -0.16899 ...
# Quantization: {'base_k': 32, 'method': 'adaptive'}
```

### Usage
```python
# Save
save_embeddings(quantized, words, 'output.vec')

# Load
embeddings, words = load_embeddings('output.vec')
```

```bash
# Command line
python convert_embeddings.py input.vec output.vec
```

### When to Use
- Maximum compatibility across tools
- Need to inspect embeddings manually
- No dependency concerns
- Standard GloVe/Word2Vec pipelines

---

## 2. Word2Vec Binary Format (.bin)

### Description
Compact binary format used by original Word2Vec implementation. Requires gensim library.

### Advantages
✅ Compact file size  
✅ Fast loading  
✅ Compatible with gensim ecosystem  
✅ Industry standard for Word2Vec  

### Disadvantages
❌ Requires gensim library  
❌ Binary (not human-readable)  
❌ No metadata storage  

### Usage
```python
# Save
save_embeddings(quantized, words, 'output.bin')

# Load
embeddings, words = load_embeddings('output.bin')
```

```bash
# Command line
python convert_embeddings.py input.vec output.bin
```

### When to Use
- Working with gensim
- Need compact files
- Standard Word2Vec workflows
- Don't need metadata

**Note:** Metadata (quantization parameters) cannot be stored in binary format. Consider using `.pt` or `.h5` if you need metadata.

---

## 3. PyTorch Format (.pt, .pth)

### Description
PyTorch's native format using `torch.save()`. Stores tensors and arbitrary Python objects with full metadata.

### Advantages
✅ Full metadata support  
✅ Native PyTorch integration  
✅ Fast loading  
✅ Can store complex structures  
✅ Perfect for deep learning workflows  

### Disadvantages
❌ Requires PyTorch library  
❌ Python-specific (not language-agnostic)  

### Stored Data
```python
{
    'embeddings': torch.Tensor,      # The embedding matrix
    'words': List[str],               # Word list
    'vocab_size': int,                # Vocabulary size
    'embedding_dim': int,             # Dimension count
    'quantization_info': dict,        # Full metadata
    'format_version': str             # Version info
}
```

### Usage
```python
# Save
save_embeddings(quantized, words, 'output.pt',
               quantization_info={'base_k': 32, 'method': 'adaptive'})

# Load
embeddings, words = load_embeddings('output.pt')

# Or load with PyTorch directly
data = torch.load('output.pt')
print(data['quantization_info'])  # Access metadata
embeddings = data['embeddings'].numpy()
```

```bash
# Command line
python convert_embeddings.py input.vec output.pt
```

### When to Use
- Research with quantization experiments
- Need full metadata tracking
- PyTorch-based workflows
- Deep learning applications
- **Recommended for research and experimentation**

---

## 4. NumPy Compressed Format (.npz)

### Description
NumPy's compressed archive format storing multiple arrays with metadata. Uses `np.savez_compressed()`.

### Advantages
✅ Full metadata support (as JSON)  
✅ No extra dependencies (NumPy only)  
✅ Compressed automatically  
✅ Native NumPy integration  
✅ Language-agnostic (via NumPy I/O)  

### Disadvantages
❌ Slightly larger than binary formats  

### Stored Data
```python
{
    'embeddings': np.ndarray,         # The embedding matrix
    'words': np.ndarray,              # Word array
    'vocab_size': np.ndarray,         # Vocabulary size
    'embedding_dim': np.ndarray,      # Dimension count
    'quantization_info': str          # JSON metadata
}
```

### Usage
```python
# Save
save_embeddings(quantized, words, 'output.npz',
               quantization_info={'base_k': 32, 'method': 'adaptive'})

# Load
embeddings, words = load_embeddings('output.npz')

# Or load with NumPy directly
data = np.load('output.npz', allow_pickle=True)
import json
quant_info = json.loads(str(data['quantization_info'][0]))
print(quant_info)
embeddings = data['embeddings']
```

```bash
# Command line
python convert_embeddings.py input.vec output.npz
```

### When to Use
- Scientific Python workflows
- Need metadata without PyTorch
- NumPy-based applications
- **Recommended for scientific computing**

---

## 5. HDF5 Format (.h5, .hdf5)

### Description
Hierarchical Data Format used extensively in scientific computing. Stores large numerical datasets efficiently with rich metadata.

### Advantages
✅ Scientific data standard  
✅ Full metadata support (attributes)  
✅ Efficient for large datasets  
✅ Language-agnostic (many language bindings)  
✅ Compressed storage  
✅ Hierarchical organization  

### Disadvantages
❌ Requires h5py library  
❌ Slightly more complex API  

### Stored Data Structure
```
File: output.h5
├── /embeddings [dataset]     # Compressed embedding matrix
├── /words [dataset]          # UTF-8 encoded word strings
└── Attributes:
    ├── vocab_size
    ├── embedding_dim
    ├── format_version
    └── quantization_info (JSON)
```

### Usage
```python
# Save
save_embeddings(quantized, words, 'output.h5',
               quantization_info={'base_k': 32, 'method': 'adaptive'})

# Load
embeddings, words = load_embeddings('output.h5')

# Or load with h5py directly
import h5py
import json

with h5py.File('output.h5', 'r') as f:
    embeddings = f['embeddings'][:]
    words = [w.decode('utf-8') for w in f['words'][:]]
    
    # Access metadata
    quant_info = json.loads(f.attrs['quantization_info'])
    print(f"Vocab: {f.attrs['vocab_size']}")
    print(f"Dims: {f.attrs['embedding_dim']}")
    print(f"Quantization: {quant_info}")
```

```bash
# Command line
python convert_embeddings.py input.vec output.h5
```

### When to Use
- Large-scale scientific data
- Need language-agnostic format
- Integration with scientific tools
- Hierarchical data organization
- **Recommended for large-scale scientific applications**

---

## Format Selection Guide

### Choose **Word2Vec Text (.vec)** if:
- You need maximum compatibility
- You're using standard NLP tools
- You want human-readable files
- Dependencies are a concern

### Choose **Word2Vec Binary (.bin)** if:
- You're working with gensim
- You need compact files
- You don't need metadata
- You want fast loading

### Choose **PyTorch (.pt)** if:
- You're doing research with quantization
- You need full metadata tracking
- You're using PyTorch for other tasks
- You want the most flexible format

### Choose **NumPy (.npz)** if:
- You're doing scientific computing
- You need metadata but not PyTorch
- You're using NumPy-based tools
- You want a self-contained format

### Choose **HDF5 (.h5)** if:
- You're working with very large datasets
- You need language-agnostic format
- You're integrating with scientific tools (MATLAB, R, etc.)
- You want industry-standard scientific format

---

## Format Conversion

You can convert between formats easily:

```bash
# Text → PyTorch
python convert_embeddings.py embeddings.vec embeddings.pt

# PyTorch → NumPy
python convert_embeddings.py embeddings.pt embeddings.npz

# NumPy → HDF5
python convert_embeddings.py embeddings.npz embeddings.h5

# Any → Binary
python convert_embeddings.py any_format.* output.bin
```

Or programmatically:

```python
from adaptive_quantization import load_embeddings, save_embeddings

# Load from any format
embeddings, words = load_embeddings('input.vec')

# Save to different formats
save_embeddings(embeddings, words, 'output.pt')   # PyTorch
save_embeddings(embeddings, words, 'output.npz')  # NumPy
save_embeddings(embeddings, words, 'output.h5')   # HDF5
save_embeddings(embeddings, words, 'output.bin')  # Binary
```

---

## Installation Requirements

### Minimal (Text format only)
```bash
pip install numpy scipy scikit-learn matplotlib
```

### Full Support
```bash
# For all formats
pip install numpy scipy scikit-learn matplotlib gensim torch h5py

# Or individually
pip install gensim      # For .bin support
pip install torch       # For .pt support
pip install h5py        # For .h5 support
```

---

## Best Practices

### For Research
Use **PyTorch (.pt)** format:
```python
save_embeddings(quantized, words, 'experiment_k32.pt',
               quantization_info={
                   'base_k': 32,
                   'method': 'adaptive',
                   'use_lebesgue': True,
                   'experiment_id': 'exp001',
                   'date': '2026-01-20'
               })
```

### For Production
Use **Word2Vec Binary (.bin)** for deployment:
```bash
python convert_embeddings.py research.pt production.bin
```

### For Archival
Use **HDF5 (.h5)** for long-term storage:
```python
save_embeddings(quantized, words, 'archive_2026.h5',
               quantization_info={
                   'base_k': 32,
                   'method': 'adaptive',
                   'corpus': 'Wikipedia 2025',
                   'algorithm_version': '2.0'
               })
```

---

## Troubleshooting

### Missing Dependencies

```python
# Error: "gensim is required for .bin files"
pip install gensim

# Error: "PyTorch is required for .pt files"
pip install torch

# Error: "h5py is required for .h5 files"
pip install h5py
```

### Format Auto-Detection Issues

If auto-detection fails, force the format:

```python
# Force format (ignores extension)
save_embeddings(emb, words, 'weird.ext', format='pytorch')
```

```bash
# Command line
python convert_embeddings.py input.txt output.xyz --format pytorch
```

---

## Performance Comparison

Benchmark on GloVe-300 (400k vocab, 300 dims):

| Format | File Size | Save Time | Load Time | Metadata | Notes |
|--------|-----------|-----------|-----------|----------|-------|
| .vec   | 467 MB    | 45s       | 12s       | Limited  | Text format |
| .bin   | 457 MB    | 15s       | 3s        | None     | Binary, gensim |
| .pt    | 463 MB    | 8s        | 2s        | Full     | PyTorch |
| .npz   | 461 MB    | 12s       | 4s        | Full     | NumPy compressed |
| .h5    | 459 MB    | 10s       | 3s        | Full     | HDF5 |

After quantization (k=32):

| Format | File Size | Theoretical Bits | Save Time | Load Time | Actual Compression |
|--------|-----------|------------------|-----------|-----------|-------------------|
| .vec   | 467 MB    | 5 bits/value     | 6s        | 1.5s      | **None** (32-bit floats) |
| .bin   | 457 MB    | 5 bits/value     | 2s        | 0.4s      | **None** (32-bit floats) |
| .pt    | 463 MB    | 5 bits/value     | 1s        | 0.3s      | **None** (32-bit floats) |
| .npz   | 461 MB    | 5 bits/value     | 1.5s      | 0.5s      | **None** (32-bit floats) |
| .h5    | 459 MB    | 5 bits/value     | 1.2s      | 0.4s      | **None** (32-bit floats) |

### ⚠️ Important Note on File Sizes

**All standard formats store values as 32-bit floats**, even after quantization to k=32 levels:
- ✅ Values are quantized (reduced to k discrete levels)
- ✅ Semantic quality improves (+3-6% STS)
- ❌ File size does NOT automatically reduce
- ❌ Each value still takes 32 bits of storage

**Why?**
- Standard formats (.vec, .bin, .pt, .npz, .h5) all use `float32` arrays
- Quantization reduces the number of **unique values**, not the **storage format**
- To get actual compression, you need custom binary format with bit-level storage

**Theoretical compression** (5 bits/value for k=32):
- 467 MB → ~73 MB would require custom format
- Current implementation: 467 MB → 467 MB (same size, better quality)

**To achieve actual file size reduction:**
```bash
# Option 1: External compression
gzip glove.quantized.vec  # Reduces to ~150-200 MB (2-3× compression)

# Option 2: Custom binary format (not yet implemented)
# Would store: codebook + 5-bit indices → true 6-8× compression

# Option 3: Use specialized vector databases
# FAISS, Milvus, etc. support quantized storage
```

**Conclusion:** Choose format based on workflow needs (metadata, compatibility) rather than file size, since all formats currently have similar sizes after quantization.
