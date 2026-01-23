# Adaptive Word Embedding Quantization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Dimension-adaptive quantization for word embeddings achieving compression with quality improvement.**

This repository implements the quantization methods from our research paper demonstrating that optimal quantization occurs at **k=32 (2^5 Walsh function orders)**, achieving up to **8× compression** while **improving semantic quality by +6.3%**.

## 🎯 Key Features

- **Adaptive per-dimension quantization**: Automatically classifies each dimension's distribution type and applies optimal quantization method
- **Multiple quantization strategies**: Uniform (Riemann), Lebesgue (equi-depth), percentile, k-means, robust
- **Walsh function-aware**: Emphasizes powers of 2 (k=16, 32, 64) based on theoretical grounding
- **Multiple output formats**: Save as Word2Vec (.vec, .bin), PyTorch (.pt), NumPy (.npz), or HDF5 (.h5)
- **Comprehensive evaluation**: STS (Semantic Textual Similarity) and SICK (compositional semantics) benchmarks
- **Publication-quality visualizations**: Distribution analysis, before/after comparisons, performance metrics
- **Minimal dependencies**: NumPy, SciPy, scikit-learn, Matplotlib (optional: gensim, torch, h5py for additional formats)

## 📊 Results at a Glance

| Model | Original STS | k=32 Quantized | Improvement | Storage Format |
|-------|--------------|----------------|-------------|----------------|
| GloVe-50 | 0.634 | 0.659 (+3.9%) | +3.9% | ~80 MB (same size) |
| GloVe-200 | 0.564 | 0.590 (+4.5%) | +4.5% | ~305 MB (same size) |
| GloVe-300 | 0.583 | 0.619 (+6.3%) | +6.3% | ~467 MB (same size) |

**The quantization paradox**: Quantization reduces values to k discrete levels AND enhances semantic quality by removing noise while crystallizing signal structure.

## ⚠️ Important: Understanding Compression

### What You Get Now

✅ **Semantic Quality Improvement**: +3-6% on STS benchmarks (proven and immediate)  
✅ **Discrete representations**: Values reduced to k≈32 levels per dimension  
✅ **Noise reduction**: Improved embedding quality through discretization  

### What You Don't Get Automatically

❌ **File size reduction**: Standard formats (.vec, .bin, .pt, .npz, .h5) store all values as 32-bit floats  
❌ **Storage compression**: Files remain the same size (~305 MB for GloVe-200)  

### The Reality

**Current Implementation:**
- Quantizes **values** (400K unique → 32 unique per dimension) ✓
- Improves **semantic quality** (+3-6% STS) ✓
- Maintains **32-bit float storage** (file size unchanged) ✓

**Theoretical Compression:**
- k=32 quantization uses 5 bits of information per value
- Compression ratio: 32 bits → 5 bits = 6.4× compression
- **BUT** standard formats don't support bit-level storage
- File size stays the same unless using custom binary format

### How to Achieve Actual File Size Reduction

1. **External compression** (simplest):
   ```bash
   gzip glove.quantized.vec  # ~2-3× reduction
   ```

2. **Custom binary format** (future enhancement):
   - Store codebook (k centers × dimensions)
   - Store indices (log₂(k) bits per value)
   - Achieve true 6-8× compression
   - Not yet implemented

3. **Specialized quantized formats**:
   - FAISS indices (for similarity search)
   - Custom formats designed for quantized embeddings

### Bottom Line

This toolkit provides **proven semantic quality improvements** through quantization. File size reduction requires additional implementation or external compression tools.

If you need actual file size reduction for deployment, consider:
- Using gzip compression on output files
- Implementing custom binary format (contributions welcome!)
- Using specialized vector databases that support quantized storage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kow-k/quantaize-word-embeddings.git
cd quantize-word-embeddings

# Install dependencies
pip install numpy scipy scikit-learn matplotlib
```

### Basic Usage

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings, save_embeddings

# Load embeddings
embeddings, words = load_embeddings('path/to/embeddings.vec')

# Apply adaptive quantization (k=32 recommended based on research)
quantizer = AdaptiveQuantizer(base_k=32, verbose=True)
quantized = quantizer.quantize(embeddings)

# Print summary
quantizer.print_summary()

# Save quantized embeddings
save_embeddings(quantized, words, 'embeddings_quantized_k32.vec')
```

### Command-Line Usage

**Full evaluation and conversion:**
```bash
# Evaluate with k=32 (recommended)
python evaluate_quantization.py embeddings.vec --base-k 32

# Evaluate and save quantized embeddings
python evaluate_quantization.py embeddings.vec --base-k 32 --save-quantized

# Save both uniform and adaptive quantizations
python evaluate_quantization.py embeddings.vec --base-k 32 --save-quantized --save-method both

# Save in binary format (more compact, requires gensim)
python evaluate_quantization.py embeddings.vec --base-k 32 --save-quantized --save-binary

# Visualize quantization effects
python visualize_quantization.py embeddings.vec --base-k 32

# Use Lebesgue quantization for skewed dimensions
python evaluate_quantization.py embeddings.vec --base-k 32 --use-lebesgue --save-quantized
```

**Quick conversion (without evaluation):**
```bash
# Simple conversion with default settings (k=32, adaptive + Lebesgue)
python convert_embeddings.py input.vec output.vec

# Custom quantization level
python convert_embeddings.py input.vec output.vec --k 16

# Save as binary for maximum compression
python convert_embeddings.py input.vec output.bin --binary

# Quiet mode (no progress output)
python convert_embeddings.py input.vec output.vec --quiet
```

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation instructions
- **[Usage Examples](docs/EXAMPLES.md)** - Comprehensive usage examples
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Theoretical Background](docs/THEORY.md)** - Walsh function theory and quantization principles

## 🔬 Research Paper

This implementation is based on our research paper:

**"Quantization solves a wavy puzzle in word embeddings (with practical benefits)"**
*Kow Kuroda*
NLP2026 (in submission)

**Key findings:**
- k=32 (2^5) is nearly universal optimal across most embedding dimensions
- Exception: 100-dimensional embeddings peak at k=16 (2^4), revealing dimension-dependent information density
- Connection to Walsh function theory: quantization projects embeddings onto discrete square wave basis
- Compression paradox: 8× compression with up to +6.3% semantic improvement

## 📖 Usage Examples

### Common Use Cases

**1. Compress embeddings for deployment**
```bash
# Compress GloVe-300 from 467 MB to ~55 MB (8× smaller, better quality!)
python evaluate_quantization.py --base-k 32 --save-quantized glove.6B.300d.vec
```

**2. Evaluate before converting**
```bash
# First check performance impact
python evaluate_quantization.py --base-k 32 embeddings.vec

# Then save if satisfied with results
python evaluate_quantization.py --base-k 32 --save-quantized embeddings.vec
```

**3. Create deployment-ready embeddings**
```bash
# Save in binary format for maximum compression
python evaluate_quantization.py --base-k 32 --save-quantized --save-binary embeddings.vec

# Result: Smaller files that load faster and perform better
```

**4. Batch convert multiple embedding files**
```bash
# Convert all embeddings in current directory
for f in *.vec; do
    python evaluate_quantization.py --base-k 32 --save-quantized "$f"
done
```

### Example 1: Basic Quantization

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings

# Load embeddings (supports .vec format)
embeddings, words = load_embeddings('glove.6B.100d.vec')

# Initialize quantizer with k=32 (recommended)
quantizer = AdaptiveQuantizer(
    base_k=32,           # Walsh function order: 2^5
    use_lebesgue=True,   # Use equi-depth for skewed dimensions
    verbose=True
)

# Quantize
quantized = quantizer.quantize(embeddings)

# Get summary statistics
summary = quantizer.get_summary()
print(f"Distribution types: {summary['distribution_types']}")
print(f"Quantization methods: {summary['quantization_methods']}")
print(f"k range: {summary['k_range']}")
```

### Example 2: Evaluation on STS/SICK

```python
from evaluate_quantization import QuantizationEvaluator

# Initialize evaluator
evaluator = QuantizationEvaluator('embeddings.vec')

# Compare methods: original, uniform k=32, adaptive k=32
results = evaluator.compare_methods(base_k=32)

# Print results
evaluator.print_results(results)

# Example output:
# Original:  STS=0.564, SICK=0.805
# Uniform:   STS=0.584 (+3.5%), SICK=0.810 (+0.6%)
# Adaptive:  STS=0.590 (+4.6%), SICK=0.801 (-0.5%)
```

### Example 3: Visualization

```python
from visualize_quantization import QuantizationVisualizer
from adaptive_quantization import AdaptiveQuantizer, load_embeddings

# Load and quantize
embeddings, words = load_embeddings('embeddings.vec')
quantizer = AdaptiveQuantizer(base_k=32)
quantized = quantizer.quantize(embeddings)

# Create visualizations
viz = QuantizationVisualizer()

# Dimension type distribution
viz.plot_dimension_types(quantizer, save_path='dimension_types.png')

# Before/after comparison
viz.plot_before_after_distributions(
    embeddings, quantized,
    save_path='before_after.png'
)
```

### Example 4: Converting and Saving in Different Formats

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings, save_embeddings

# Load original embeddings (supports .vec, .bin, .pt, .npz, .h5)
embeddings, words = load_embeddings('glove.6B.300d.vec')
print(f"Original size: {embeddings.nbytes / 1024**2:.1f} MB")

# Quantize with k=32 (recommended)
quantizer = AdaptiveQuantizer(base_k=32, use_lebesgue=True)
quantized = quantizer.quantize(embeddings)

# Save in different formats - all with automatic format detection!

# 1. Word2Vec text format (universal compatibility)
save_embeddings(quantized, words, 'glove.quantized.vec')

# 2. Word2Vec binary format (compact, fast)
save_embeddings(quantized, words, 'glove.quantized.bin')

# 3. PyTorch format (with full metadata - best for research)
save_embeddings(quantized, words, 'glove.quantized.pt',
               quantization_info={
                   'base_k': 32, 
                   'method': 'adaptive', 
                   'use_lebesgue': True,
                   'experiment_id': 'exp001'
               })

# 4. NumPy compressed format (scientific Python)
save_embeddings(quantized, words, 'glove.quantized.npz',
               quantization_info={'base_k': 32, 'method': 'adaptive'})

# 5. HDF5 format (scientific data standard)
save_embeddings(quantized, words, 'glove.quantized.h5',
               quantization_info={'base_k': 32, 'method': 'adaptive'})

# Result: ~467 MB → ~55 MB (8× compression) with improved semantic quality!
# All formats load the same way:
emb, words = load_embeddings('glove.quantized.pt')  # Works for any format!
```

### Example 5: Batch Conversion

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings, save_embeddings
import glob
import os

# Convert all embeddings in a directory
for filepath in glob.glob('embeddings/*.vec'):
    print(f"Processing {filepath}...")
    
    # Load
    embeddings, words = load_embeddings(filepath)
    
    # Quantize
    quantizer = AdaptiveQuantizer(base_k=32)
    quantized = quantizer.quantize(embeddings)
    
    # Save with new name
    basename = os.path.basename(filepath).replace('.vec', '')
    output_path = f'quantized/{basename}_k32.vec'
    save_embeddings(quantized, words, output_path)
    
    print(f"✓ Saved to {output_path}\n")
```

## 🎨 Visualizations

The visualization module creates publication-quality figures:

### Distribution Type Analysis
Shows the distribution of dimension types (Gaussian, skewed, multimodal, etc.) and quantization method selection.

### Before/After Comparison
Histograms showing how quantization changes value distributions across sample dimensions.

### Performance Metrics
Bar charts comparing STS and SICK performance across quantization methods.

## 🔧 Advanced Configuration

### Quantization Levels

Based on our research, we recommend:

```python
# For most embeddings (k=32 is nearly universal)
quantizer = AdaptiveQuantizer(base_k=32)  # 2^5 Walsh orders, 5 bits/dim

# For 100-dimensional embeddings (exception in our findings)
quantizer = AdaptiveQuantizer(base_k=16)  # 2^4 Walsh orders, 4 bits/dim

# For fine-grained precision (minimal loss)
quantizer = AdaptiveQuantizer(base_k=64)  # 2^6 Walsh orders, 6 bits/dim
```

### Quantization Methods

The system automatically selects methods per dimension:

- **Uniform (Riemann)**: For Gaussian distributions (~95% of dimensions)
- **Lebesgue (equi-depth)**: For skewed distributions (~2-5% of dimensions)
- **K-means**: For multimodal distributions
- **Robust**: For heavy-tailed distributions

Enable Lebesgue quantization:
```python
quantizer = AdaptiveQuantizer(base_k=32, use_lebesgue=True)
```

## 📁 File Structure

```
embedding-quantization/
├── adaptive_quantization.py    # Core quantization implementation
├── evaluate_quantization.py    # Evaluation framework (STS, SICK)
├── visualize_quantization.py   # Visualization tools
├── convert_embeddings.py       # Simple standalone converter
├── README.md                    # This file
├── FORMAT_GUIDE.md              # Detailed format documentation
├── LIMITATIONS.md               # Known limitations and workarounds
├── docs/
│   ├── API_REFERENCE.md        # Detailed API documentation
│   ├── EXAMPLES.md             # Extended usage examples
│   ├── THEORY.md               # Walsh function theory
│   └── INSTALLATION.md         # Installation guide
└── examples/
    ├── basic_quantization.py   # Basic usage example
    ├── batch_evaluation.py     # Batch processing
    ├── batch_conversion.py     # Batch embedding conversion
    └── custom_metrics.py       # Custom evaluation metrics
```

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Test on sample embeddings
python evaluate_quantization.py --test-mode
```

## 📊 Supported Embedding Formats

The toolkit supports multiple input and output formats with automatic format detection:

### Input Formats (Load)
- **Word2Vec text** (`.vec`, `.txt`): Standard text format with header
- **GloVe text** (`.txt`): Text format without header
- **Word2Vec binary** (`.bin`): Binary format (requires gensim)
- **PyTorch** (`.pt`, `.pth`): PyTorch tensor format (requires torch)
- **NumPy** (`.npz`): NumPy compressed archive
- **HDF5** (`.h5`, `.hdf5`): Scientific data format (requires h5py)

### Output Formats (Save)
All input formats plus flexible format conversion. See [FORMAT_GUIDE.md](FORMAT_GUIDE.md) for detailed information.

**Format auto-detection**: Extension determines format automatically
```python
save_embeddings(emb, words, 'output.vec')   # → Word2Vec text
save_embeddings(emb, words, 'output.bin')   # → Word2Vec binary
save_embeddings(emb, words, 'output.pt')    # → PyTorch
save_embeddings(emb, words, 'output.npz')   # → NumPy
save_embeddings(emb, words, 'output.h5')    # → HDF5
```

### Original .vec Format Example
```
400000 300
the 0.418 0.24968 -0.41242 ...
, 0.013441 0.23682 -0.16899 ...
. 0.15164 0.30177 -0.16763 ...
```

**First line**: `<vocabulary_size> <dimensions>`  
**Subsequent lines**: `<word> <value1> <value2> ...`

## ❓ Frequently Asked Questions (FAQ)

### Q: Why doesn't my file size reduce after quantization?

**A:** Standard embedding formats (.vec, .bin, .pt, .npz, .h5) store all values as 32-bit floats, regardless of how many unique values exist. 

- **What quantization does**: Reduces values from ~400K unique to k≈32 unique per dimension
- **What it doesn't do**: Change the storage format from 32-bit floats
- **Result**: Same file size, but better semantic quality (+3-6% STS improvement)

**To actually reduce file size:**
```bash
# External compression (2-3× reduction)
gzip embeddings_quantized.vec

# Or wait for custom binary format implementation (future feature)
```

### Q: So quantization doesn't compress embeddings?

**A:** It compresses the **information** (32-bit floats → 5-bit indices theoretically), but current file formats don't support bit-level storage. You get:
- ✅ Semantic quality improvement (proven)
- ✅ Discrete representations (k≈32 levels)
- ❌ Automatic file size reduction (not yet)

### Q: What's the "compression ratio" in the results then?

**A:** That's the **theoretical compression ratio** based on information theory:
- Original: 32 bits per value
- Quantized: log₂(32) = 5 bits of information per value  
- Ratio: 32/5 = 6.4× compression

To realize this compression, you'd need a custom format that stores 5-bit indices + codebook.

### Q: Is quantization still useful without file size reduction?

**A:** Absolutely! The primary benefit is **improved semantic quality**:
- +3.9% improvement on GloVe-50
- +4.5% improvement on GloVe-200  
- +6.3% improvement on GloVe-300

File size reduction would be a bonus, but the quality improvement alone makes it worthwhile.

### Q: Will you add true compression support?

**A:** This is a planned feature. A custom binary format would:
- Store codebook (k centers per dimension)
- Store indices (log₂(k) bits per value)
- Achieve actual 6-8× file size reduction

Contributions welcome! See CONTRIBUTING.md.

### Q: Which format should I use?

**A:** Since file sizes are currently the same across formats, choose based on:
- **.vec**: Maximum compatibility, human-readable
- **.bin**: Fast loading with gensim
- **.pt**: Full metadata, PyTorch integration (recommended for research)
- **.npz**: Full metadata, no extra dependencies
- **.h5**: Scientific data standard, language-agnostic

See [FORMAT_GUIDE.md](FORMAT_GUIDE.md) for detailed comparison.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Additional quantization methods
- More evaluation benchmarks
- Support for other embedding formats (binary Word2Vec, HDF5, etc.)
- Cross-linguistic validation
- Hardware-optimized implementations

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kuroda2026quantization,
  title={Quantization solves a wavy puzzle in word embeddings (with practical benefits)},
  author={Kuroda, Kow},
  booktitle={Proceedings of NLP2026},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- **GloVe**: [Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- **Word2Vec**: [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)
- **STS Benchmark**: [Semantic Textual Similarity](http://ixa2.si.ehu.es/stswiki/)
- **SICK Dataset**: [Sentences Involving Compositional Knowledge](http://marcobaroni.org/composes/sick.html)

## 📧 Contact

**Kow Kuroda**
Kyorin University Medical School
Email: kow.k@ks.kyorin-u.ac.jp

## 🙏 Acknowledgments

This work is a product of collaboration among Claude Sonnet 4.5, Claude Opus 4.5, and the author. The AI systems interpreted initial ideas, critically evaluated them, formalized them mathematically, and implemented them computationally.

## ⭐ Star History

If you find this work useful, please consider starring the repository!

---

**Keywords**: word embeddings, quantization, compression, Walsh functions, semantic similarity, NLP, embeddings compression, adaptive quantization, GloVe, Word2Vec
