# Adaptive Word Embedding Quantization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Dimension-adaptive quantization for word embeddings achieving compression with quality improvement.**

This repository implements the quantization methods from our research paper demonstrating that optimal quantization occurs at **k=32 (2^5 Walsh function orders)**, achieving up to **8× compression** while **improving semantic quality by +6.3%**.

## 🎯 Key Features

- **Adaptive per-dimension quantization**: Automatically classifies each dimension's distribution type and applies optimal quantization method
- **Multiple quantization strategies**: Uniform (Riemann), Lebesgue (equi-depth), percentile, k-means, robust
- **Walsh function-aware**: Emphasizes powers of 2 (k=16, 32, 64) based on theoretical grounding
- **Comprehensive evaluation**: STS (Semantic Textual Similarity) and SICK (compositional semantics) benchmarks
- **Publication-quality visualizations**: Distribution analysis, before/after comparisons, performance metrics
- **Minimal dependencies**: NumPy, SciPy, scikit-learn, Matplotlib

## 📊 Results at a Glance

| Model | Original STS | k=32 Quantized | Improvement | Compression |
|-------|--------------|----------------|-------------|-------------|
| GloVe-50 | 0.634 | 0.659 (+3.9%) | +3.9% | 6.6× |
| GloVe-200 | 0.564 | 0.590 (+4.5%) | +4.5% | 6.6× |
| GloVe-300 | 0.583 | 0.619 (+6.3%) | +6.3% | 8.4× |

**The compression paradox**: Aggressive quantization simultaneously reduces storage AND enhances semantic quality by removing noise while crystallizing signal structure.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/embedding-quantization.git
cd embedding-quantization

# Install dependencies
pip install numpy scipy scikit-learn matplotlib
```

### Basic Usage

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings

# Load embeddings
embeddings, words = load_embeddings('path/to/embeddings.vec')

# Apply adaptive quantization (k=32 recommended based on research)
quantizer = AdaptiveQuantizer(base_k=32, verbose=True)
quantized = quantizer.quantize(embeddings)

# Print summary
quantizer.print_summary()
```

### Command-Line Usage

```bash
# Evaluate with k=32 (recommended)
python evaluate_quantization.py embeddings.vec --base-k 32

# Visualize quantization effects
python visualize_quantization.py embeddings.vec --base-k 32

# Use Lebesgue quantization for skewed dimensions
python evaluate_quantization.py embeddings.vec --base-k 32 --use-lebesgue
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
├── README.md                    # This file
├── docs/
│   ├── API_REFERENCE.md        # Detailed API documentation
│   ├── EXAMPLES.md             # Extended usage examples
│   ├── THEORY.md               # Walsh function theory
│   └── INSTALLATION.md         # Installation guide
└── examples/
    ├── basic_quantization.py   # Basic usage example
    ├── batch_evaluation.py     # Batch processing
    └── custom_metrics.py       # Custom evaluation metrics
```

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Test on sample embeddings
python evaluate_quantization.py --test-mode
```

## 📊 Embedding Format

The code expects embeddings in `.vec` format (GloVe/Word2Vec style):

```
400000 100
the 0.418 0.24968 -0.41242 ...
, 0.013441 0.23682 -0.16899 ...
. 0.15164 0.30177 -0.16763 ...
```

**First line**: `<vocabulary_size> <dimensions>`  
**Subsequent lines**: `<word> <value1> <value2> ...`

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
