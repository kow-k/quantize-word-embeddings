# API Reference

Complete API documentation for the Adaptive Word Embedding Quantization library.

## Table of Contents

- [adaptive_quantization Module](#adaptive_quantization-module)
  - [AdaptiveQuantizer](#adaptivequantizer)
  - [UniformQuantizer](#uniformquantizer)
  - [DimensionClassifier](#dimensionclassifier)
  - [DimensionProfile](#dimensionprofile)
  - [Utility Functions](#utility-functions)
- [evaluate_quantization Module](#evaluate_quantization-module)
  - [QuantizationEvaluator](#quantizationevaluator)
  - [EmbeddingEvaluator](#embeddingevaluator)
- [visualize_quantization Module](#visualize_quantization-module)
  - [QuantizationVisualizer](#quantizationvisualizer)

---

## adaptive_quantization Module

### AdaptiveQuantizer

Main class for adaptive quantization of word embeddings.

```python
class AdaptiveQuantizer(base_k=20, use_lebesgue=False, verbose=True)
```

**Parameters:**

- **base_k** (int, default=20): Base number of quantization levels
  - 16: Recommended for 100-dimensional embeddings (2^4 Walsh orders)
  - 32: Recommended for most embeddings (2^5 Walsh orders) - **nearly universal optimal**
  - 64: Fine-grained quantization (2^6 Walsh orders)
  - 100-200: Very fine precision, minimal loss
  
  The actual k per dimension may vary based on distribution type:
  - Gaussian: k = base_k
  - Skewed: k = base_k × 1.5
  - Multimodal: k = min(n_modes × base_k × 0.6, base_k × 2)
  - Heavy-tailed: k = base_k × 1.25

- **use_lebesgue** (bool, default=False): Use true Lebesgue (equi-depth) quantization for skewed dimensions
  - False: Use percentile method (hybrid approach, faster)
  - True: Use true equi-depth with equal counts per bin (more accurate for high skew)

- **verbose** (bool, default=True): Print progress messages during analysis and quantization

**Attributes:**

- **dimension_profiles** (List[DimensionProfile]): List of profiles for each dimension after analysis
- **base_k** (int): Base quantization level
- **use_lebesgue** (bool): Whether to use Lebesgue quantization
- **verbose** (bool): Verbosity setting

**Methods:**

#### `analyze_dimensions(embeddings)`

Analyze all dimensions and create distribution profiles.

```python
profiles = quantizer.analyze_dimensions(embeddings)
```

**Parameters:**
- **embeddings** (np.ndarray): Input embeddings, shape (n_words, n_dims)

**Returns:**
- List[DimensionProfile]: List of dimension profiles with classification and settings

**Example:**
```python
quantizer = AdaptiveQuantizer(base_k=32)
profiles = quantizer.analyze_dimensions(embeddings)

for profile in profiles[:5]:
    print(f"Dim {profile.dim_id}: {profile.dist_type}, k={profile.optimal_k}")
```

#### `quantize(embeddings)`

Apply adaptive quantization to embeddings.

```python
quantized = quantizer.quantize(embeddings)
```

**Parameters:**
- **embeddings** (np.ndarray): Input embeddings, shape (n_words, n_dims)

**Returns:**
- np.ndarray: Quantized embeddings with same shape as input

**Example:**
```python
quantizer = AdaptiveQuantizer(base_k=32, use_lebesgue=True)
quantized = quantizer.quantize(embeddings)

# Check compression
original_unique = [len(np.unique(embeddings[:, i])) for i in range(embeddings.shape[1])]
quantized_unique = [len(np.unique(quantized[:, i])) for i in range(quantized.shape[1])]
print(f"Average unique values: {np.mean(original_unique):.0f} → {np.mean(quantized_unique):.0f}")
```

#### `quantize_dimension(values, profile)`

Quantize a single dimension according to its profile.

```python
quantized_dim = quantizer.quantize_dimension(values, profile)
```

**Parameters:**
- **values** (np.ndarray): 1D array of values for this dimension
- **profile** (DimensionProfile): Profile with quantization settings

**Returns:**
- np.ndarray: Quantized values for this dimension

#### `get_summary()`

Get summary statistics of dimension profiles.

```python
summary = quantizer.get_summary()
```

**Returns:**
- Dict with keys:
  - 'n_dimensions': Total number of dimensions
  - 'distribution_types': Counter of distribution types
  - 'quantization_methods': Counter of quantization methods
  - 'k_range': (min_k, max_k) tuple
  - 'k_mean': Mean k value across dimensions
  - 'k_median': Median k value

**Example:**
```python
summary = quantizer.get_summary()
print(f"Dimensions: {summary['n_dimensions']}")
print(f"Distribution types: {summary['distribution_types']}")
print(f"k range: {summary['k_range'][0]}-{summary['k_range'][1]}")
print(f"k mean: {summary['k_mean']:.1f}")
```

#### `print_summary()`

Print formatted summary of dimension analysis.

```python
quantizer.print_summary()
```

**Example output:**
```
Dimension Analysis Summary:
  Total dimensions: 100
  Distribution types:
    gaussian: 89 (89.0%)
    right_skewed: 7 (7.0%)
    left_skewed: 3 (3.0%)
    multimodal: 1 (1.0%)
  Quantization methods:
    uniform: 89 (89.0%)
    lebesgue: 10 (10.0%)
    kmeans: 1 (1.0%)
  k range: [32, 48]
  k mean: 33.2
  k median: 32.0
```

---

### UniformQuantizer

Simple uniform (Riemann-style) quantizer for baseline comparison.

```python
class UniformQuantizer(k=20, verbose=True)
```

**Parameters:**
- **k** (int, default=20): Number of quantization levels (same for all dimensions)
- **verbose** (bool, default=True): Print progress messages

**Methods:**

#### `quantize(embeddings)`

Apply uniform quantization to all dimensions.

```python
uniform_quantizer = UniformQuantizer(k=32)
quantized = uniform_quantizer.quantize(embeddings)
```

**Parameters:**
- **embeddings** (np.ndarray): Input embeddings

**Returns:**
- np.ndarray: Uniformly quantized embeddings

---

### DimensionClassifier

Static methods for classifying dimension distribution types.

```python
class DimensionClassifier
```

**Static Methods:**

#### `detect_modes(values, prominence=0.1)`

Detect number of modes in distribution using peak detection.

```python
n_modes = DimensionClassifier.detect_modes(dimension_values)
```

**Parameters:**
- **values** (np.ndarray): 1D array of values
- **prominence** (float, default=0.1): Minimum prominence for peak detection

**Returns:**
- int: Number of detected modes

#### `classify_dimension(values, dim_id, base_k=20, use_lebesgue=False)`

Classify a dimension's distribution type and determine optimal settings.

```python
profile = DimensionClassifier.classify_dimension(values, dim_id=0, base_k=32)
```

**Parameters:**
- **values** (np.ndarray): 1D array of values for this dimension
- **dim_id** (int): Dimension index
- **base_k** (int, default=20): Base quantization level
- **use_lebesgue** (bool, default=False): Use Lebesgue for skewed distributions

**Returns:**
- DimensionProfile: Profile with classification and quantization settings

**Classification Logic:**
- **Gaussian**: Shapiro-Wilk p > 0.05, |skewness| < 0.2, |kurtosis| < 0.5
  - Method: uniform, k = base_k
- **Right/Left Skewed**: |skewness| > 0.5
  - Method: percentile or lebesgue, k = base_k × 1.5
- **Multimodal**: n_modes ≥ 2
  - Method: kmeans, k = min(n_modes × base_k × 0.6, base_k × 2)
- **Heavy-tailed**: |kurtosis| > 3.0
  - Method: robust, k = base_k × 1.25
- **Other**: Default case
  - Method: uniform, k = base_k

---

### DimensionProfile

Data class storing dimension distribution profile and quantization settings.

```python
@dataclass
class DimensionProfile:
    dim_id: int                    # Dimension index
    dist_type: str                 # Distribution type classification
    skewness: float                # Skewness value
    kurtosis: float                # Kurtosis value
    n_modes: int                   # Number of detected modes
    shapiro_p: float               # Shapiro-Wilk test p-value
    optimal_k: int                 # Optimal k for this dimension
    quantization_method: str       # Selected quantization method
```

**Example:**
```python
profile = DimensionProfile(
    dim_id=0,
    dist_type='gaussian',
    skewness=0.05,
    kurtosis=-0.2,
    n_modes=1,
    shapiro_p=0.42,
    optimal_k=32,
    quantization_method='uniform'
)
print(profile)
# Output: Dim 0: gaussian (skew=0.05, kurt=-0.20, k=32, method=uniform)
```

---

### Utility Functions

#### `load_embeddings(filepath)`

Load embeddings from .vec format file (Word2Vec/GloVe style).

```python
from adaptive_quantization import load_embeddings

embeddings, words = load_embeddings('glove.6B.100d.vec')
print(f"Loaded {len(words)} words, {embeddings.shape[1]} dimensions")
```

**Parameters:**
- **filepath** (str): Path to .vec format embedding file

**Returns:**
- Tuple[np.ndarray, List[str]]: (embeddings array, list of words)

**File format expected:**
```
<vocab_size> <dimensions>
word1 val1 val2 val3 ...
word2 val1 val2 val3 ...
```

#### `save_embeddings(filepath, embeddings, words)`

Save embeddings to .vec format file.

```python
from adaptive_quantization import save_embeddings

save_embeddings('quantized.vec', quantized_embeddings, words)
```

**Parameters:**
- **filepath** (str): Output file path
- **embeddings** (np.ndarray): Embeddings to save
- **words** (List[str]): List of words

---

## evaluate_quantization Module

### QuantizationEvaluator

High-level evaluator for comparing quantization strategies.

```python
class QuantizationEvaluator(embedding_path)
```

**Parameters:**
- **embedding_path** (str): Path to embedding file (.vec format)

**Methods:**

#### `compare_methods(base_k=20, use_lebesgue=False)`

Compare three quantization strategies: original, uniform, adaptive.

```python
evaluator = QuantizationEvaluator('embeddings.vec')
results = evaluator.compare_methods(base_k=32, use_lebesgue=True)
```

**Parameters:**
- **base_k** (int, default=20): Base quantization level
- **use_lebesgue** (bool, default=False): Use Lebesgue for skewed dimensions

**Returns:**
- Dict with keys 'original', 'uniform', 'adaptive', each containing:
  - 'sts': STS correlation score
  - 'sick': SICK correlation score
  - 'embeddings': Quantized embeddings array
  - 'method': Method name

**Example:**
```python
results = evaluator.compare_methods(base_k=32)

print(f"Original:  STS={results['original']['sts']:.4f}")
print(f"Uniform:   STS={results['uniform']['sts']:.4f} "
      f"({(results['uniform']['sts']-results['original']['sts'])*100:+.1f}%)")
print(f"Adaptive:  STS={results['adaptive']['sts']:.4f} "
      f"({(results['adaptive']['sts']-results['original']['sts'])*100:+.1f}%)")
```

#### `print_results(results)`

Print formatted comparison results.

```python
evaluator.print_results(results)
```

**Example output:**
```
=== Quantization Method Comparison ===

Original (no quantization):
  STS:  0.5640
  SICK: 0.8050

Uniform (k=32):
  STS:  0.5842 (+3.6%)
  SICK: 0.8102 (+0.6%)

Adaptive (base_k=32):
  STS:  0.5895 (+4.5%)
  SICK: 0.8013 (-0.5%)
```

---

### EmbeddingEvaluator

Low-level evaluator for computing semantic similarity metrics.

```python
class EmbeddingEvaluator()
```

**Methods:**

#### `evaluate_sts(embeddings, word_to_idx, sts_file=None)`

Evaluate on Semantic Textual Similarity benchmark.

```python
evaluator = EmbeddingEvaluator()
sts_score = evaluator.evaluate_sts(embeddings, word_to_idx)
print(f"STS Spearman ρ: {sts_score:.4f}")
```

**Parameters:**
- **embeddings** (np.ndarray): Embedding matrix
- **word_to_idx** (Dict[str, int]): Word to index mapping
- **sts_file** (str, optional): Path to custom STS data file

**Returns:**
- float: Spearman correlation with human judgments

**Details:**
- Uses built-in STS sentence pairs if sts_file not provided
- Computes sentence embeddings by averaging word vectors
- Returns Spearman ρ between predicted similarities and human ratings

#### `evaluate_sick(embeddings, word_to_idx)`

Evaluate on SICK (Sentences Involving Compositional Knowledge).

```python
sick_score = evaluator.evaluate_sick(embeddings, word_to_idx)
print(f"SICK Spearman ρ: {sick_score:.4f}")
```

**Parameters:**
- **embeddings** (np.ndarray): Embedding matrix
- **word_to_idx** (Dict[str, int]): Word to index mapping

**Returns:**
- float: Spearman correlation with human relatedness judgments

#### `cosine_similarity(v1, v2)`

Compute cosine similarity between two vectors.

```python
sim = evaluator.cosine_similarity(vec1, vec2)
```

**Parameters:**
- **v1** (np.ndarray): First vector
- **v2** (np.ndarray): Second vector

**Returns:**
- float: Cosine similarity (-1 to 1), or None if invalid input

#### `sentence_vector(sentence, embeddings, word_to_idx)`

Get sentence embedding by averaging word vectors.

```python
sent_vec = evaluator.sentence_vector("The dog runs.", embeddings, word_to_idx)
```

**Parameters:**
- **sentence** (str): Input sentence
- **embeddings** (np.ndarray): Embedding matrix
- **word_to_idx** (Dict[str, int]): Word to index mapping

**Returns:**
- np.ndarray: Averaged sentence vector, or None if no words found

---

## visualize_quantization Module

### QuantizationVisualizer

Create publication-quality visualizations.

```python
class QuantizationVisualizer(figsize_base=(12, 8))
```

**Parameters:**
- **figsize_base** (tuple, default=(12, 8)): Base figure size for plots

**Attributes:**
- **colors** (dict): Color scheme for different distribution types

**Methods:**

#### `plot_dimension_types(quantizer, save_path=None)`

Plot distribution of dimension types and quantization methods.

```python
viz = QuantizationVisualizer()
fig = viz.plot_dimension_types(quantizer, save_path='dimension_types.png')
```

**Parameters:**
- **quantizer** (AdaptiveQuantizer): Fitted quantizer with dimension profiles
- **save_path** (str, optional): Path to save figure

**Returns:**
- matplotlib.figure.Figure: Generated figure

**Creates:**
- Pie chart of distribution types
- Bar chart of quantization method selection

#### `plot_dimension_profiles(quantizer, n_samples=20, save_path=None)`

Plot sample dimension profiles showing classification details.

```python
viz.plot_dimension_profiles(quantizer, n_samples=20, save_path='profiles.png')
```

**Parameters:**
- **quantizer** (AdaptiveQuantizer): Fitted quantizer
- **n_samples** (int, default=20): Number of dimensions to sample
- **save_path** (str, optional): Path to save figure

**Returns:**
- matplotlib.figure.Figure: Grid of dimension profile cards

#### `plot_before_after_distributions(original, quantized, n_samples=9, save_path=None)`

Plot before/after histograms for sample dimensions.

```python
viz.plot_before_after_distributions(
    embeddings, quantized, 
    n_samples=9, 
    save_path='before_after.png'
)
```

**Parameters:**
- **original** (np.ndarray): Original embeddings
- **quantized** (np.ndarray): Quantized embeddings
- **n_samples** (int, default=9): Number of dimensions to show
- **save_path** (str, optional): Path to save figure

**Returns:**
- matplotlib.figure.Figure: 3×3 grid of histogram comparisons

#### `plot_performance_comparison(results, save_path=None)`

Plot performance comparison across quantization methods.

```python
viz.plot_performance_comparison(results, save_path='performance.png')
```

**Parameters:**
- **results** (Dict): Results from QuantizationEvaluator.compare_methods()
- **save_path** (str, optional): Path to save figure

**Returns:**
- matplotlib.figure.Figure: Bar chart with performance metrics

#### `create_comparison_figure(skipgram_results, cbow_results, save_path=None)`

Create side-by-side comparison for different architectures.

```python
viz.create_comparison_figure(
    skipgram_results, 
    cbow_results,
    save_path='architecture_comparison.png'
)
```

**Parameters:**
- **skipgram_results** (Dict): Results for Skip-gram embeddings
- **cbow_results** (Dict): Results for CBOW embeddings
- **save_path** (str, optional): Path to save figure

**Returns:**
- matplotlib.figure.Figure: Side-by-side comparison

---

## Complete Usage Example

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings
from evaluate_quantization import QuantizationEvaluator
from visualize_quantization import QuantizationVisualizer

# 1. Load embeddings
embeddings, words = load_embeddings('glove.6B.100d.vec')
print(f"Loaded {len(words)} words, {embeddings.shape[1]} dimensions")

# 2. Apply adaptive quantization
quantizer = AdaptiveQuantizer(
    base_k=32,           # Recommended: k=32 (2^5) for most embeddings
    use_lebesgue=True,   # Use equi-depth for skewed dimensions
    verbose=True
)
quantized = quantizer.quantize(embeddings)

# 3. Analyze results
summary = quantizer.get_summary()
quantizer.print_summary()

# 4. Evaluate performance
evaluator = QuantizationEvaluator('glove.6B.100d.vec')
results = evaluator.compare_methods(base_k=32, use_lebesgue=True)
evaluator.print_results(results)

# 5. Create visualizations
viz = QuantizationVisualizer()
viz.plot_dimension_types(quantizer, save_path='types.png')
viz.plot_before_after_distributions(embeddings, quantized, save_path='comparison.png')
viz.plot_performance_comparison(results, save_path='performance.png')

print("\nComplete! Check generated PNG files.")
```

---

## Error Handling

All functions handle common errors gracefully:

- **Invalid file paths**: Raises FileNotFoundError with helpful message
- **Malformed .vec files**: Raises ValueError with line number
- **Empty embeddings**: Raises ValueError
- **Dimension mismatches**: Raises ValueError with expected vs actual shapes
- **Missing words in vocabulary**: Returns None for similarity, skips in sentence averaging

**Example:**
```python
try:
    embeddings, words = load_embeddings('nonexistent.vec')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please provide valid embedding file path.")
```

---

## Performance Considerations

### Memory Usage

- **Original embeddings**: 400k words × 300 dims × 4 bytes = ~467 MB
- **Quantized (k=32)**: Same shape, but values from smaller set
- **Analysis overhead**: Temporary arrays for statistics (~2× embedding size)

**Tip**: For large embeddings (>1M words), consider processing in batches or using memory-mapped arrays.

### Speed

- **Analysis**: ~1-2 seconds per 100 dimensions (depends on vocabulary size)
- **Quantization**: ~0.5-1 seconds per 100 dimensions
- **Evaluation (STS+SICK)**: ~5-10 seconds on built-in datasets

**Tip**: Set `verbose=False` to disable progress messages for faster execution.

### Parallelization

Current implementation is single-threaded. For large-scale processing:

```python
# Process dimensions in parallel (example)
from multiprocessing import Pool

def quantize_dimension_parallel(args):
    dim, values, profile = args
    quantizer = AdaptiveQuantizer(base_k=32)
    return quantizer.quantize_dimension(values, profile)

with Pool(8) as pool:
    results = pool.map(quantize_dimension_parallel, dimension_args)
```

---

## Version History

- **v1.0.0** (2026-01): Initial release
  - Adaptive dimension-wise quantization
  - Walsh function-aware k selection (emphasis on powers of 2)
  - STS and SICK evaluation
  - Publication-quality visualizations

---

For more examples and tutorials, see [EXAMPLES.md](EXAMPLES.md).
