# Usage Examples

Comprehensive examples demonstrating the adaptive quantization library.

## Table of Contents

1. [Basic Quantization](#1-basic-quantization)
2. [Walsh Function-Based k Selection](#2-walsh-function-based-k-selection)
3. [Comparative Evaluation](#3-comparative-evaluation)
4. [Distribution Analysis](#4-distribution-analysis)
5. [Custom Quantization Pipeline](#5-custom-quantization-pipeline)
6. [Batch Processing](#6-batch-processing)
7. [Visualization Gallery](#7-visualization-gallery)
8. [Performance Optimization](#8-performance-optimization)

---

## 1. Basic Quantization

### Example 1.1: Quick Start

```python
from adaptive_quantization import AdaptiveQuantizer, load_embeddings

# Load embeddings
embeddings, words = load_embeddings('glove.6B.100d.vec')
print(f"Loaded: {embeddings.shape}")

# Quantize with k=32 (recommended for most embeddings)
quantizer = AdaptiveQuantizer(base_k=32, verbose=True)
quantized = quantizer.quantize(embeddings)

# Check compression
original_unique = np.mean([len(np.unique(embeddings[:, i])) 
                          for i in range(embeddings.shape[1])])
quantized_unique = np.mean([len(np.unique(quantized[:, i])) 
                           for i in range(quantized.shape[1])])

print(f"\nCompression:")
print(f"  Original: {original_unique:.0f} unique values per dimension")
print(f"  Quantized: {quantized_unique:.0f} unique values per dimension")
print(f"  Reduction: {(1 - quantized_unique/original_unique)*100:.1f}%")
```

### Example 1.2: Silent Quantization

```python
# For scripts/automation - disable verbose output
quantizer = AdaptiveQuantizer(base_k=32, verbose=False)
quantized = quantizer.quantize(embeddings)

# Get summary programmatically
summary = quantizer.get_summary()
print(f"Quantized {summary['n_dimensions']} dimensions")
print(f"k range: {summary['k_range']}")
```

### Example 1.3: Save Quantized Embeddings

```python
from adaptive_quantization import save_embeddings

# Quantize
quantizer = AdaptiveQuantizer(base_k=32)
quantized = quantizer.quantize(embeddings)

# Save to file
save_embeddings('glove.100d.quantized.k32.vec', quantized, words)
print("Saved quantized embeddings")

# Later: load and use
quantized_loaded, words_loaded = load_embeddings('glove.100d.quantized.k32.vec')
```

---

## 2. Walsh Function-Based k Selection

Based on our research findings: **k=32 (2^5) is nearly universal optimal**.

### Example 2.1: Recommended k Values

```python
# For most embeddings (50-d, 200-d, 300-d): use k=32
quantizer_universal = AdaptiveQuantizer(base_k=32)  # 2^5 Walsh orders

# For 100-dimensional embeddings (exception): use k=16
quantizer_100d = AdaptiveQuantizer(base_k=16)  # 2^4 Walsh orders

# For fine-grained (minimal loss): use k=64
quantizer_fine = AdaptiveQuantizer(base_k=64)  # 2^6 Walsh orders
```

### Example 2.2: Testing Powers of 2

```python
from evaluate_quantization import QuantizationEvaluator

evaluator = QuantizationEvaluator('embeddings.vec')

# Test powers of 2: 2^4, 2^5, 2^6
powers_of_2 = [16, 32, 64]
results = {}

for k in powers_of_2:
    print(f"\nTesting k={k} (2^{int(np.log2(k))})...")
    result = evaluator.compare_methods(base_k=k)
    results[k] = result['adaptive']
    
    print(f"  STS:  {result['adaptive']['sts']:.4f}")
    print(f"  SICK: {result['adaptive']['sick']:.4f}")

# Find optimal
optimal_k = max(results.keys(), key=lambda k: results[k]['sts'])
print(f"\nOptimal k: {optimal_k} (2^{int(np.log2(optimal_k))})")
```

### Example 2.3: Dimension-Dependent Selection

```python
def select_optimal_k(embeddings):
    """Select optimal k based on dimensionality."""
    n_dims = embeddings.shape[1]
    
    if n_dims <= 100:
        # Exception: 100-d needs finer quantization
        return 16  # 2^4
    else:
        # Most embeddings: universal optimal
        return 32  # 2^5

# Usage
optimal_k = select_optimal_k(embeddings)
print(f"Selected k={optimal_k} for {embeddings.shape[1]}-dimensional embeddings")

quantizer = AdaptiveQuantizer(base_k=optimal_k)
quantized = quantizer.quantize(embeddings)
```

---

## 3. Comparative Evaluation

### Example 3.1: Three-Way Comparison

```python
from evaluate_quantization import QuantizationEvaluator

evaluator = QuantizationEvaluator('embeddings.vec')

# Compare: Original, Uniform k=32, Adaptive k=32
results = evaluator.compare_methods(base_k=32, use_lebesgue=True)

# Print formatted results
evaluator.print_results(results)

# Extract improvements
baseline_sts = results['original']['sts']
uniform_gain = (results['uniform']['sts'] - baseline_sts) / baseline_sts * 100
adaptive_gain = (results['adaptive']['sts'] - baseline_sts) / baseline_sts * 100

print(f"\nImprovement over baseline:")
print(f"  Uniform:  {uniform_gain:+.1f}%")
print(f"  Adaptive: {adaptive_gain:+.1f}%")
```

### Example 3.2: Custom Evaluation Metrics

```python
from evaluate_quantization import EmbeddingEvaluator
import numpy as np

evaluator = EmbeddingEvaluator()

# Create word-to-index mapping
word_to_idx = {word: i for i, word in enumerate(words)}

# Evaluate multiple versions
versions = {
    'original': embeddings,
    'quantized_16': quantizer_16.quantize(embeddings),
    'quantized_32': quantizer_32.quantize(embeddings),
    'quantized_64': quantizer_64.quantize(embeddings),
}

print("Evaluation Results:\n")
print(f"{'Method':<15} {'STS':>8} {'SICK':>8} {'Mean':>8}")
print("-" * 45)

for name, emb in versions.items():
    sts = evaluator.evaluate_sts(emb, word_to_idx)
    sick = evaluator.evaluate_sick(emb, word_to_idx)
    mean_score = (sts + sick) / 2
    print(f"{name:<15} {sts:>8.4f} {sick:>8.4f} {mean_score:>8.4f}")
```

### Example 3.3: Cross-Architecture Comparison

```python
# Load different embedding types
skipgram, words_sg = load_embeddings('skipgram.300d.vec')
cbow, words_cbow = load_embeddings('cbow.300d.vec')

# Evaluate both
eval_sg = QuantizationEvaluator('skipgram.300d.vec')
eval_cbow = QuantizationEvaluator('cbow.300d.vec')

results_sg = eval_sg.compare_methods(base_k=32)
results_cbow = eval_cbow.compare_methods(base_k=32)

# Compare architectures
print("Skip-gram:")
print(f"  Original:  {results_sg['original']['sts']:.4f}")
print(f"  Quantized: {results_sg['adaptive']['sts']:.4f} "
      f"({(results_sg['adaptive']['sts']-results_sg['original']['sts'])*100:+.1f}%)")

print("\nCBOW:")
print(f"  Original:  {results_cbow['original']['sts']:.4f}")
print(f"  Quantized: {results_cbow['adaptive']['sts']:.4f} "
      f"({(results_cbow['adaptive']['sts']-results_cbow['original']['sts'])*100:+.1f}%)")
```

---

## 4. Distribution Analysis

### Example 4.1: Analyze Dimension Distributions

```python
from adaptive_quantization import AdaptiveQuantizer

# Analyze without quantizing
quantizer = AdaptiveQuantizer(base_k=32)
profiles = quantizer.analyze_dimensions(embeddings)

# Print distribution statistics
quantizer.print_summary()

# Examine individual dimensions
print("\nSample Dimension Profiles:")
for profile in profiles[:10]:
    print(f"  Dim {profile.dim_id}: {profile.dist_type:12s} "
          f"skew={profile.skewness:+.2f} "
          f"kurt={profile.kurtosis:+.2f} "
          f"k={profile.optimal_k:3d} "
          f"method={profile.quantization_method}")
```

### Example 4.2: Identify Problematic Dimensions

```python
# Find highly skewed dimensions
skewed_dims = [p for p in profiles if abs(p.skewness) > 1.0]
print(f"\nHighly skewed dimensions: {len(skewed_dims)}")
for p in skewed_dims:
    print(f"  Dim {p.dim_id}: skewness={p.skewness:+.2f}, "
          f"type={p.dist_type}")

# Find multimodal dimensions
multimodal_dims = [p for p in profiles if p.n_modes >= 2]
print(f"\nMultimodal dimensions: {len(multimodal_dims)}")
for p in multimodal_dims:
    print(f"  Dim {p.dim_id}: {p.n_modes} modes, "
          f"method={p.quantization_method}")

# Find dimensions with extreme kurtosis
heavy_tailed = [p for p in profiles if abs(p.kurtosis) > 5.0]
print(f"\nHeavy-tailed dimensions: {len(heavy_tailed)}")
```

### Example 4.3: Distribution Type Percentages

```python
from collections import Counter

summary = quantizer.get_summary()
dist_counts = summary['distribution_types']
total = summary['n_dimensions']

print("Distribution Type Breakdown:\n")
for dist_type, count in sorted(dist_counts.items(), 
                               key=lambda x: x[1], reverse=True):
    pct = count / total * 100
    print(f"  {dist_type:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Show which quantization method is used
    dims_of_type = [p for p in profiles if p.dist_type == dist_type]
    method_counts = Counter([p.quantization_method for p in dims_of_type])
    for method, m_count in method_counts.items():
        print(f"    → {method}: {m_count}")
```

---

## 5. Custom Quantization Pipeline

### Example 5.1: Selective Dimension Quantization

```python
# Quantize only Gaussian dimensions uniformly
def selective_quantize(embeddings, profiles, base_k=32):
    quantized = embeddings.copy()
    quantizer = AdaptiveQuantizer(base_k=base_k)
    
    gaussian_dims = [p for p in profiles if p.dist_type == 'gaussian']
    print(f"Quantizing {len(gaussian_dims)} Gaussian dimensions...")
    
    for profile in gaussian_dims:
        quantized[:, profile.dim_id] = quantizer.quantize_dimension(
            embeddings[:, profile.dim_id], 
            profile
        )
    
    return quantized

# Usage
profiles = quantizer.analyze_dimensions(embeddings)
selective_quantized = selective_quantize(embeddings, profiles, base_k=32)
```

### Example 5.2: Aggressive Quantization for Skewed Dimensions

```python
# Use higher k for skewed dimensions to preserve information
def adaptive_k_quantize(embeddings, profiles):
    quantized = embeddings.copy()
    quantizer = AdaptiveQuantizer(base_k=32)
    
    for profile in profiles:
        # Adjust k based on distribution
        if abs(profile.skewness) > 1.0:
            # Use 2× k for highly skewed
            profile.optimal_k *= 2
            print(f"Dim {profile.dim_id}: Increased k to {profile.optimal_k}")
        
        quantized[:, profile.dim_id] = quantizer.quantize_dimension(
            embeddings[:, profile.dim_id],
            profile
        )
    
    return quantized

custom_quantized = adaptive_k_quantize(embeddings, profiles)
```

### Example 5.3: Two-Stage Quantization

```python
# Stage 1: Coarse quantization for initial compression
stage1_quantizer = AdaptiveQuantizer(base_k=16)  # 2^4
stage1_quantized = stage1_quantizer.quantize(embeddings)

# Evaluate
eval1 = EmbeddingEvaluator()
sts1 = eval1.evaluate_sts(stage1_quantized, word_to_idx)
print(f"Stage 1 (k=16): STS={sts1:.4f}")

# Stage 2: Refine if performance drop is significant
if sts1 < original_sts * 0.95:  # 5% drop threshold
    print("Performance drop detected, refining...")
    stage2_quantizer = AdaptiveQuantizer(base_k=32)  # 2^5
    final_quantized = stage2_quantizer.quantize(embeddings)
else:
    print("Stage 1 sufficient")
    final_quantized = stage1_quantized

sts_final = eval1.evaluate_sts(final_quantized, word_to_idx)
print(f"Final: STS={sts_final:.4f}")
```

---

## 6. Batch Processing

### Example 6.1: Process Multiple Embedding Files

```python
import glob
import os

embedding_files = glob.glob('embeddings/*.vec')

for filepath in embedding_files:
    name = os.path.basename(filepath).replace('.vec', '')
    print(f"\nProcessing: {name}")
    
    # Load
    embeddings, words = load_embeddings(filepath)
    
    # Quantize with k=32
    quantizer = AdaptiveQuantizer(base_k=32, verbose=False)
    quantized = quantizer.quantize(embeddings)
    
    # Save
    output_path = f'quantized/{name}.k32.vec'
    save_embeddings(output_path, quantized, words)
    
    # Log statistics
    summary = quantizer.get_summary()
    print(f"  Dimensions: {summary['n_dimensions']}")
    print(f"  k mean: {summary['k_mean']:.1f}")
    print(f"  Saved to: {output_path}")
```

### Example 6.2: Batch Evaluation Report

```python
import pandas as pd

# Process multiple embeddings and collect results
results_list = []

for filepath in embedding_files:
    name = os.path.basename(filepath).replace('.vec', '')
    
    evaluator = QuantizationEvaluator(filepath)
    results = evaluator.compare_methods(base_k=32)
    
    results_list.append({
        'name': name,
        'original_sts': results['original']['sts'],
        'quantized_sts': results['adaptive']['sts'],
        'improvement': (results['adaptive']['sts'] - results['original']['sts']) / 
                      results['original']['sts'] * 100,
        'original_sick': results['original']['sick'],
        'quantized_sick': results['adaptive']['sick']
    })

# Create report
df = pd.DataFrame(results_list)
print("\nBatch Evaluation Report:")
print(df.to_string(index=False))

# Save to CSV
df.to_csv('quantization_report.csv', index=False)
print("\nSaved report to: quantization_report.csv")
```

### Example 6.3: Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def process_embedding(filepath, base_k=32):
    """Process single embedding file."""
    name = os.path.basename(filepath)
    embeddings, words = load_embeddings(filepath)
    
    quantizer = AdaptiveQuantizer(base_k=base_k, verbose=False)
    quantized = quantizer.quantize(embeddings)
    
    output_path = f'quantized/{name}'
    save_embeddings(output_path, quantized, words)
    
    return name, quantizer.get_summary()

# Process in parallel
with Pool(4) as pool:
    process_func = partial(process_embedding, base_k=32)
    results = pool.map(process_func, embedding_files)

# Print results
for name, summary in results:
    print(f"{name}: {summary['n_dimensions']} dims, "
          f"k={summary['k_mean']:.1f}")
```

---

## 7. Visualization Gallery

### Example 7.1: Complete Visualization Suite

```python
from visualize_quantization import QuantizationVisualizer

# Load and quantize
embeddings, words = load_embeddings('embeddings.vec')
quantizer = AdaptiveQuantizer(base_k=32, use_lebesgue=True)
quantized = quantizer.quantize(embeddings)

# Initialize visualizer
viz = QuantizationVisualizer()

# 1. Dimension types
viz.plot_dimension_types(
    quantizer, 
    save_path='figures/dimension_types.png'
)

# 2. Sample profiles
viz.plot_dimension_profiles(
    quantizer, 
    n_samples=20, 
    save_path='figures/profiles.png'
)

# 3. Before/after distributions
viz.plot_before_after_distributions(
    embeddings, 
    quantized,
    n_samples=9,
    save_path='figures/before_after.png'
)

# 4. Performance comparison
evaluator = QuantizationEvaluator('embeddings.vec')
results = evaluator.compare_methods(base_k=32)
viz.plot_performance_comparison(
    results,
    save_path='figures/performance.png'
)

print("All visualizations saved to figures/")
```

### Example 7.2: Custom Color Scheme

```python
# Initialize with custom colors
viz = QuantizationVisualizer()
viz.colors = {
    'gaussian': '#1f77b4',      # Blue
    'right_skewed': '#ff7f0e',  # Orange
    'left_skewed': '#2ca02c',   # Green
    'multimodal': '#d62728',    # Red
    'heavy_tailed': '#9467bd',  # Purple
    'other': '#8c564b'          # Brown
}

viz.plot_dimension_types(quantizer, save_path='custom_colors.png')
```

### Example 7.3: High-Resolution Figures for Publication

```python
# High DPI for publication
viz = QuantizationVisualizer(figsize_base=(16, 10))

fig = viz.plot_dimension_types(quantizer)
fig.savefig('publication_figure.png', dpi=600, bbox_inches='tight')
fig.savefig('publication_figure.pdf', bbox_inches='tight')  # Vector format

print("Saved high-resolution figures")
```

---

## 8. Performance Optimization

### Example 8.1: Memory-Efficient Processing

```python
# For very large embeddings (>1M words)
def quantize_in_chunks(embeddings, chunk_size=100000):
    """Process embeddings in chunks to save memory."""
    quantizer = AdaptiveQuantizer(base_k=32, verbose=False)
    
    # Analyze on sample
    sample_size = min(50000, embeddings.shape[0])
    sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    quantizer.analyze_dimensions(embeddings[sample_indices])
    
    # Quantize in chunks
    quantized = np.zeros_like(embeddings)
    n_chunks = (embeddings.shape[0] + chunk_size - 1) // chunk_size
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, embeddings.shape[0])
        chunk = embeddings[start:end]
        
        # Quantize chunk
        for dim in range(embeddings.shape[1]):
            profile = quantizer.dimension_profiles[dim]
            quantized[start:end, dim] = quantizer.quantize_dimension(
                chunk[:, dim], profile
            )
        
        print(f"Processed chunk {i+1}/{n_chunks}")
    
    return quantized
```

### Example 8.2: Fast k Selection

```python
# Quick k selection without full analysis
def fast_k_selection(embeddings):
    """Fast k selection based on basic statistics."""
    skewness = np.mean([abs(stats.skew(embeddings[:, i])) 
                       for i in range(embeddings.shape[1])])
    
    if skewness > 0.5:
        return 16  # 2^4 for high skew
    else:
        return 32  # 2^5 for low skew

k = fast_k_selection(embeddings)
print(f"Selected k={k} based on mean skewness")
```

### Example 8.3: Caching Dimension Profiles

```python
import pickle

# Save profiles for reuse
quantizer = AdaptiveQuantizer(base_k=32)
quantizer.analyze_dimensions(embeddings)

with open('dimension_profiles.pkl', 'wb') as f:
    pickle.dump(quantizer.dimension_profiles, f)

print("Saved dimension profiles")

# Later: load and reuse
with open('dimension_profiles.pkl', 'rb') as f:
    loaded_profiles = pickle.load(f)

# Reuse profiles
new_quantizer = AdaptiveQuantizer(base_k=32, verbose=False)
new_quantizer.dimension_profiles = loaded_profiles
quantized = new_quantizer.quantize(embeddings)

print("Quantized using cached profiles (much faster!)")
```

---

## Complete Pipeline Example

```python
#!/usr/bin/env python3
"""
Complete quantization pipeline with all best practices.
"""

from adaptive_quantization import AdaptiveQuantizer, load_embeddings, save_embeddings
from evaluate_quantization import QuantizationEvaluator
from visualize_quantization import QuantizationVisualizer
import numpy as np

def main():
    # Configuration
    INPUT_FILE = 'glove.6B.300d.vec'
    OUTPUT_FILE = 'glove.6B.300d.quantized.k32.vec'
    BASE_K = 32  # 2^5 Walsh orders (recommended)
    
    print("=" * 60)
    print("Adaptive Word Embedding Quantization Pipeline")
    print("=" * 60)
    
    # 1. Load embeddings
    print("\n[1/5] Loading embeddings...")
    embeddings, words = load_embeddings(INPUT_FILE)
    print(f"  Loaded: {embeddings.shape[0]} words × {embeddings.shape[1]} dims")
    print(f"  Size: {embeddings.nbytes / 1024**2:.1f} MB")
    
    # 2. Quantize
    print(f"\n[2/5] Quantizing with k={BASE_K} (2^{int(np.log2(BASE_K))})...")
    quantizer = AdaptiveQuantizer(base_k=BASE_K, use_lebesgue=True, verbose=True)
    quantized = quantizer.quantize(embeddings)
    
    # 3. Analyze results
    print("\n[3/5] Analyzing results...")
    summary = quantizer.get_summary()
    quantizer.print_summary()
    
    # 4. Evaluate
    print("\n[4/5] Evaluating performance...")
    evaluator = QuantizationEvaluator(INPUT_FILE)
    results = evaluator.compare_methods(base_k=BASE_K, use_lebesgue=True)
    evaluator.print_results(results)
    
    # 5. Visualize
    print("\n[5/5] Creating visualizations...")
    viz = QuantizationVisualizer()
    viz.plot_dimension_types(quantizer, save_path='dimension_types.png')
    viz.plot_before_after_distributions(embeddings, quantized, 
                                       save_path='before_after.png')
    viz.plot_performance_comparison(results, save_path='performance.png')
    
    # 6. Save
    print(f"\nSaving quantized embeddings to: {OUTPUT_FILE}")
    save_embeddings(OUTPUT_FILE, quantized, words)
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Compression: {embeddings.nbytes / quantized.nbytes:.1f}× (effective)")
    print(f"Performance: {results['adaptive']['sts']:.4f} STS "
          f"({(results['adaptive']['sts']-results['original']['sts'])*100:+.1f}%)")
    print("\nGenerated files:")
    print("  - dimension_types.png")
    print("  - before_after.png")
    print("  - performance.png")

if __name__ == "__main__":
    main()
```

---

## Tips and Best Practices

### 1. Choosing k

- **Start with k=32** (2^5): Nearly universal optimal
- **For 100-d embeddings**: Try k=16 (2^4) first
- **If unsure**: Test powers of 2 (16, 32, 64) and compare

### 2. Lebesgue vs Percentile

- **Low skew (<5%)**: Either works fine
- **High skew (>10%)**: Use `use_lebesgue=True`
- **Speed matters**: Use default (percentile is faster)

### 3. Evaluation

- **Always use dual metrics**: STS + SICK catches overfitting
- **Watch for divergence**: If STS↑ but SICK↓, investigate
- **Baseline comparison**: Always compare to original

### 4. Visualization

- **Start with dimension_types**: Shows distribution of types
- **Before/after**: Visual confirmation of quantization effect
- **Performance chart**: Shows improvement clearly

### 5. Production Use

- **Save profiles**: Cache dimension analysis for faster reuse
- **Batch process**: Use multiprocessing for multiple files
- **Monitor**: Track k values and skewness percentages

---

For more details, see [API_REFERENCE.md](API_REFERENCE.md).
