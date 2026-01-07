#!/usr/bin/env python3
"""
Adaptive Quantization for Word Embeddings

This module implements dimension-wise adaptive quantization that:
1. Classifies each dimension's distribution type
2. Selects optimal quantization level (k) per dimension
3. Applies appropriate quantization method per distribution type

Author: Based on research by Kow Kuroda
Paper: "Dimension-adaptive quantization for heterogeneous embeddings"
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DimensionProfile:
    """Profile of a single dimension's distribution."""
    dim_id: int
    dist_type: str
    skewness: float
    kurtosis: float
    n_modes: int
    shapiro_p: float
    optimal_k: int
    quantization_method: str
    
    def __repr__(self):
        return (f"Dim {self.dim_id}: {self.dist_type} "
                f"(skew={self.skewness:.2f}, kurt={self.kurtosis:.2f}, "
                f"k={self.optimal_k}, method={self.quantization_method})")


class DimensionClassifier:
    """Classify distribution type for each dimension."""
    
    @staticmethod
    def detect_modes(values: np.ndarray, prominence: float = 0.1) -> int:
        """Detect number of modes in distribution."""
        hist, edges = np.histogram(values, bins=50)
        # Normalize
        hist = hist / hist.max()
        # Find peaks
        peaks, _ = find_peaks(hist, prominence=prominence)
        return len(peaks)
    
    @staticmethod
    def classify_dimension(values: np.ndarray, dim_id: int, base_k: int = 20, 
                          use_lebesgue: bool = False) -> DimensionProfile:
        """
        Classify a dimension's distribution type.
        
        Args:
            values: 1D array of values for this dimension
            dim_id: Dimension index
            base_k: Base quantization level (default: 20)
                   Recommended: 20 (coarse, fast), 50-100 (medium), 200+ (fine)
            use_lebesgue: Use true Lebesgue/equi-depth for skewed distributions
        
        Returns:
            DimensionProfile with classification and recommended settings
        """
        # Basic statistics
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)
        
        # Test normality (sample if too large)
        sample_size = min(5000, len(values))
        sample = np.random.choice(values, sample_size, replace=False)
        _, shapiro_p = stats.shapiro(sample)
        
        # Detect modes
        n_modes = DimensionClassifier.detect_modes(values)
        
        # Classify distribution type and scale k proportionally
        # All k values scale with base_k (default multipliers for base_k=20)
        if shapiro_p > 0.05 and abs(skewness) < 0.2 and abs(kurtosis) < 0.5:
            dist_type = "gaussian"
            optimal_k = base_k  # 1.0x base
            method = "uniform"
            
        elif abs(skewness) > 0.5:
            if skewness > 0:
                dist_type = "right_skewed"
            else:
                dist_type = "left_skewed"
            optimal_k = int(base_k * 1.5)  # 1.5x base (was 30 when base=20)
            # Choose between percentile (hybrid) or lebesgue (true equi-depth)
            method = "lebesgue" if use_lebesgue else "percentile"
            
        elif n_modes >= 2:
            dist_type = "multimodal"
            optimal_k = min(int(n_modes * base_k * 0.6), base_k * 2)  # Scale with modes, cap at 2x
            method = "kmeans"
            
        elif abs(kurtosis) > 3.0:
            dist_type = "heavy_tailed"
            optimal_k = int(base_k * 1.25)  # 1.25x base (was 25 when base=20)
            method = "robust"
            
        else:
            dist_type = "other"
            optimal_k = base_k  # 1.0x base
            method = "uniform"
        
        return DimensionProfile(
            dim_id=dim_id,
            dist_type=dist_type,
            skewness=skewness,
            kurtosis=kurtosis,
            n_modes=n_modes,
            shapiro_p=shapiro_p,
            optimal_k=optimal_k,
            quantization_method=method
        )


class AdaptiveQuantizer:
    """Adaptive quantization engine."""
    
    def __init__(self, base_k: int = 20, use_lebesgue: bool = False, verbose: bool = True):
        """
        Initialize adaptive quantizer.
        
        Args:
            base_k: Base number of quantization levels (default: 20)
                   - 20: Coarse, fast, high compression
                   - 50-100: Medium precision
                   - 200+: Fine precision, minimal loss
            use_lebesgue: Use true Lebesgue (equi-depth) for skewed dims (default: False)
                         - False: Use percentile method (hybrid)
                         - True: Use true equi-depth with equal counts
            verbose: Print progress messages
        """
        self.base_k = base_k
        self.use_lebesgue = use_lebesgue
        self.verbose = verbose
        self.dimension_profiles: List[DimensionProfile] = []
    
    def analyze_dimensions(self, embeddings: np.ndarray) -> List[DimensionProfile]:
        """Analyze all dimensions and create profiles."""
        n_words, n_dims = embeddings.shape
        
        if self.verbose:
            print(f"Analyzing {n_dims} dimensions...")
        
        profiles = []
        for dim in range(n_dims):
            profile = DimensionClassifier.classify_dimension(
                embeddings[:, dim], dim, base_k=self.base_k, 
                use_lebesgue=self.use_lebesgue
            )
            profiles.append(profile)
            
            if self.verbose and dim % 20 == 0:
                print(f"  Dim {dim}: {profile.dist_type} (k={profile.optimal_k})")
        
        self.dimension_profiles = profiles
        return profiles
    
    def _uniform_quantize(self, values: np.ndarray, k: int) -> np.ndarray:
        """Uniform symmetric quantization."""
        vmin, vmax = values.min(), values.max()
        # Create k equally-spaced levels
        levels = np.linspace(vmin, vmax, k)
        # Assign each value to nearest level
        quantized = levels[np.argmin(np.abs(values[:, None] - levels), axis=1)]
        return quantized
    
    def _percentile_quantize(self, values: np.ndarray, k: int) -> np.ndarray:
        """Percentile-based quantization (preserves distribution shape)."""
        percentiles = np.linspace(0, 100, k)
        levels = np.percentile(values, percentiles)
        quantized = levels[np.argmin(np.abs(values[:, None] - levels), axis=1)]
        return quantized
    
    def _lebesgue_quantize(self, values: np.ndarray, k: int) -> np.ndarray:
        """
        Lebesgue-style equi-depth quantization.
        
        Partitions by frequency (not value range), ensuring equal counts
        per bin. Bin centers are the mean within each bin.
        
        This is the true Lebesgue analogue: partition the "y-axis" 
        (frequency) rather than "x-axis" (values).
        """
        n = len(values)
        sorted_vals = np.sort(values)
        
        # Create bin boundaries with equal counts
        bin_size = n // k
        bin_centers = []
        
        for i in range(k):
            start = i * bin_size
            end = (i + 1) * bin_size if i < k - 1 else n  # Last bin gets remainder
            bin_center = np.mean(sorted_vals[start:end])
            bin_centers.append(bin_center)
        
        bin_centers = np.array(bin_centers)
        
        # Assign each value to nearest bin center
        quantized = bin_centers[np.argmin(np.abs(values[:, None] - bin_centers), axis=1)]
        return quantized
    
    def _kmeans_quantize(self, values: np.ndarray, k: int) -> np.ndarray:
        """K-means clustering quantization."""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(values.reshape(-1, 1))
        quantized = kmeans.cluster_centers_[labels].flatten()
        return quantized
    
    def _robust_quantize(self, values: np.ndarray, k: int) -> np.ndarray:
        """Robust quantization (clips outliers before quantizing)."""
        # Clip to 1st and 99th percentiles
        q1, q99 = np.percentile(values, [1, 99])
        clipped = np.clip(values, q1, q99)
        # Then uniform quantize
        return self._uniform_quantize(clipped, k)
    
    def quantize_dimension(self, values: np.ndarray, profile: DimensionProfile) -> np.ndarray:
        """Quantize a single dimension according to its profile."""
        method = profile.quantization_method
        k = profile.optimal_k
        
        if method == "uniform":
            return self._uniform_quantize(values, k)
        elif method == "percentile":
            return self._percentile_quantize(values, k)
        elif method == "lebesgue":
            return self._lebesgue_quantize(values, k)
        elif method == "kmeans":
            return self._kmeans_quantize(values, k)
        elif method == "robust":
            return self._robust_quantize(values, k)
        else:
            return self._uniform_quantize(values, k)
    
    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply adaptive quantization to embeddings.
        
        Args:
            embeddings: Input embeddings (n_words × n_dims)
            
        Returns:
            Adaptively quantized embeddings
        """
        if not self.dimension_profiles:
            self.analyze_dimensions(embeddings)
        
        n_words, n_dims = embeddings.shape
        quantized = np.zeros_like(embeddings)
        
        if self.verbose:
            print(f"\nQuantizing {n_dims} dimensions adaptively...")
        
        for dim, profile in enumerate(self.dimension_profiles):
            quantized[:, dim] = self.quantize_dimension(
                embeddings[:, dim], 
                profile
            )
            
            if self.verbose and dim % 20 == 0:
                unique_before = len(np.unique(embeddings[:, dim]))
                unique_after = len(np.unique(quantized[:, dim]))
                print(f"  Dim {dim}: {unique_before} → {unique_after} unique values "
                      f"({profile.quantization_method})")
        
        return quantized
    
    def get_summary(self) -> Dict:
        """Get summary statistics of dimension profiles."""
        if not self.dimension_profiles:
            return {}
        
        dist_types = [p.dist_type for p in self.dimension_profiles]
        methods = [p.quantization_method for p in self.dimension_profiles]
        ks = [p.optimal_k for p in self.dimension_profiles]
        
        from collections import Counter
        
        return {
            'n_dimensions': len(self.dimension_profiles),
            'distribution_types': dict(Counter(dist_types)),
            'quantization_methods': dict(Counter(methods)),
            'k_range': (min(ks), max(ks)),
            'k_mean': np.mean(ks),
            'k_median': np.median(ks),
        }
    
    def print_summary(self):
        """Print summary of dimension analysis."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("ADAPTIVE QUANTIZATION SUMMARY")
        print("="*70)
        print(f"Total dimensions: {summary['n_dimensions']}")
        print()
        print("Distribution types:")
        for dist_type, count in summary['distribution_types'].items():
            pct = 100 * count / summary['n_dimensions']
            print(f"  {dist_type:20s}: {count:3d} ({pct:5.1f}%)")
        print()
        print("Quantization methods:")
        for method, count in summary['quantization_methods'].items():
            pct = 100 * count / summary['n_dimensions']
            print(f"  {method:20s}: {count:3d} ({pct:5.1f}%)")
        print()
        print(f"Quantization levels (k):")
        print(f"  Range: [{summary['k_range'][0]}, {summary['k_range'][1]}]")
        print(f"  Mean: {summary['k_mean']:.1f}")
        print(f"  Median: {summary['k_median']:.1f}")
        print("="*70 + "\n")


class UniformQuantizer:
    """Baseline uniform quantization for comparison."""
    
    def __init__(self, k: int = 20):
        self.k = k
    
    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply uniform quantization to all dimensions."""
        n_words, n_dims = embeddings.shape
        quantized = np.zeros_like(embeddings)
        
        for dim in range(n_dims):
            values = embeddings[:, dim]
            vmin, vmax = values.min(), values.max()
            levels = np.linspace(vmin, vmax, self.k)
            quantized[:, dim] = levels[np.argmin(np.abs(values[:, None] - levels), axis=1)]
        
        return quantized


def load_embeddings(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings from various formats.
    
    Supports:
    - Word2Vec text format (.vec, .txt)
    - GloVe text format (.txt)
    - Word2Vec binary format (.bin)
    - GloVe binary format (.bin) - will try gensim first
    """
    import os
    from pathlib import Path
    
    # Try gensim KeyedVectors first for .bin files
    if filepath.endswith('.bin'):
        try:
            from gensim.models import KeyedVectors
            print(f"  Attempting to load as binary with gensim...")
            
            # Try Word2Vec binary format first
            try:
                kv = KeyedVectors.load_word2vec_format(filepath, binary=True)
                print(f"  ✓ Loaded as Word2Vec binary format")
                return kv.vectors, list(kv.index_to_key)
            except:
                pass
            
            # Try GloVe format (text-like but in .bin extension)
            try:
                kv = KeyedVectors.load_word2vec_format(filepath, binary=False, no_header=True)
                print(f"  ✓ Loaded as GloVe text format (in .bin file)")
                return kv.vectors, list(kv.index_to_key)
            except:
                pass
                
        except ImportError:
            print(f"  Warning: gensim not available, trying manual parsing...")
    
    # Try text format (Word2Vec or GloVe)
    try:
        words = []
        vectors = []
        
        # First try with header (Word2Vec format)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip().split()
            
            # Check if first line is header (two integers: vocab_size, dim)
            has_header = False
            if len(first_line) == 2:
                try:
                    vocab_size, dim = int(first_line[0]), int(first_line[1])
                    has_header = True
                    print(f"  Detected Word2Vec format with header: {vocab_size} words, {dim} dims")
                except ValueError:
                    # First line is not a header, treat as data
                    pass
            
            # If no header, reset to beginning
            if not has_header:
                f.seek(0)
                print(f"  Detected GloVe format (no header)")
            
            # Read embeddings
            for line_num, line in enumerate(f, start=2 if has_header else 1):
                parts = line.strip().split()
                if len(parts) < 10:  # Skip malformed lines
                    continue
                
                word = parts[0]
                try:
                    vec = [float(x) for x in parts[1:]]
                    words.append(word)
                    vectors.append(vec)
                except ValueError:
                    # Skip lines with non-numeric values
                    continue
        
        if len(vectors) > 0:
            print(f"  ✓ Loaded {len(vectors)} vectors with {len(vectors[0])} dimensions")
            return np.array(vectors), words
        else:
            raise ValueError("No valid vectors found in file")
            
    except UnicodeDecodeError as e:
        raise ValueError(
            f"File appears to be binary but couldn't be loaded.\n"
            f"Error: {e}\n"
            f"Solutions:\n"
            f"  1. Install gensim: pip install gensim\n"
            f"  2. Convert to text format first\n"
            f"  3. Make sure file is actually a valid embedding file"
        )



def save_embeddings(filepath: str, embeddings: np.ndarray, words: List[str]):
    """Save embeddings in Word2Vec format."""
    n_words, n_dims = embeddings.shape
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{n_words} {n_dims}\n")
        for word, vec in zip(words, embeddings):
            vec_str = ' '.join([f"{v:.6f}" for v in vec])
            f.write(f"{word} {vec_str}\n")


# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Adaptive quantization for word embeddings (dimension-wise)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-generate output filename (adds -adaptive suffix)
  python adaptive_quantization.py embeddings.vec
  
  # Specify output filename
  python adaptive_quantization.py embeddings.vec output.vec
  
  # With explicit output option
  python adaptive_quantization.py embeddings.vec -o quantized.vec
  python adaptive_quantization.py embeddings.vec --output quantized.vec
        """
    )
    
    parser.add_argument('input', 
                       help='Input embedding file (.vec format)')
    parser.add_argument('output', nargs='?', default=None,
                       help='Output embedding file (default: <input>-adaptive.vec)')
    parser.add_argument('-o', '--output-file', dest='output_alt',
                       help='Alternative way to specify output file')
    parser.add_argument('-k', '--default-k', type=int, default=20,
                       help='Default k value for Gaussian dimensions (default: 20)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Determine output filename
    if args.output_alt:
        output_file = args.output_alt
    elif args.output:
        output_file = args.output
    else:
        # Auto-generate: input.vec -> input-adaptive.vec
        output_file = args.input.replace('.vec', '-adaptive.vec')
        if output_file == args.input:  # No .vec extension
            output_file = args.input + '-adaptive.vec'
    
    input_file = args.input
    verbose = not args.quiet
    
    if verbose:
        print(f"Loading embeddings from: {input_file}")
    
    embeddings, words = load_embeddings(input_file)
    
    if verbose:
        print(f"Loaded {len(words)} words, {embeddings.shape[1]} dimensions\n")
    
    # Apply adaptive quantization
    quantizer = AdaptiveQuantizer(verbose=verbose)
    quantized = quantizer.quantize(embeddings)
    
    # Print summary
    if verbose:
        quantizer.print_summary()
    
    # Save
    save_embeddings(output_file, quantized, words)
    
    if verbose:
        print(f"Saved adaptively quantized embeddings to: {output_file}")
        
        # Compare with uniform
        print("\nFor comparison, uniform quantization (k=20):")
        uniform = UniformQuantizer(k=20)
        uniform_quantized = uniform.quantize(embeddings)
        
        # Quick stats
        print(f"  Adaptive - mean unique/dim: {np.mean([len(np.unique(quantized[:, i])) for i in range(quantized.shape[1])]):.1f}")
        print(f"  Uniform  - mean unique/dim: {np.mean([len(np.unique(uniform_quantized[:, i])) for i in range(uniform_quantized.shape[1])]):.1f}")
    else:
        print(f"{output_file}")
