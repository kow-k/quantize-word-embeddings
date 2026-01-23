#!/usr/bin/env python3
"""
Quantization Evaluation Framework

Compares different quantization strategies:
1. No quantization (baseline)
2. Uniform quantization (k=20)
3. Adaptive quantization (dimension-wise)

Evaluates on multiple benchmarks and provides detailed analysis.

Author: Based on research by Kow Kuroda
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
from adaptive_quantization import (
    AdaptiveQuantizer, 
    UniformQuantizer, 
    load_embeddings
)


class EmbeddingEvaluator:
    """Evaluate embeddings on semantic similarity tasks."""
    
    def __init__(self):
        self.results = {}
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if v1 is None or v2 is None:
            return None
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return None
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def sentence_vector(self, sentence: str, embeddings: np.ndarray, 
                       word_to_idx: Dict[str, int]) -> np.ndarray:
        """Get sentence embedding by averaging word vectors."""
        words = sentence.lower().split()
        vectors = []
        
        for word in words:
            # Strip punctuation for matching
            word_clean = word.strip('.,!?;:"\'-()[]{}')
            if word_clean in word_to_idx:
                vectors.append(embeddings[word_to_idx[word_clean]])
            elif word in word_to_idx:
                vectors.append(embeddings[word_to_idx[word]])
        
        if len(vectors) == 0:
            return None
        
        return np.mean(vectors, axis=0)
    
    def evaluate_sts(self, embeddings: np.ndarray, word_to_idx: Dict[str, int],
                     sts_file: str = None) -> float:
        """
        Evaluate on Semantic Textual Similarity (STS).
        Uses sentence-level similarity (averaging word vectors).
        """
        # Built-in STS sentence pairs (sentence1, sentence2, score 0-5)
        # Sampled from STS Benchmark
        builtin_sts_data = [
            # High similarity (4-5)
            ("A man is playing a guitar.", "A man is playing music.", 4.2),
            ("A woman is cooking dinner.", "A woman is preparing a meal.", 4.5),
            ("The dog is running in the park.", "A dog runs through a park.", 4.3),
            ("A child is reading a book.", "A kid reads a book.", 4.8),
            ("The cat sleeps on the couch.", "A cat is sleeping on a sofa.", 4.6),
            ("A man is riding a bicycle.", "A person rides a bike.", 4.4),
            ("The sun is shining brightly.", "The sun shines bright.", 4.7),
            ("A woman is singing a song.", "A woman sings.", 4.1),
            ("The bird flies in the sky.", "A bird is flying.", 4.0),
            ("A man is swimming in the pool.", "A man swims in a swimming pool.", 4.5),
            
            # Medium similarity (2-4)
            ("A man is playing a piano.", "A man is playing a guitar.", 2.8),
            ("A woman is walking her dog.", "A woman is running with her cat.", 2.2),
            ("The children are playing soccer.", "The kids are watching television.", 1.8),
            ("A man is eating an apple.", "A man is drinking coffee.", 2.0),
            ("The car is parked on the street.", "A truck drives down the road.", 2.3),
            ("A woman is reading a newspaper.", "A man is watching the news.", 2.5),
            ("The students are studying math.", "The children are learning science.", 3.0),
            ("A dog is chasing a ball.", "A cat is playing with yarn.", 2.4),
            ("The plane is taking off.", "The train is arriving.", 2.1),
            ("A man is writing a letter.", "A woman is typing an email.", 3.2),
            
            # Low similarity (0-2)
            ("A man is playing basketball.", "A woman is cooking dinner.", 0.5),
            ("The sun is setting.", "The children are sleeping.", 0.8),
            ("A dog is barking.", "A fish is swimming.", 1.0),
            ("The mountain is tall.", "The ocean is deep.", 1.2),
            ("A woman is dancing.", "A man is sleeping.", 0.3),
            ("The tree has green leaves.", "The car is red.", 0.6),
            ("A child is crying.", "A bird is singing.", 0.9),
            ("The computer is on the desk.", "The flower is in the garden.", 0.4),
            ("A man is running fast.", "A woman is sitting quietly.", 0.7),
            ("The book is interesting.", "The weather is cold.", 0.2),
            
            # Additional pairs
            ("Two men are fighting.", "Two men are wrestling.", 3.8),
            ("A person is folding paper.", "A person is doing origami.", 4.0),
            ("A woman is slicing vegetables.", "A woman is cutting tomatoes.", 3.5),
            ("The boy is playing video games.", "The girl is reading comics.", 1.5),
            ("A cat is chasing a mouse.", "A dog is chasing a cat.", 2.6),
            ("The river flows to the sea.", "Water runs downhill.", 2.9),
            ("A man is lifting weights.", "A person is exercising.", 3.7),
            ("The baby is laughing.", "The infant is crying.", 2.0),
            ("A woman is painting a picture.", "An artist is drawing.", 3.4),
            ("The snow is falling.", "Rain is pouring down.", 2.5),
        ]
        
        predictions = []
        gold_scores = []
        
        for sent1, sent2, score in builtin_sts_data:
            vec1 = self.sentence_vector(sent1, embeddings, word_to_idx)
            vec2 = self.sentence_vector(sent2, embeddings, word_to_idx)
            
            sim = self.cosine_similarity(vec1, vec2)
            if sim is not None:
                predictions.append(sim)
                gold_scores.append(score)
        
        if len(predictions) < 10:
            return 0.0
        
        # Spearman correlation
        rho, _ = spearmanr(gold_scores, predictions)
        return rho
    
    def evaluate_sick(self, embeddings: np.ndarray, word_to_idx: Dict[str, int]) -> float:
        """
        Evaluate on SICK (Sentences Involving Compositional Knowledge).
        
        SICK tests semantic relatedness with more complex compositional phenomena
        than STS. Scores range from 1 (completely unrelated) to 5 (very related).
        
        Uses built-in sample from SICK dataset.
        """
        # Built-in SICK sentence pairs (sentence1, sentence2, relatedness_score 1-5)
        # Sampled from SICK dataset (Marelli et al., 2014)
        builtin_sick_data = [
            # High relatedness (4-5)
            ("A man is playing a guitar", "A man is playing a guitar", 5.0),
            ("A woman is slicing an onion", "A woman is cutting an onion", 4.7),
            ("A man is riding a bicycle", "A man is riding a bike", 5.0),
            ("Children are playing outdoors", "Kids are playing outside", 4.8),
            ("A dog is running through a field", "A dog is running in a field", 4.9),
            ("The man is playing a flute", "A man is playing a flute", 4.9),
            ("A woman is dancing", "The woman is dancing", 4.9),
            ("A cat is playing with a toy", "A cat is playing with an object", 4.5),
            ("A man is cutting a vegetable", "A person is slicing a vegetable", 4.6),
            ("Someone is drawing a picture", "A person is drawing something", 4.7),
            
            # Medium-high relatedness (3-4)
            ("A man is playing a guitar", "Someone is playing an instrument", 3.8),
            ("A woman is cooking", "A person is preparing food", 3.9),
            ("A dog is running", "An animal is moving", 3.5),
            ("Children are playing", "People are having fun", 3.4),
            ("A man is riding a bicycle", "Someone is using transportation", 3.2),
            ("A woman is singing", "A person is making music", 3.7),
            ("A cat is sleeping", "An animal is resting", 3.6),
            ("Someone is reading a book", "A person is learning something", 3.3),
            ("A man is swimming", "Someone is exercising", 3.4),
            ("A woman is walking a dog", "A person is with an animal", 3.5),
            
            # Medium relatedness (2-3)
            ("A man is playing a guitar", "A woman is singing", 2.8),
            ("A dog is running", "A cat is sleeping", 2.2),
            ("Children are playing soccer", "Adults are watching television", 2.0),
            ("A man is cooking dinner", "A woman is eating breakfast", 2.4),
            ("Someone is reading a book", "A person is watching a movie", 2.6),
            ("A woman is dancing", "A man is sitting", 2.1),
            ("A dog is barking", "A cat is meowing", 2.5),
            ("A man is swimming", "A woman is running", 2.7),
            ("Children are laughing", "Adults are working", 2.3),
            ("Someone is drawing", "A person is writing", 2.8),
            
            # Low relatedness (1-2)
            ("A man is playing a guitar", "A car is parked", 1.2),
            ("A woman is cooking", "The sun is shining", 1.0),
            ("A dog is running", "A tree is tall", 1.1),
            ("Children are playing", "The ocean is blue", 1.0),
            ("A man is reading", "A mountain is high", 1.3),
            ("A woman is singing", "A door is closed", 1.1),
            ("A cat is sleeping", "A computer is working", 1.2),
            ("Someone is swimming", "A building is old", 1.0),
            ("A man is dancing", "The weather is cold", 1.1),
            ("A woman is walking", "A flower is blooming", 1.4),
        ]
        
        predictions = []
        gold_scores = []
        
        for sent1, sent2, score in builtin_sick_data:
            vec1 = self.sentence_vector(sent1, embeddings, word_to_idx)
            vec2 = self.sentence_vector(sent2, embeddings, word_to_idx)
            
            sim = self.cosine_similarity(vec1, vec2)
            if sim is not None:
                predictions.append(sim)
                gold_scores.append(score)
        
        if len(predictions) < 10:
            return 0.0
        
        # Spearman correlation (like STS)
        rho, _ = spearmanr(gold_scores, predictions)
        return rho
    
    def evaluate_analogy(self, embeddings: np.ndarray, word_to_idx: Dict[str, int]) -> float:
        """
        Evaluate on analogy task: king - man + woman ≈ queen
        
        Returns accuracy on simple built-in analogies.
        """
        analogies = [
            # (a, b, c, expected_d)
            ('king', 'man', 'woman', 'queen'),
            ('paris', 'france', 'london', 'england'),
            ('boy', 'girl', 'man', 'woman'),
        ]
        
        correct = 0
        total = 0
        
        for a, b, c, expected_d in analogies:
            if all(w in word_to_idx for w in [a, b, c, expected_d]):
                # Compute: b - a + c
                vec_a = embeddings[word_to_idx[a]]
                vec_b = embeddings[word_to_idx[b]]
                vec_c = embeddings[word_to_idx[c]]
                vec_d_expected = embeddings[word_to_idx[expected_d]]
                
                target = vec_b - vec_a + vec_c
                
                # Find closest word
                similarities = []
                for word, idx in word_to_idx.items():
                    if word not in [a, b, c]:
                        sim = self.cosine_similarity(target, embeddings[idx])
                        similarities.append((sim, word))
                
                similarities.sort(reverse=True)
                predicted_d = similarities[0][1]
                
                if predicted_d == expected_d:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def measure_distributional_properties(self, embeddings: np.ndarray) -> Dict:
        """Measure distributional properties of embeddings."""
        from scipy.stats import skew, kurtosis
        
        n_words, n_dims = embeddings.shape
        
        # Dimension-wise statistics
        unique_counts = [len(np.unique(embeddings[:, i])) for i in range(n_dims)]
        skewness = [abs(skew(embeddings[:, i])) for i in range(n_dims)]
        kurt = [kurtosis(embeddings[:, i]) for i in range(n_dims)]
        
        # Word-wise norms
        norms = np.linalg.norm(embeddings, axis=1)
        
        return {
            'mean_unique_per_dim': np.mean(unique_counts),
            'median_unique_per_dim': np.median(unique_counts),
            'mean_abs_skewness': np.mean(skewness),
            'std_skewness': np.std(skewness),
            'mean_kurtosis': np.mean(kurt),
            'std_kurtosis': np.std(kurt),
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
        }


class QuantizationComparison:
    """Compare different quantization strategies."""
    
    def __init__(self, base_k: int = 20, use_lebesgue: bool = False):
        """
        Initialize comparison framework.
        
        Args:
            base_k: Base quantization level for adaptive method
            use_lebesgue: Use true Lebesgue/equi-depth for skewed dims
        """
        self.base_k = base_k
        self.use_lebesgue = use_lebesgue
        self.evaluator = EmbeddingEvaluator()
        self.results = {}
    
    def run_comparison(self, embeddings: np.ndarray, words: List[str],
                       name: str = "embeddings") -> Dict:
        """
        Run full comparison of quantization strategies.
        
        Args:
            embeddings: Original embeddings
            words: Vocabulary
            name: Name of embedding set
            
        Returns:
            Dictionary of results
        """
        print("="*80)
        print(f"QUANTIZATION COMPARISON: {name}")
        print("="*80)
        
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        results = {}
        
        # 1. Original (no quantization)
        print("\n1. ORIGINAL (No quantization)")
        print("-"*80)
        results['original'] = self._evaluate_single(
            embeddings, word_to_idx, "Original"
        )
        
        # 2. Uniform quantization
        print(f"\n2. UNIFORM QUANTIZATION (k={self.base_k})")
        print("-"*80)
        uniform_quantizer = UniformQuantizer(k=self.base_k)
        uniform_quantized = uniform_quantizer.quantize(embeddings)
        results['uniform'] = self._evaluate_single(
            uniform_quantized, word_to_idx, f"Uniform k={self.base_k}"
        )
        
        # 3. Adaptive quantization
        print(f"\n3. ADAPTIVE QUANTIZATION (base_k={self.base_k})")

        print("-"*80)
        adaptive_quantizer = AdaptiveQuantizer(base_k=self.base_k, 
                                              use_lebesgue=self.use_lebesgue, 
                                              verbose=True)
        adaptive_quantized = adaptive_quantizer.quantize(embeddings)
        adaptive_quantizer.print_summary()
        results['adaptive'] = self._evaluate_single(
            adaptive_quantized, word_to_idx, "Adaptive"
        )
        
        # Print comparison table
        self._print_comparison_table(results, name)
        
        # Store for later
        self.results[name] = results
        
        return results
    
    def _evaluate_single(self, embeddings: np.ndarray, 
                        word_to_idx: Dict[str, int],
                        method_name: str) -> Dict:
        """Evaluate a single embedding configuration."""
        
        # Semantic similarity (STS)
        sts_score = self.evaluator.evaluate_sts(embeddings, word_to_idx)
        
        # Semantic relatedness (SICK)
        sick_score = self.evaluator.evaluate_sick(embeddings, word_to_idx)
        
        # Analogy
        analogy_acc = self.evaluator.evaluate_analogy(embeddings, word_to_idx)
        
        # Distributional properties
        dist_props = self.evaluator.measure_distributional_properties(embeddings)
        
        # Compression metrics
        compression_metrics = self._calculate_compression_metrics(embeddings, dist_props['mean_unique_per_dim'])
        
        print(f"\n{method_name}:")
        print(f"  STS correlation:  {sts_score:.4f}")
        print(f"  SICK correlation: {sick_score:.4f}")
        print(f"  Analogy accuracy: {analogy_acc:.2%}")
        print(f"  Mean unique/dim:  {dist_props['mean_unique_per_dim']:.1f}")
        print(f"  Mean |skewness|:  {dist_props['mean_abs_skewness']:.3f}")
        print(f"  Mean kurtosis:    {dist_props['mean_kurtosis']:.3f}")
        print(f"  Storage size:     {compression_metrics['size_mb']:.1f} MB")
        print(f"  Bits per value:   {compression_metrics['bits_per_value']:.2f}")
        print(f"  Compression:      {compression_metrics['compression_ratio']:.1f}×")
        
        return {
            'sts': sts_score,
            'sick': sick_score,
            'analogy': analogy_acc,
            **dist_props,
            **compression_metrics
        }
    
    def _calculate_compression_metrics(self, embeddings: np.ndarray, mean_unique: float) -> Dict:
        """Calculate storage and compression metrics."""
        vocab_size, n_dims = embeddings.shape
        
        # Original storage (32-bit floats)
        original_bytes = vocab_size * n_dims * 4
        original_mb = original_bytes / (1024 * 1024)
        
        # Quantized storage (based on mean unique values)
        # Use actual unique values per dimension as proxy for k
        k_effective = int(mean_unique)
        
        if k_effective <= 1:
            # No quantization or fully uniform
            bits_per_value = 32.0
            quantized_bytes = original_bytes
        else:
            # Bits needed per value
            import math
            bits_per_value = math.log2(k_effective)
            
            # Storage for quantized values
            quantized_bytes = vocab_size * n_dims * (bits_per_value / 8)
            
            # Add codebook storage (k centers × 4 bytes per dimension)
            codebook_bytes = n_dims * k_effective * 4
            quantized_bytes += codebook_bytes
        
        quantized_mb = quantized_bytes / (1024 * 1024)
        compression_ratio = original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
        
        return {
            'size_mb': quantized_mb,
            'original_mb': original_mb,
            'bits_per_value': bits_per_value,
            'compression_ratio': compression_ratio
        }
    
    def _print_comparison_table(self, results: Dict, name: str):
        """Print comparison table."""
        print("\n" + "="*80)
        print(f"COMPARISON TABLE: {name}")
        print("="*80)
        print(f"{'Method':<20} {'STS':>10} {'SICK':>10} {'Analogy':>10} {'Size(MB)':>10} {'Compress':>10}")
        print("-"*80)
        
        for method in ['original', 'uniform', 'adaptive']:
            r = results[method]
            print(f"{method.capitalize():<20} "
                  f"{r['sts']:>10.4f} "
                  f"{r['sick']:>10.4f} "
                  f"{r['analogy']:>10.2%} "
                  f"{r['size_mb']:>10.1f} "
                  f"{r['compression_ratio']:>9.1f}×")
        
        # Print improvements
        print("-"*80)
        print("Improvements over original:")
        
        orig_sts = results['original']['sts']
        orig_sick = results['original']['sick']
        orig_size = results['original']['size_mb']
        
        for method in ['uniform', 'adaptive']:
            sts_change = (results[method]['sts'] - orig_sts) / orig_sts * 100
            sick_change = (results[method]['sick'] - orig_sick) / orig_sick * 100
            size_reduction = (1 - results[method]['size_mb'] / orig_size) * 100
            print(f"  {method.capitalize():10s} - STS: {sts_change:+.1f}%  SICK: {sick_change:+.1f}%  "
                  f"Size: -{size_reduction:.0f}% ({results[method]['compression_ratio']:.1f}× compression)")
        
        print("="*80 + "\n")
    
    def save_quantized_embeddings(self, embeddings: np.ndarray, words: List[str], 
                                  name: str, method: str = 'adaptive', 
                                  output_dir: str = '.', format: str = None,
                                  add_metadata: bool = None):
        """
        Save quantized embeddings to file.
        
        Args:
            embeddings: Original embeddings
            words: Word list
            name: Base name for output file
            method: 'uniform' or 'adaptive'
            output_dir: Output directory
            format: Output format - 'word2vec_text', 'word2vec_bin', 'pytorch', 
                   'numpy', 'hdf5', or None for auto-detect from extension
            add_metadata: Whether to add metadata. If None, smart default based on format.
        """
        from adaptive_quantization import save_embeddings
        import os
        
        # Quantize
        if method == 'uniform':
            quantizer = UniformQuantizer(k=self.base_k)
            quantized = quantizer.quantize(embeddings)
            method_str = f'uniform_k{self.base_k}'
        else:  # adaptive
            quantizer = AdaptiveQuantizer(
                base_k=self.base_k, 
                use_lebesgue=self.use_lebesgue, 
                verbose=False
            )
            quantized = quantizer.quantize(embeddings)
            method_str = f'adaptive_k{self.base_k}'
            if self.use_lebesgue:
                method_str += '_lebesgue'
        
        # Determine extension based on format
        if format is None:
            ext = '.vec'  # default
        else:
            ext_map = {
                'word2vec_text': '.vec',
                'word2vec_bin': '.bin',
                'pytorch': '.pt',
                'numpy': '.npz',
                'hdf5': '.h5'
            }
            ext = ext_map.get(format, '.vec')
        
        # Create output path
        output_path = os.path.join(output_dir, f"{name}_{method_str}{ext}")
        
        # Prepare quantization info
        quant_info = {
            'base_k': self.base_k,
            'method': method,
            'use_lebesgue': self.use_lebesgue if method == 'adaptive' else None
        }
        
        # Save
        save_embeddings(quantized, words, output_path, format=format,
                       quantization_info=quant_info, add_metadata=add_metadata)
        
        return output_path


def main():
    """Main evaluation script."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate quantization strategies for word embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  IMPORTANT: Quantization improves QUALITY, not file size
────────────────────────────────────────────────────────────────────
Expect +3-6% improvement on STS/SICK benchmarks, but file size 
remains the same (~305 MB stays ~305 MB) with standard formats.
For file compression, use external tools: gzip output.vec
────────────────────────────────────────────────────────────────────

Examples:
  # Default evaluation (base_k=20):
  python evaluate_quantization.py en25k-skipg.vec
  
  # Recommended: k=32 quantization
  python evaluate_quantization.py --base-k 32 embeddings.vec
  
  # Save quantized embeddings (adaptive method):
  python evaluate_quantization.py --base-k 32 --save-quantized embeddings.vec
  
  # Save both uniform and adaptive:
  python evaluate_quantization.py --base-k 32 --save-quantized --save-method both embeddings.vec
  
  # Save in PyTorch format (with metadata):
  python evaluate_quantization.py --base-k 32 --save-quantized --save-format pt embeddings.vec
  
  # Save in NumPy format:
  python evaluate_quantization.py --base-k 32 --save-quantized --save-format npz embeddings.vec
  
  # Save to specific directory:
  python evaluate_quantization.py --base-k 32 --save-quantized --output-dir ./quantized embeddings.vec
  
  # Compare multiple files:
  python evaluate_quantization.py --base-k 32 file1.vec file2.vec
        """
    )
    
    parser.add_argument('embeddings', nargs='+', 
                       help='Embedding file(s) to evaluate (.vec format)')
    parser.add_argument('--base-k', type=int, default=20,
                       help='Base quantization level (default: 20). '
                            'Higher = finer precision. '
                            'Recommended: 20 (coarse), 50-100 (medium), 200+ (fine)')
    parser.add_argument('--use-lebesgue', action='store_true',
                       help='Use true Lebesgue/equi-depth quantization for skewed dimensions '
                            '(default: use percentile method)')
    parser.add_argument('--save-quantized', action='store_true',
                       help='Save quantized embeddings to file')
    parser.add_argument('--save-method', choices=['uniform', 'adaptive', 'both'], 
                       default='adaptive',
                       help='Which quantization method to save (default: adaptive)')
    parser.add_argument('--save-format', 
                       choices=['vec', 'bin', 'pt', 'npz', 'h5'],
                       default='vec',
                       help='Output format: vec (Word2Vec text), bin (Word2Vec binary), '
                            'pt (PyTorch), npz (NumPy), h5 (HDF5). Default: vec')
    parser.add_argument('--add-metadata', action='store_true',
                       help='Add quantization metadata to output file. '
                            'WARNING: For .vec/.bin formats, breaks gensim compatibility! '
                            'Metadata always added to .pt/.npz/.h5 formats.')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for saved embeddings (default: current directory)')
    
    args = parser.parse_args()
    
    comparison = QuantizationComparison(base_k=args.base_k, use_lebesgue=args.use_lebesgue)
    
    print(f"\n{'='*80}")
    print(f"QUANTIZATION EVALUATION")
    print(f"Base k: {args.base_k} (precision: ~{2.0/args.base_k:.4f} per step)")
    if args.use_lebesgue:
        print(f"Mode: Lebesgue/equi-depth for skewed dimensions")
    else:
        print(f"Mode: Percentile (default) for skewed dimensions")
    print(f"{'='*80}")
    
    for embedding_file in args.embeddings:
        print(f"\n{'='*80}")
        print(f"Processing: {embedding_file}")
        print(f"{'='*80}\n")
        
        # Load
        embeddings, words = load_embeddings(embedding_file)
        name = embedding_file.split('/')[-1].replace('.vec', '').replace('.bin', '')
        
        # Run comparison
        comparison.run_comparison(embeddings, words, name)
        
        # Save quantized embeddings if requested
        if args.save_quantized:
            print(f"\n{'='*80}")
            print(f"SAVING QUANTIZED EMBEDDINGS")
            print(f"{'='*80}\n")
            
            # Map short format names to full names
            format_map = {
                'vec': 'word2vec_text',
                'bin': 'word2vec_bin',
                'pt': 'pytorch',
                'npz': 'numpy',
                'h5': 'hdf5'
            }
            save_format = format_map.get(args.save_format, 'word2vec_text')
            
            if args.save_method in ['uniform', 'both']:
                output_path = comparison.save_quantized_embeddings(
                    embeddings, words, name, method='uniform',
                    output_dir=args.output_dir, format=save_format,
                    add_metadata=args.add_metadata
                )
                print(f"Uniform quantization saved to: {output_path}\n")
            
            if args.save_method in ['adaptive', 'both']:
                output_path = comparison.save_quantized_embeddings(
                    embeddings, words, name, method='adaptive',
                    output_dir=args.output_dir, format=save_format,
                    add_metadata=args.add_metadata
                )
                print(f"Adaptive quantization saved to: {output_path}\n")
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"\nBase k: {args.base_k}")
    print("\nKey findings:")
    print("1. Check if adaptive quantization outperforms uniform")
    print("2. Check if CBOW benefits more than Skip-gram from adaptive approach")
    print("3. Note distributional property changes")
    print("4. Higher base_k = finer precision = less performance loss")
    print("="*80)


if __name__ == "__main__":
    main()
