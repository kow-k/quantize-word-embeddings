#!/usr/bin/env python3
"""
Visualization Tools for Adaptive Quantization Analysis

Creates publication-quality figures showing:
1. Dimension type distributions
2. Before/after quantization comparisons
3. Performance improvements
4. Distribution shape changes

Author: Based on research by Kow Kuroda
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict
from adaptive_quantization import AdaptiveQuantizer, load_embeddings
from scipy.stats import skew, kurtosis


class QuantizationVisualizer:
    """Create visualizations for adaptive quantization analysis."""
    
    def __init__(self, figsize_base=(12, 8)):
        self.figsize_base = figsize_base
        self.colors = {
            'gaussian': '#2E86AB',
            'right_skewed': '#A23B72',
            'left_skewed': '#F18F01',
            'multimodal': '#C73E1D',
            'heavy_tailed': '#6A994E',
            'other': '#999999'
        }
    
    def plot_dimension_types(self, quantizer: AdaptiveQuantizer, 
                            save_path: str = None):
        """Plot distribution of dimension types."""
        summary = quantizer.get_summary()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Distribution types pie chart
        types = summary['distribution_types']
        labels = list(types.keys())
        sizes = list(types.values())
        colors = [self.colors.get(t, '#999999') for t in labels]
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Distribution Types Across Dimensions', fontsize=14, fontweight='bold')
        
        # Plot 2: Quantization methods bar chart
        methods = summary['quantization_methods']
        method_names = list(methods.keys())
        counts = list(methods.values())
        
        bars = ax2.bar(method_names, counts, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Quantization Method', fontsize=12)
        ax2.set_ylabel('Number of Dimensions', fontsize=12)
        ax2.set_title('Quantization Method Selection', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved dimension types plot to: {save_path}")
        
        return fig
    
    def plot_dimension_profiles(self, quantizer: AdaptiveQuantizer,
                               n_samples: int = 20,
                               save_path: str = None):
        """Plot sample dimension profiles showing before/after."""
        profiles = quantizer.dimension_profiles
        
        # Sample evenly across dimensions
        n_dims = len(profiles)
        sample_indices = np.linspace(0, n_dims-1, min(n_samples, n_dims), dtype=int)
        
        # Create grid
        n_cols = 5
        n_rows = (len(sample_indices) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, dim_idx in enumerate(sample_indices):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            profile = profiles[dim_idx]
            
            # Create simple visualization showing profile info
            ax.axis('off')
            
            # Text info
            info_text = (
                f"Dimension {profile.dim_id}\n"
                f"Type: {profile.dist_type}\n"
                f"Skew: {profile.skewness:.2f}\n"
                f"Kurt: {profile.kurtosis:.2f}\n"
                f"k: {profile.optimal_k}\n"
                f"Method: {profile.quantization_method}"
            )
            
            color = self.colors.get(profile.dist_type, '#999999')
            ax.text(0.5, 0.5, info_text, 
                   ha='center', va='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        # Hide unused subplots
        for idx in range(len(sample_indices), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Sample Dimension Profiles', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved dimension profiles to: {save_path}")
        
        return fig
    
    def plot_before_after_distributions(self, original: np.ndarray,
                                       quantized: np.ndarray,
                                       n_samples: int = 9,
                                       save_path: str = None):
        """Plot before/after histograms for sample dimensions."""
        n_dims = original.shape[1]
        sample_indices = np.linspace(0, n_dims-1, n_samples, dtype=int)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, dim_idx in enumerate(sample_indices):
            ax = axes[idx]
            
            # Original distribution
            ax.hist(original[:, dim_idx], bins=50, alpha=0.6, 
                   color='#2E86AB', edgecolor='black', linewidth=0.5,
                   label='Original')
            
            # Quantized distribution
            ax.hist(quantized[:, dim_idx], bins=30, alpha=0.6,
                   color='#F18F01', edgecolor='black', linewidth=0.5,
                   label='Quantized')
            
            # Statistics
            orig_skew = skew(original[:, dim_idx])
            quant_skew = skew(quantized[:, dim_idx])
            
            ax.set_title(f'Dim {dim_idx}\n'
                        f'Skew: {orig_skew:.2f} → {quant_skew:.2f}',
                        fontsize=10)
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Before/After Quantization: Sample Dimensions',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved before/after distributions to: {save_path}")
        
        return fig
    
    def plot_performance_comparison(self, results: Dict,
                                   save_path: str = None):
        """Plot performance comparison across methods."""
        methods = list(results.keys())
        sts_scores = [results[m]['sts'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#999999', '#2E86AB', '#A23B72']
        bars = ax.bar(methods, sts_scores, color=colors, 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, score in zip(bars, sts_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add improvement annotations
        baseline = sts_scores[0]
        for i, (method, score) in enumerate(zip(methods[1:], sts_scores[1:]), 1):
            change = (score - baseline) / baseline * 100
            y_pos = max(sts_scores) * 1.05
            ax.text(i, y_pos, f'{change:+.1f}%', 
                   ha='center', fontsize=10, 
                   color='green' if change > 0 else 'red',
                   fontweight='bold')
        
        ax.set_ylabel('STS Correlation (ρ)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Comparison: Quantization Methods', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(sts_scores) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved performance comparison to: {save_path}")
        
        return fig
    
    def create_comparison_figure(self, skipgram_results: Dict,
                                 cbow_results: Dict,
                                 save_path: str = None):
        """Create side-by-side comparison for Skip-gram vs CBOW."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = ['Original', 'Uniform', 'Adaptive']
        
        # Skip-gram
        skipgram_scores = [skipgram_results[m.lower()]['sts'] for m in methods]
        bars1 = ax1.bar(methods, skipgram_scores, 
                       color=['#999999', '#2E86AB', '#A23B72'],
                       edgecolor='black', linewidth=1.5)
        
        for bar, score in zip(bars1, skipgram_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('STS Correlation', fontsize=12)
        ax1.set_title('Skip-gram (Gaussian)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(skipgram_scores) * 1.15)
        ax1.grid(axis='y', alpha=0.3)
        
        # CBOW
        cbow_scores = [cbow_results[m.lower()]['sts'] for m in methods]
        bars2 = ax2.bar(methods, cbow_scores,
                       color=['#999999', '#2E86AB', '#A23B72'],
                       edgecolor='black', linewidth=1.5)
        
        for bar, score in zip(bars2, cbow_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('STS Correlation', fontsize=12)
        ax2.set_title('CBOW (Skewed)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(max(skipgram_scores), max(cbow_scores)) * 1.15)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Architecture Comparison: Adaptive Quantization Benefits',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved architecture comparison to: {save_path}")
        
        return fig


# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize quantization effects on word embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (base_k=20):
  python visualize_quantization.py embeddings.vec
  
  # Fine-grained quantization:
  python visualize_quantization.py --base-k 100 embeddings.vec
  
  # Very fine quantization:
  python visualize_quantization.py --base-k 200 embeddings.vec
        """
    )
    
    parser.add_argument('embedding', help='Embedding file to visualize (.vec format)')
    parser.add_argument('--base-k', type=int, default=20,
                       help='Base quantization level (default: 20). '
                            'Higher = finer precision. '
                            'Recommended: 20 (coarse), 50-100 (medium), 200+ (fine)')
    parser.add_argument('--use-lebesgue', action='store_true',
                       help='Use true Lebesgue/equi-depth quantization for skewed dimensions '
                            '(default: use percentile method)')
    
    args = parser.parse_args()
    
    filepath = args.embedding
    name = filepath.split('/')[-1].replace('.vec', '')
    
    print(f"Loading and analyzing: {filepath}")
    print(f"Base k: {args.base_k} (precision: ~{2.0/args.base_k:.4f} per step)")
    if args.use_lebesgue:
        print(f"Mode: Lebesgue/equi-depth for skewed dimensions")
    else:
        print(f"Mode: Percentile (default) for skewed dimensions")
    
    embeddings, words = load_embeddings(filepath)
    
    # Analyze with adaptive quantization
    quantizer = AdaptiveQuantizer(base_k=args.base_k, use_lebesgue=args.use_lebesgue, verbose=True)
    quantized = quantizer.quantize(embeddings)
    
    # Create visualizations
    viz = QuantizationVisualizer()
    
    print("\nCreating visualizations...")
    
    # Add base_k and method to output filenames for clarity
    method_suffix = "_lebesgue" if args.use_lebesgue else ""
    suffix = f"_k{args.base_k}{method_suffix}" if (args.base_k != 20 or args.use_lebesgue) else ""
    
    # Dimension types
    viz.plot_dimension_types(quantizer, f"{name}{suffix}_dimension_types.png")
    
    # Dimension profiles
    viz.plot_dimension_profiles(quantizer, n_samples=20, save_path=f"{name}{suffix}_profiles.png")
    
    # Before/after distributions
    viz.plot_before_after_distributions(embeddings, quantized, 
                                       n_samples=9, save_path=f"{name}{suffix}_before_after.png")
    
    print("\nVisualization complete!")
    print(f"\nOutput files:")
    print(f"  {name}{suffix}_dimension_types.png")
    print(f"  {name}{suffix}_profiles.png")
    print(f"  {name}{suffix}_before_after.png")

