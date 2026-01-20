#!/usr/bin/env python3
"""
Batch Embedding Conversion Example

Demonstrates how to convert multiple embedding files at once
with various quantization strategies.

Usage:
    python batch_conversion.py input_dir/ output_dir/ --k 32
    python batch_conversion.py *.vec --output-dir quantized/

Author: Based on research by Kow Kuroda
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from adaptive_quantization import AdaptiveQuantizer, load_embeddings, save_embeddings


def convert_file(input_path, output_dir, base_k=32, use_lebesgue=True, 
                format='word2vec_text', suffix='_quantized'):
    """
    Convert a single embedding file.
    
    Args:
        input_path: Path to input embedding file
        output_dir: Output directory
        base_k: Quantization level
        use_lebesgue: Use adaptive Lebesgue quantization
        format: Output format ('word2vec_text', 'word2vec_bin', 'pytorch', 'numpy', 'hdf5')
        suffix: Suffix to add to output filename
    
    Returns:
        Tuple of (input_size_mb, output_size_mb, compression_ratio)
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output path
    basename = Path(input_path).stem  # filename without extension
    ext_map = {
        'word2vec_text': '.vec',
        'word2vec_bin': '.bin',
        'pytorch': '.pt',
        'numpy': '.npz',
        'hdf5': '.h5'
    }
    ext = ext_map.get(format, '.vec')
    output_filename = f"{basename}{suffix}_k{base_k}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\nProcessing: {input_path}")
    print(f"Output:     {output_path}")
    
    # Load
    embeddings, words = load_embeddings(input_path)
    vocab_size, n_dims = embeddings.shape
    input_size = embeddings.nbytes / (1024**2)
    
    print(f"  Loaded {vocab_size:,} words × {n_dims}d ({input_size:.1f} MB)")
    
    # Quantize
    quantizer = AdaptiveQuantizer(
        base_k=base_k,
        use_lebesgue=use_lebesgue,
        verbose=False
    )
    quantized = quantizer.quantize(embeddings)
    
    # Save
    quant_info = {
        'base_k': base_k,
        'method': 'adaptive',
        'use_lebesgue': use_lebesgue
    }
    save_embeddings(quantized, words, output_path, 
                   format=format, quantization_info=quant_info)
    
    # Calculate metrics
    output_size = os.path.getsize(output_path) / (1024**2)
    compression = input_size / output_size if output_size > 0 else 1.0
    
    print(f"  Compression: {input_size:.1f} MB → {output_size:.1f} MB ({compression:.1f}×)")
    
    return input_size, output_size, compression


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert multiple embedding files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all .vec files in current directory:
  python batch_conversion.py *.vec --output-dir quantized/
  
  # Convert all files in a directory:
  python batch_conversion.py embeddings/*.vec --output-dir compressed/ --k 32
  
  # Save as PyTorch format:
  python batch_conversion.py *.vec --output-dir quantized/ --format pt
  
  # Save as NumPy compressed:
  python batch_conversion.py *.vec --output-dir quantized/ --format npz
  
  # Save as HDF5:
  python batch_conversion.py *.vec --output-dir quantized/ --format h5
  
  # Custom suffix:
  python batch_conversion.py *.vec --output-dir out/ --suffix _compressed

Supported Formats:
  vec  → Word2Vec text (default)
  bin  → Word2Vec binary (requires gensim)
  pt   → PyTorch (with metadata, requires torch)
  npz  → NumPy compressed (with metadata)
  h5   → HDF5 (scientific standard, requires h5py)
        """
    )
    
    parser.add_argument('inputs', nargs='+',
                       help='Input embedding files (supports wildcards)')
    parser.add_argument('--output-dir', default='quantized',
                       help='Output directory (default: quantized/)')
    parser.add_argument('--k', type=int, default=32,
                       help='Quantization level (default: 32)')
    parser.add_argument('--format', 
                       choices=['vec', 'bin', 'pt', 'npz', 'h5'],
                       default='vec',
                       help='Output format (default: vec)')
    parser.add_argument('--no-lebesgue', action='store_true',
                       help='Disable Lebesgue quantization')
    parser.add_argument('--suffix', default='_quantized',
                       help='Suffix for output files (default: _quantized)')
    
    args = parser.parse_args()
    
    # Expand wildcards
    input_files = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        if matched:
            input_files.extend(matched)
        else:
            # If no match, treat as literal filename
            if os.path.exists(pattern):
                input_files.append(pattern)
            else:
                print(f"Warning: No files match pattern: {pattern}", file=sys.stderr)
    
    if not input_files:
        print("Error: No input files found", file=sys.stderr)
        return 1
    
    # Map short format names to full names
    format_map = {
        'vec': 'word2vec_text',
        'bin': 'word2vec_bin',
        'pt': 'pytorch',
        'npz': 'numpy',
        'h5': 'hdf5'
    }
    output_format = format_map.get(args.format, 'word2vec_text')
    
    print("="*80)
    print("BATCH EMBEDDING CONVERSION")
    print("="*80)
    print(f"Files to process: {len(input_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quantization: k={args.k}, method=adaptive" + 
          ("" if not args.no_lebesgue else " (no Lebesgue)"))
    print(f"Format: {args.format.upper()}")
    print("="*80)
    
    # Process files
    total_input = 0
    total_output = 0
    successful = 0
    failed = 0
    
    for filepath in input_files:
        try:
            input_size, output_size, compression = convert_file(
                filepath,
                args.output_dir,
                base_k=args.k,
                use_lebesgue=not args.no_lebesgue,
                format=output_format,
                suffix=args.suffix
            )
            total_input += input_size
            total_output += output_size
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("BATCH CONVERSION SUMMARY")
    print("="*80)
    print(f"Files processed:    {successful}/{len(input_files)}")
    if failed > 0:
        print(f"Failed:             {failed}")
    print(f"Total input size:   {total_input:.1f} MB")
    print(f"Total output size:  {total_output:.1f} MB")
    if total_output > 0:
        overall_compression = total_input / total_output
        savings = total_input - total_output
        print(f"Overall compression: {overall_compression:.1f}× (saved {savings:.1f} MB)")
    print("="*80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
