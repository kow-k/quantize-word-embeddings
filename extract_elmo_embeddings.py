#!/usr/bin/env python3
"""
ELMo Pseudo-Static Embedding Extractor

Extracts vocabulary from STS benchmark datasets, embeds each word as a
single-word sentence through ELMo (NLPL format), and averages the 3 ELMo
layers to produce one static 1024-d vector per word.

Output is a .vec or .h5 file ready for evaluate_quantization.py.

Requirements:
    pip install simple_elmo h5py

Usage:
    # Basic: embed all STS words, save as .vec
    python extract_elmo_embeddings.py \\
        --elmo-dir models-open/Wiki-2017-ELMo/ \\
        --sts-dir data/sts/ \\
        --output elmo_sts_vocab.vec

    # Save as HDF5
    python extract_elmo_embeddings.py \\
        --elmo-dir models-open/Wiki-2017-ELMo/ \\
        --sts-dir data/sts/ \\
        --output elmo_sts_vocab.h5

    # Use only ELMo layer 2 (top LSTM layer) instead of averaging all 3
    python extract_elmo_embeddings.py \\
        --elmo-dir models-open/Wiki-2017-ELMo/ \\
        --sts-dir data/sts/ \\
        --output elmo_sts_vocab.vec \\
        --layer 2

    # Add extra vocab from a plain word-list file (one word per line)
    python extract_elmo_embeddings.py \\
        --elmo-dir models-open/Wiki-2017-ELMo/ \\
        --sts-dir data/sts/ \\
        --output elmo_sts_vocab.vec \\
        --extra-vocab extra_words.txt

    # Then evaluate quantization on the result:
    python evaluate_quantization.py --base-k 8 elmo_sts_vocab.vec

Notes on layer selection:
    --layer all  Average of all 3 layers (default, matches ELMo paper)
    --layer 0    Character CNN output
    --layer 1    LSTM layer 1
    --layer 2    LSTM layer 2 (top)

Author: Based on research by Kow Kuroda
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary extraction from STS datasets
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Whitespace tokenizer with light punctuation stripping. No language-specific rules."""
    tokens = []
    for tok in text.lower().split():
        tok = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", tok)
        if tok:
            tokens.append(tok)
    return tokens


def extract_sts_vocabulary(sts_dir: str,
                            extra_vocab_file: Optional[str] = None) -> List[str]:
    """
    Walk an STS data directory and collect unique words from all sentence pairs.

    Handles the main STS formats:
      - Tab-separated, sentences in columns 5 & 6 (STS 2012-2016)
      - Tab-separated, sentences in columns 7 & 8 (STS Benchmark / GLUE)
      - Plain sentence pairs separated by a tab (two-column)

    Returns a sorted, deduplicated word list.
    """
    sts_dir = Path(sts_dir)
    if not sts_dir.exists():
        raise FileNotFoundError(f"STS directory not found: {sts_dir}")

    vocab: set = set()
    files_processed = 0

    for path in sorted(sts_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".txt", ".tsv", ".csv", ""}:
            continue
        if path.stem.lower() in {"readme", "license", "licence"}:
            continue

        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    parts = line.split("\t")

                    candidate_texts = []
                    if len(parts) >= 8:
                        # STS Benchmark / GLUE style
                        candidate_texts = [parts[5], parts[6]]
                    elif len(parts) >= 6:
                        # STS 2012-2016 style
                        candidate_texts = [parts[4], parts[5]]
                    elif len(parts) == 2:
                        candidate_texts = [parts[0], parts[1]]
                    elif len(parts) == 1 and parts[0]:
                        candidate_texts = [parts[0]]

                    for text in candidate_texts:
                        vocab.update(_tokenize(text))

            files_processed += 1
        except Exception:
            continue

    if files_processed == 0:
        raise ValueError(
            f"No readable files found in STS directory: {sts_dir}\n"
            "  Expected .txt or .tsv files with tab-separated sentence pairs."
        )

    # Add extra vocabulary from file if provided
    if extra_vocab_file:
        extra_path = Path(extra_vocab_file)
        if not extra_path.exists():
            print(f"  Warning: extra vocab file not found: {extra_vocab_file}",
                  file=sys.stderr)
        else:
            with open(extra_path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    word = line.strip().lower()
                    if word:
                        vocab.add(word)
            print(f"  Added extra vocab from: {extra_vocab_file}")

    vocab.discard("")
    word_list = sorted(vocab)
    print(f"  Scanned {files_processed} STS file(s), found {len(word_list):,} unique words")
    return word_list


# ─────────────────────────────────────────────────────────────────────────────
# ELMo embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def embed_words_elmo(words: List[str],
                     elmo_dir: str,
                     layer: str = "all",
                     batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of words using ELMo (NLPL format, via simple_elmo).

    Each word is presented as a single-word sentence so there is no
    context ambiguity.  The token vector (the only token per sentence)
    is taken after layer selection.

    Args:
        words:      Vocabulary list.
        elmo_dir:   Directory containing options.json and model.hdf5.
        layer:      '0', '1', '2', or 'all' (average all 3 layers).
        batch_size: Words per forward pass. Reduce if you hit OOM.

    Returns:
        np.ndarray of shape (len(words), 1024), dtype float32
    """
    # simple_elmo uses tf.compat.v1 LSTM cells which are incompatible with
    # Keras 3 (default in TensorFlow >= 2.16).  Setting TF_USE_LEGACY_KERAS=1
    # before the first TF import redirects TF to use tf_keras (Keras 2) instead.
    # Requires: pip install tf_keras
    import os as _os
    if "TF_USE_LEGACY_KERAS" not in _os.environ:
        _os.environ["TF_USE_LEGACY_KERAS"] = "1"

    try:
        from simple_elmo import ElmoModel
    except ImportError:
        raise ImportError(
            "simple_elmo is required.\n"
            "  Install with: pip install simple_elmo\n"
            "  Also required for TF >= 2.16: pip install tf_keras"
        )

    print(f"  Loading ELMo model from: {elmo_dir}")
    model = ElmoModel()
    model.load(str(elmo_dir))
    print(f"  Model loaded. Embedding {len(words):,} words "
          f"in batches of {batch_size}...")

    # Determine which layer indices to use
    if layer == "all":
        layer_indices = [0, 1, 2]
    else:
        idx = int(layer)
        if idx not in (0, 1, 2):
            raise ValueError(f"Layer must be 0, 1, 2, or 'all'. Got: {layer}")
        layer_indices = [idx]

    all_vecs: List[np.ndarray] = []
    n_batches = (len(words) + batch_size - 1) // batch_size

    for batch_num in range(n_batches):
        batch_words = words[batch_num * batch_size: (batch_num + 1) * batch_size]

        # simple_elmo expects a list of sentences; each sentence is a list of tokens
        sentences = [[w] for w in batch_words]

        # get_elmo_vectors returns shape (n_sentences, max_tokens, 3, dim)
        # For single-word sentences max_tokens = 1
        vectors = model.get_elmo_vectors(sentences, warmup=False)

        for i in range(len(batch_words)):
            # vectors[i] shape: (1, 3, 1024)  →  token_layers: (3, 1024)
            token_layers = vectors[i, 0, :, :]
            if len(layer_indices) == 1:
                vec = token_layers[layer_indices[0]]
            else:
                vec = token_layers[layer_indices, :].mean(axis=0)
            all_vecs.append(vec.astype(np.float32))

        done = min((batch_num + 1) * batch_size, len(words))
        print(f"    {done:,}/{len(words):,} words embedded...", end="\r")

    print()  # clear progress line
    return np.array(all_vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_vec(embeddings: np.ndarray, words: List[str], filepath: str):
    """Save as Word2Vec text format (.vec)."""
    vocab_size, dim = embeddings.shape
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(f"{vocab_size} {dim}\n")
        for word, vec in zip(words, embeddings):
            vec_str = " ".join(f"{v:.6f}" for v in vec)
            fh.write(f"{word} {vec_str}\n")
    size_mb = os.path.getsize(filepath) / 1024 ** 2
    print(f"  Saved {vocab_size:,} vectors ({dim}d) -> {filepath}  [{size_mb:.1f} MB]")


def save_hdf5(embeddings: np.ndarray, words: List[str], filepath: str):
    """Save as HDF5 in our own format (keys: 'embeddings', 'words')."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to save .h5 files.\n"
                          "  Install with: pip install h5py")

    vocab_size, dim = embeddings.shape
    with h5py.File(filepath, "w") as fh:
        fh.create_dataset("embeddings", data=embeddings, compression="gzip")
        dt = h5py.string_dtype(encoding="utf-8")
        fh.create_dataset("words", data=np.array(words, dtype=object), dtype=dt)
        fh.attrs["vocab_size"] = vocab_size
        fh.attrs["embedding_dim"] = dim
        fh.attrs["source"] = "ELMo (NLPL Wiki-2017)"
        fh.attrs["format_version"] = "1.0"
    size_mb = os.path.getsize(filepath) / 1024 ** 2
    print(f"  Saved {vocab_size:,} vectors ({dim}d) -> {filepath}  [{size_mb:.1f} MB]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract pseudo-static ELMo embeddings for STS vocabulary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layer notes:
  all  Average of character CNN + LSTM 1 + LSTM 2  (default, recommended)
  0    Character CNN only
  1    LSTM layer 1
  2    LSTM layer 2 (top, most context-sensitive)

Examples:
  python extract_elmo_embeddings.py \\
      --elmo-dir models-open/Wiki-2017-ELMo/ \\
      --sts-dir data/sts/ \\
      --output elmo_sts_vocab.vec

  python extract_elmo_embeddings.py \\
      --elmo-dir models-open/Wiki-2017-ELMo/ \\
      --sts-dir data/sts/ \\
      --output elmo_sts_vocab.h5 --layer 2

  python evaluate_quantization.py --base-k 8 elmo_sts_vocab.vec
        """
    )
    parser.add_argument("--elmo-dir", required=True,
                        help="Directory containing options.json and model.hdf5")
    parser.add_argument("--sts-dir", required=True,
                        help="Root directory of STS benchmark data")
    parser.add_argument("--output", required=True,
                        help="Output file (.vec or .h5)")
    parser.add_argument("--layer", default="all",
                        choices=["0", "1", "2", "all"],
                        help="ELMo layer(s) to use (default: all = average)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Words per forward pass (default: 64; "
                             "reduce to 16-32 if you hit memory errors)")
    parser.add_argument("--extra-vocab", default=None,
                        help="Plain text file with extra words, one per line")

    args = parser.parse_args()

    print("=" * 72)
    print("ELMo PSEUDO-STATIC EMBEDDING EXTRACTOR")
    print("=" * 72)
    print(f"  ELMo model : {args.elmo_dir}")
    print(f"  STS data   : {args.sts_dir}")
    print(f"  Output     : {args.output}")
    layer_label = ("average of all 3 layers" if args.layer == "all"
                   else f"layer {args.layer}")
    print(f"  Layer      : {layer_label}")
    print(f"  Batch size : {args.batch_size}")
    print("=" * 72)

    # Verify ELMo directory
    elmo_dir = Path(args.elmo_dir)
    for required in ("options.json", "model.hdf5"):
        if not (elmo_dir / required).exists():
            print(f"Error: {required} not found in {elmo_dir}", file=sys.stderr)
            sys.exit(1)

    # 1. Extract vocabulary
    print("\n[1/3] Extracting STS vocabulary...")
    words = extract_sts_vocabulary(args.sts_dir, args.extra_vocab)

    # 2. Embed
    print(f"\n[2/3] Embedding {len(words):,} words through ELMo...")
    embeddings = embed_words_elmo(
        words,
        elmo_dir=args.elmo_dir,
        layer=args.layer,
        batch_size=args.batch_size,
    )
    print(f"  Embedding matrix: {embeddings.shape[0]:,} x {embeddings.shape[1]}")

    # 3. Save
    print(f"\n[3/3] Saving to {args.output}...")
    ext = Path(args.output).suffix.lower()
    if ext in (".h5", ".hdf5"):
        save_hdf5(embeddings, words, args.output)
    else:
        if ext not in (".vec", ".txt"):
            print(f"  Note: unrecognised extension '{ext}', "
                  f"saving as Word2Vec text format")
        save_vec(embeddings, words, args.output)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"\nNext step - evaluate quantization:")
    print(f"  python evaluate_quantization.py --base-k 8 {args.output}")


if __name__ == "__main__":
    main()
