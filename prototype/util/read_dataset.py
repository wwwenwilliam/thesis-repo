"""
Dataset readers for vector similarity join benchmarks.

Supported formats:
  - .u8bin: 8-byte header (n, d as uint32) + row-major uint8 data
  - .fvecs: per-row int32 dim prefix + float32 data
"""
import numpy as np


def read_u8bin(path, max_vectors=None):
    """Load a .u8bin file into a float32 numpy array.

    Args:
        path: Path to the .u8bin file.
        max_vectors: If set, load at most this many vectors.

    Returns:
        numpy float32 array of shape (n, d).
    """
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        d = int.from_bytes(f.read(4), 'little')
    print(f"u8bin header: {n:,} vectors x {d}D")

    if max_vectors:
        n = min(n, max_vectors)

    data = np.memmap(path, dtype='uint8', mode='r', offset=8, shape=(n, d))
    vectors = np.array(data[:n], dtype=np.float32)
    print(f"Loaded {vectors.shape[0]:,} x {vectors.shape[1]} from {path}")
    return vectors


def read_fvecs(path, max_vectors=None):
    """Load a .fvecs file into a float32 numpy array.

    Args:
        path: Path to the .fvecs file.
        max_vectors: If set, load at most this many vectors.

    Returns:
        numpy float32 array of shape (n, d).
    """
    with open(path, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        x = np.fromfile(f, dtype='float32')

    vectors = x.reshape(-1, dim + 1)[:, 1:].copy()
    if max_vectors:
        vectors = vectors[:max_vectors]

    print(f"Loaded {vectors.shape[0]:,} x {vectors.shape[1]} from {path}")
    return vectors
