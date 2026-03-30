"""
Dataset readers for vector similarity join benchmarks.

Supported formats:
  - .u8bin: 8-byte header (n, d as uint32) + row-major uint8 data
  - .fbin:  8-byte header (n, d as uint32) + row-major float32 data
  - .fvecs: per-row int32 dim prefix + float32 data
  - .bvecs: per-row int32 dim prefix + uint8 data
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


def read_fbin(path, max_vectors=None):
    """Load a .fbin file into a float32 numpy array.

    Format: 4-byte uint32 n, 4-byte uint32 d, then n*d float32 values.
    """
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        d = int.from_bytes(f.read(4), 'little')
    print(f"fbin header: {n:,} vectors x {d}D")

    if max_vectors:
        n = min(n, max_vectors)

    data = np.memmap(path, dtype='float32', mode='r', offset=8, shape=(n, d))
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


def read_bvecs(path, max_vectors=None):
    """Load a .bvecs file into a float32 numpy array.

    Format: repeating [4-byte int32 dim, dim uint8 values] per row.
    """
    with open(path, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        f.seek(0)
        x = np.fromfile(f, dtype='uint8')

    row_bytes = 4 + dim  # 4 bytes for dim prefix + dim bytes of data
    n = len(x) // row_bytes
    if max_vectors:
        n = min(n, max_vectors)
    vectors = x[:n * row_bytes].reshape(n, row_bytes)[:, 4:].astype(np.float32)

    print(f"Loaded {vectors.shape[0]:,} x {vectors.shape[1]} from {path}")
    return vectors
