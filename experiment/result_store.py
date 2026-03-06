"""
File-based result storage for similarity join algorithms.

Saves result triplets (a_indices, b_indices, distances) to disk as raw binary
files that can be memory-mapped for reading. Chunks are appended one at a time
so the full result set never needs to be in RAM.

On-disk layout per method:
  <output_dir>/<method_name>/
    a_indices.bin   - int64 array of A indices
    b_indices.bin   - int64 array of B indices
    distances.bin   - float32 array of distances
    metadata.json   - pair_count, dtype info, timing, etc.
"""
import json
import os
import numpy as np


# Default dtypes for each array
DTYPE_INDICES = np.int64
DTYPE_DIST = np.float32


class ResultSet:
    """Handle to a saved result directory. Supports lazy loading and chunked iteration."""

    def __init__(self, result_dir):
        self.result_dir = str(result_dir)
        self._metadata = {}
        meta_path = os.path.join(self.result_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)

    @property
    def pair_count(self):
        return self._metadata.get("pair_count", 0)

    @property
    def metadata(self):
        return dict(self._metadata)

    def _mmap(self, name, dtype):
        """Memory-map a binary file."""
        path = os.path.join(self.result_dir, name)
        n = self.pair_count
        if n == 0:
            return np.empty(0, dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="r", shape=(n,))

    def load_all(self):
        """Load the full result arrays into RAM. Only use when they fit."""
        a = np.array(self._mmap("a_indices.bin", DTYPE_INDICES))
        b = np.array(self._mmap("b_indices.bin", DTYPE_INDICES))
        d = np.array(self._mmap("distances.bin", DTYPE_DIST))
        return a, b, d

    def mmap_all(self):
        """Memory-map the result arrays (zero-copy, read-only)."""
        a = self._mmap("a_indices.bin", DTYPE_INDICES)
        b = self._mmap("b_indices.bin", DTYPE_INDICES)
        d = self._mmap("distances.bin", DTYPE_DIST)
        return a, b, d

    def iter_chunks(self, chunk_size=1_000_000):
        """
        Yield (a_indices, b_indices, distances) slices without loading
        the full arrays into memory.
        """
        a, b, d = self.mmap_all()
        n = len(a)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield a[start:end].copy(), b[start:end].copy(), d[start:end].copy()

    def __len__(self):
        return self.pair_count


class ResultStore:
    """Save and load similarity join results to/from disk."""

    @staticmethod
    def save(output_dir, method_name, a_indices, b_indices, distances, metadata=None):
        """Write full result arrays to binary files."""
        result_dir = os.path.join(output_dir, method_name)
        os.makedirs(result_dir, exist_ok=True)

        a_indices.astype(DTYPE_INDICES).tofile(os.path.join(result_dir, "a_indices.bin"))
        b_indices.astype(DTYPE_INDICES).tofile(os.path.join(result_dir, "b_indices.bin"))
        distances.astype(DTYPE_DIST).tofile(os.path.join(result_dir, "distances.bin"))

        meta = {"method": method_name, "pair_count": int(len(a_indices))}
        if metadata:
            meta.update(metadata)
        with open(os.path.join(result_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  -> saved {len(a_indices):,} pairs to {result_dir}")
        return ResultSet(result_dir)

    @staticmethod
    def save_chunks(output_dir, method_name, a_chunks, b_chunks, d_chunks, metadata=None):
        """
        Append chunk lists to single binary files, freeing each chunk after write.

        Peak memory = one chunk, not the full result set.
        """
        result_dir = os.path.join(output_dir, method_name)
        os.makedirs(result_dir, exist_ok=True)

        a_path = os.path.join(result_dir, "a_indices.bin")
        b_path = os.path.join(result_dir, "b_indices.bin")
        d_path = os.path.join(result_dir, "distances.bin")

        total_pairs = 0
        with open(a_path, "wb") as fa, open(b_path, "wb") as fb, open(d_path, "wb") as fd:
            for i in range(len(a_chunks)):
                fa.write(a_chunks[i].astype(DTYPE_INDICES).tobytes())
                fb.write(b_chunks[i].astype(DTYPE_INDICES).tobytes())
                fd.write(d_chunks[i].astype(DTYPE_DIST).tobytes())
                total_pairs += len(a_chunks[i])
                # Free memory as we go
                a_chunks[i] = None
                b_chunks[i] = None
                d_chunks[i] = None

        meta = {"method": method_name, "pair_count": total_pairs}
        if metadata:
            meta.update(metadata)
        with open(os.path.join(result_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  -> saved {total_pairs:,} pairs to {result_dir}")
        return ResultSet(result_dir)

    @staticmethod
    def load(result_dir):
        """Load a previously saved result set."""
        meta_path = os.path.join(result_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved results found in {result_dir}")
        return ResultSet(result_dir)

    @staticmethod
    def list_runs(output_dir):
        """List available method result directories under output_dir."""
        if not os.path.exists(output_dir):
            return []
        return sorted(
            name for name in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, name))
            and os.path.exists(os.path.join(output_dir, name, "metadata.json"))
        )
