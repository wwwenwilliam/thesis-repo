"""
Unified parameter object for the benchmark prototype.

Define one Params() instance and pass it through to all algorithms
and batching strategies.  Each component reads only the sub-object
it needs, e.g.  params.centroid_join.n_clusters.

The full object is serialisable to dict/JSON for result metadata.
"""
import dataclasses
from dataclasses import dataclass, field


# ── Join algorithm parameters ────────────────────────────────────

@dataclass
class BruteForceParams:
    """Parameters for brute_force_join (none currently)."""
    pass


@dataclass
class CuvsCagraParams:
    """Parameters for cuVS CAGRA join."""
    k: int = 64


@dataclass
class CuvsIvfFlatParams:
    """Parameters for cuVS IVF-Flat join."""
    k: int = 64


@dataclass
class CuvsBruteForceParams:
    """Parameters for cuVS brute-force join."""
    k: int = 32


@dataclass
class CentroidJoinParams:
    """Parameters for centroid_join."""
    sample_fraction: float = 0.20
    n_clusters: int = 1024
    k_db_candidates: int = 32768
    use_ivf: bool = False


# ── Batching strategy parameters ─────────────────────────────────

@dataclass
class SimpleBatchParams:
    """Parameters for simple_batch (none currently)."""
    pass


@dataclass
class CentroidBatchParams:
    """Parameters for centroid_batch."""
    sample_fraction: float = 0.25
    centroid_threshold: float = 0.1




# ── Top-level unified object ─────────────────────────────────────

@dataclass
class Params:
    """Central configuration object threaded through the entire pipeline."""

    # ── global ──
    threshold: float = 100_000.0

    # ── per-method batch sizes ──
    batch_sizes: dict = field(default_factory=lambda: {
        "brute_force":       25_000,
        "cuvs_ivf_flat":     1_000_000,
        "cuvs_cagra":        20_000,
        "cuvs_brute_force":  100_000,
        "centroid_join":     1_000_000,
        "centroid_batch":    25_000,
        "centroid_centroid": 1_000_000,

    })

    # ── per-algorithm ──
    brute_force: BruteForceParams = field(default_factory=BruteForceParams)
    cuvs_cagra: CuvsCagraParams = field(default_factory=CuvsCagraParams)

    cuvs_ivf_flat: CuvsIvfFlatParams = field(default_factory=CuvsIvfFlatParams)
    cuvs_brute_force: CuvsBruteForceParams = field(default_factory=CuvsBruteForceParams)
    centroid_join: CentroidJoinParams = field(default_factory=CentroidJoinParams)

    # ── per-batching ──
    simple_batch: SimpleBatchParams = field(default_factory=SimpleBatchParams)
    centroid_batch: CentroidBatchParams = field(default_factory=CentroidBatchParams)


    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON metadata)."""
        return dataclasses.asdict(self)

