tests on Ubuntu 24.04, Intel i9-10900K, 32GB@3600MHz, RTX 2080 ti (Turing)

```
Loading SIFT data...
u8bin header: 100,000,000 vectors x 128D
Loaded 1,000,000 x 128 from /home/william/thesis_ws/datasets/sift/learn.100M.u8bin
Loaded 1000000 vectors of dim 128

============================================================
Benchmark: N=1000000, M=1000000, D=128, threshold=20000.0
============================================================

Centroid Clustering
  params: n_clusters=1024, k_db_candidates=8192, batch_size=1,000,000
Centroid join: 18.1581s
  kmeans+assign: 14.21s | centroid search: 0.32s | brute force: 3.41s
  1 block pairs, 1024 cluster iterations
  pairs compared: 8,192,000,000 (8.19B)
  peak GPU memory: 4257 MB
Centroid finished in 18.3021s found: 49271139 pairs

IVF-Flat
  params: n_lists=512, n_probes=32, k_candidates=512, batch_size=250,000
IVF build (4 blocks): 0.5961s
IVF search+filter: 51.7609s
  search: 51.60s | filter: 0.13s | 10 searches
  pairs compared: 1,280,000,000 (1.28B)
  peak GPU memory: 4546 MB
IVF finished in 52.4190s found: 13412661 pairs

Brute Force
  params: batch_size=20,000
Brute force: 39.5510s
  1275 tiles, pairs compared: 510,000,000,000 (510.00B)
  peak GPU memory: 10792 MB
Brute force finished in 39.6973s found: 49342622 pairs

============================================================
Results Summary
============================================================

centroid:
  Time: 18.3021s
  Pairs: 49271139
  Dists: min=13.00000 max=20000.00000 avg=9778.65527
  Recall vs brute_force: 0.9986

ivf:
  Time: 52.4190s
  Pairs: 13412661
  Dists: min=13.00000 max=20000.00000 avg=11850.43848
  Recall vs brute_force: 0.2718

brute_force:
  Time: 39.6973s
  Pairs: 49342622
  Dists: min=13.00000 max=20000.00000 avg=9789.68555

centroid top 5 pairs:
  A[95512] <-> B[95510] dist=13.00000
  A[734103] <-> B[734102] dist=13.00000
  A[95511] <-> B[95509] dist=15.00000
  A[718012] <-> B[718009] dist=24.00000
  A[982504] <-> B[982503] dist=40.00000

ivf top 5 pairs:
  A[734103] <-> B[734102] dist=13.00000
  A[95512] <-> B[95510] dist=13.00000
  A[95511] <-> B[95509] dist=15.00000
  A[718012] <-> B[718009] dist=24.00000
  A[982504] <-> B[982503] dist=40.00000

brute_force top 5 pairs:
  A[734103] <-> B[734102] dist=13.00000
  A[95512] <-> B[95510] dist=13.00000
  A[95511] <-> B[95509] dist=15.00000
  A[718012] <-> B[718009] dist=24.00000
  A[982504] <-> B[982503] dist=40.00000

```