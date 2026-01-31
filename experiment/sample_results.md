tests on Ubuntu 24.04, Intel i9-10900K, 32GB@3600MHz, RTX 2080 ti (Turing)

```
Loading SIFT data...
Loaded 1000000 vectors of dim 128

============================================================
Benchmark: N=1000000, M=1000000, D=128, threshold=30000.0
============================================================

[1/3] IVF-Flat
IVF build (10 blocks): 0.5887s
IVF search+filter: 99.2989s
IVF finished in 103.3089s found: 25299931 pairs

[2/3] Centroid Clustering
Centroid join: 53.7357s
Centroid finished in 58.0979s found: 32706086 pairs

[3/3] Brute Force
Brute force: 182.3995s
Brute force finished in 186.7684s found: 32876558 pairs

============================================================
Results Summary
============================================================

ivf:
  Time: 103.3089s
  Pairs: 25299931
  Dists: min=0.00000 max=30000.00000 avg=21863.74381
  Recall vs brute_force: 0.7695

centroid:
  Time: 58.0979s
  Pairs: 32706086
  Dists: min=0.00000 max=30000.00000 avg=22632.99313
  Recall vs brute_force: 0.9948

brute_force:
  Time: 186.7684s
  Pairs: 32876558
  Dists: min=0.00000 max=30000.00000 avg=22633.05356

ivf top 5 pairs:
  A[271240] <-> B[192424] dist=0.00000
  A[271241] <-> B[192425] dist=0.00000
  A[271242] <-> B[192426] dist=0.00000
  A[271243] <-> B[192427] dist=0.00000
  A[271244] <-> B[192428] dist=0.00000

centroid top 5 pairs:
  A[271657] <-> B[192841] dist=0.00000
  A[271752] <-> B[192936] dist=0.00000
  A[271763] <-> B[192947] dist=0.00000
  A[271831] <-> B[193015] dist=0.00000
  A[271934] <-> B[193118] dist=0.00000

brute_force top 5 pairs:
  A[264065] <-> B[199787] dist=0.00000
  A[264066] <-> B[199788] dist=0.00000
  A[264067] <-> B[199789] dist=0.00000
  A[264068] <-> B[199790] dist=0.00000
  A[264069] <-> B[199791] dist=0.00000
```