profiling:

```
nsys profile   --trace=cuda,nvtx   --nvtx-capture=range@benchmark   --output=profile_report   --force-overwrite=true   python benchmark.py
```