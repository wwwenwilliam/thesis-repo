import json, glob, os
import pandas as pd
data = []
for f in glob.glob("results/*/10M/*/metadata.json"):
    with open(f) as fp:
        d = json.load(fp)
    parts = f.split(os.sep)
    idx = parts.index("10M")
    dataset = parts[idx - 1]
    method = d.get("method", "unknown")
    time_s = d.get("time_s", 0)
    cluster_time_s = d.get("cluster_time_s")
    if cluster_time_s is None or cluster_time_s == "N/A":
        cluster_time_s = 0.0
    
    data.append({
        "dataset": dataset,
        "method": method,
        "time_s": float(time_s),
        "cluster_time_s": float(cluster_time_s),
        "join_time_s": float(time_s) - float(cluster_time_s)
    })
df = pd.DataFrame(data)
print(df.sort_values(["dataset", "method"]))
