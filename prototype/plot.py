"""
Paper-ready sweep result plots.

Usage:
    python plot.py results/SIFT/SWEEP/sweep_summary.csv
    python plot.py results/SIFT/SWEEP/sweep_summary.csv --out figures/
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Visual identity ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth":     0.6,
    "grid.linewidth":     0.4,
    "lines.linewidth":    1.3,
    "lines.markersize":   5,
    "axes.grid":          True,
    "grid.alpha":         0.3,
})

# Colour palette — distinct, colourblind-friendly
METHOD_STYLE = {
    "brute_force": {
        "color": "#F4A261",   # orange
        "marker": "D",
        "label": "Brute Force",
    },
    "cuvs_cagra": {
        "color": "#E63946",   # red
        "marker": "o",
        "label": "cuVS CAGRA",
    },
    "cuvs_ivf_flat": {
        "color": "#457B9D",   # steel blue
        "marker": "s",
        "label": "cuVS IVF-Flat",
    },
    "centroid_join": {
        "color": "#2A9D8F",   # teal
        "marker": "^",
        "label": "Centroid Join",
    },
    "centroid_batch": {
        "color": "#94D2BD",   # light teal
        "marker": "v",
        "label": "Centroid Batch",
    },
    "centroid_centroid": {
        "color": "#0A9396",   # dark teal
        "marker": "p",
        "label": "Centroid-Centroid",
    },
}

METHOD_ORDER = ["brute_force", "cuvs_cagra", "cuvs_ivf_flat", "centroid_join"]


def _style(base_method):
    s = METHOD_STYLE.get(base_method, {
        "color": "#6C757D", "marker": "x", "label": base_method,
    })
    return s["color"], s["marker"], s["label"]


def _log_time_axis(ax):
    """Configure a log-scale y-axis with clean time labels."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:g}s"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _save(fig, path):
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  -> {path}")


def load_sweep_data(csv_path):
    """Load sweep CSV; keep brute_force row for the scatter plot."""
    df = pd.read_csv(csv_path)
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.dropna(subset=["time_s"])
    return df


def load_10m_data(results_dir):
    """Load 10M baseline datasets from metadata.json and statistics.json files."""
    pattern = os.path.join(results_dir, "*", "10M", "*", "metadata.json")
    data = []
    
    stats_cache = {}
    
    for f in glob.glob(pattern):
        try:
            with open(f) as fp:
                d = json.load(fp)
            parts = f.split(os.sep)
            dataset = parts[-4]  # .../results/DATASET/10M/METHOD/...
            method = d.get("method", "unknown")
            time_s = float(d.get("time_s", 0))
            
            cluster_time_s = d.get("cluster_time_s")
            if cluster_time_s is None or cluster_time_s == "N/A":
                cluster_time_s = 0.0
            else:
                cluster_time_s = float(cluster_time_s)
                
            # Load stats for dataset if not loaded
            if dataset not in stats_cache:
                stats_path = os.path.join(results_dir, dataset, "10M", "statistics.json")
                if os.path.exists(stats_path):
                    with open(stats_path) as sfp:
                        stats_cache[dataset] = json.load(sfp)
                else:
                    stats_cache[dataset] = []
                    
            # Find recall
            recall = 1.0 if "brute" in method else np.nan
            for s in stats_cache[dataset]:
                if s.get("method") == method and "recall" in s:
                    recall = float(s["recall"])
                    break
                    
            data.append({
                "dataset": dataset,
                "method": method,
                "time_s": time_s,
                "cluster_time_s": cluster_time_s,
                "join_time_s": time_s - cluster_time_s,
                "recall": recall
            })
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════
#  Figure 1 – Scatter: log(Time) vs Recall (all runs + brute force)
# ═══════════════════════════════════════════════════════════════
def plot_scatter(df, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for base in METHOD_ORDER:
        grp = df[df["base_method"] == base]
        if grp.empty:
            continue
        c, m, lbl = _style(base)

        if base == "brute_force":
            # Single point — draw as a horizontal reference line + marker
            t = grp["time_s"].iloc[0]
            ax.axhline(t, color=c, linewidth=0.8, linestyle="--",
                       alpha=0.6, zorder=2)
            ax.scatter([1.0], [t], color=c, marker=m, s=45,
                       edgecolors="white", linewidths=0.4,
                       label=lbl, zorder=4)
        else:
            sub = grp.dropna(subset=["recall"])
            ax.scatter(
                sub["recall"], sub["time_s"],
                color=c, marker=m, label=lbl,
                s=32, alpha=0.80, edgecolors="white", linewidths=0.3,
                zorder=3,
            )

    _log_time_axis(ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Time (s)  [log scale]")
    ax.set_title("Recall vs. Execution Time — All Configurations")
    ax.set_xlim(left=-0.02, right=1.04)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc",
              loc="upper left")

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "scatter_time_recall.pdf"))


# ═══════════════════════════════════════════════════════════════
#  Figure 2 – Clean Pareto frontier only (no background clutter)
# ═══════════════════════════════════════════════════════════════
def _pareto_front(grp):
    """Return the Pareto-optimal rows (lowest time at each distinct recall)."""
    grp = grp.sort_values("recall").reset_index(drop=True)
    best, min_time = [], float("inf")
    for _, row in grp.iloc[::-1].iterrows():
        if row["time_s"] < min_time:
            min_time = row["time_s"]
            best.append(row)
    return pd.DataFrame(best[::-1]).reset_index(drop=True)


def plot_pareto(df, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Draw brute force reference line first (no recall value, so handle separately)
    bf = df[df["base_method"] == "brute_force"]
    if not bf.empty:
        c, m, lbl = _style("brute_force")
        t = bf["time_s"].iloc[0]
        ax.axhline(t, color=c, linewidth=0.8, linestyle="--",
                   alpha=0.6, zorder=2)
        ax.scatter([1.0], [t], color=c, marker=m, s=45,
                   edgecolors="white", linewidths=0.4,
                   label=lbl, zorder=4)

    for base in METHOD_ORDER:
        if base == "brute_force":
            continue
        grp = df[(df["base_method"] == base)].dropna(subset=["recall"])
        if grp.empty:
            continue
        c, m, lbl = _style(base)
        front = _pareto_front(grp)
        ax.plot(
            front["recall"], front["time_s"],
            color=c, marker=m, label=lbl,
            markersize=5, markeredgecolor="white", markeredgewidth=0.4,
            zorder=3,
        )

    _log_time_axis(ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Time (s)  [log scale]")
    ax.set_title("Recall vs. Execution Time — Best Configurations")
    ax.set_xlim(left=-0.02, right=1.04)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc",
              loc="upper left")

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "pareto_time_recall.pdf"))


# ═══════════════════════════════════════════════════════════════
#  Helpers for centroid scaling plots
# ═══════════════════════════════════════════════════════════════
def _prep_centroid(df):
    cj = df[df["base_method"] == "centroid_join"].copy()
    cj["n_clusters"] = pd.to_numeric(cj["n_clusters"], errors="coerce")
    cj["k_db_candidates"] = pd.to_numeric(cj["k_db_candidates"],
                                           errors="coerce")
    return cj.dropna(subset=["n_clusters", "k_db_candidates"])


def _seq_colors(values, cmap_name="YlGnBu"):
    cmap = matplotlib.colormaps[cmap_name]
    n = max(1, len(values) - 1)
    return {v: cmap(0.3 + 0.6 * i / n) for i, v in enumerate(sorted(values))}


def _two_panel_log(fig, ax_time, ax_recall, x_vals, groups, group_colors,
                   xlabel, group_label, x_log_base=2):
    """Fill a standard (time | recall) two-panel layout."""
    for val in sorted(groups.keys()):
        sub = groups[val]
        c = group_colors[val]
        lbl = f"${group_label} = {int(val)}$"
        kw = dict(color=c, marker="o", markersize=4.5,
                  markeredgecolor="white", markeredgewidth=0.3, label=lbl)
        ax_time.plot(sub[x_vals], sub["time_s"], **kw)
        ax_recall.plot(sub[x_vals], sub["recall"], **kw)

    for ax in (ax_time, ax_recall):
        ax.set_xscale("log", base=x_log_base)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel(xlabel)

    _log_time_axis(ax_time)
    ax_time.set_ylabel("Time (s)  [log scale]")
    ax_time.set_title("(a)  Execution Time")
    ax_time.legend(frameon=True, fancybox=False, edgecolor="#cccccc",
                   title=f"${{{group_label.replace('_', r'\_')}}}$",
                   title_fontsize=8.5)

    ax_recall.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_recall.set_ylabel("Recall")
    ax_recall.set_title("(b)  Recall")
    ax_recall.legend(frameon=True, fancybox=False, edgecolor="#cccccc",
                     title=f"${{{group_label.replace('_', r'\_')}}}$",
                     title_fontsize=8.5)


# ═══════════════════════════════════════════════════════════════
#  Figure 3a – Centroid: k_db sweep, one line per n_clusters
# ═══════════════════════════════════════════════════════════════
def plot_centroid_kdb_scaling(df, out_dir):
    cj = _prep_centroid(df)
    if cj.empty:
        print("  [SKIP] No centroid_join data")
        return

    nc_values = sorted(cj["n_clusters"].unique())
    colors = _seq_colors(nc_values)
    groups = {nc: cj[cj["n_clusters"] == nc].sort_values("k_db_candidates")
              for nc in nc_values}

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    _two_panel_log(fig, ax_t, ax_r,
                   x_vals="k_db_candidates", groups=groups,
                   group_colors=colors,
                   xlabel=r"$k_{db}$ (database candidates)",
                   group_label="n_{clusters}")
    fig.tight_layout(w_pad=2.5)
    _save(fig, os.path.join(out_dir, "centroid_scaling_kdb.pdf"))


# ═══════════════════════════════════════════════════════════════
#  Figure 3b – Centroid: n_clusters sweep, one line per k_db
# ═══════════════════════════════════════════════════════════════
def plot_centroid_nc_scaling(df, out_dir):
    cj = _prep_centroid(df)
    if cj.empty:
        return

    kdb_values = sorted(cj["k_db_candidates"].unique())
    colors = _seq_colors(kdb_values, cmap_name="PuBuGn")
    groups = {kdb: cj[cj["k_db_candidates"] == kdb].sort_values("n_clusters")
              for kdb in kdb_values}

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    _two_panel_log(fig, ax_t, ax_r,
                   x_vals="n_clusters", groups=groups,
                   group_colors=colors,
                   xlabel=r"$n_{clusters}$",
                   group_label="k_{db}")
    fig.tight_layout(w_pad=2.5)
    _save(fig, os.path.join(out_dir, "centroid_scaling_nc.pdf"))


# ═══════════════════════════════════════════════════════════════
#  Figure 4 – Batching Overhead Comparison (10M Data)
# ═══════════════════════════════════════════════════════════════

DATASET_DIMS = {
    "DEEP": 96,
    "SIFT": 128,
    "SimSearch": 256
}

def _format_dataset_label(ds):
    dim = DATASET_DIMS.get(ds)
    return f"{ds} ({dim}d)" if dim else ds

def plot_batching_overhead(df, out_dir):
    if df.empty: return
    
    datasets = sorted(df["dataset"].unique())
    n_ds = len(datasets)
    methods = ["brute_force", "centroid_batch", "centroid_join", "centroid_centroid"]
    n_methods = len(methods)
    
    fig, (ax_time, ax_recall) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(n_ds)
    width = 0.8 / n_methods
    
    c_over = "#E9C46A"   # yellow for overhead
    
    from matplotlib.patches import Patch
    handles_time, labels_time = [], []
    
    for i, m in enumerate(methods):
        times, overs, recalls = [], [], []
        c, _, lbl = _style(m)
        
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["method"] == m)]
            if not sub.empty:
                t = sub["time_s"].iloc[0]
                o = sub["cluster_time_s"].iloc[0]
                r = sub["recall"].iloc[0]
            else:
                t, o, r = 0, 0, np.nan
                
            times.append(t - o)
            overs.append(o)
            recalls.append(r)
            
        offset = (i - n_methods/2 + 0.5) * width
        
        # Panel A: Time (stacked)
        bars = ax_time.bar(x + offset, times, width, label=lbl, 
                           color=c, edgecolor="white", linewidth=0.5)
        handles_time.append(bars[0])
        labels_time.append(lbl)
        
        if any(o > 0 for o in overs):
            ax_time.bar(x + offset, overs, width, bottom=times,
                        color=c_over, edgecolor="white", linewidth=0.5)
                        
        # Panel B: Recall
        ax_recall.bar(x + offset, recalls, width, label=lbl, 
                      color=c, edgecolor="white", linewidth=0.5)
    
    # Add cluster overhead to time legend
    handles_time.append(Patch(facecolor=c_over, edgecolor='white', linewidth=0.5))
    labels_time.append("Clustering Overhead")
    ax_time.legend(handles_time, labels_time, frameon=True, edgecolor="#cccccc", fontsize=8)
    
    _log_time_axis(ax_time)
    ax_time.set_ylabel("Execution Time (s) [log scale]")
    ax_time.set_title("(a) Execution Time with Overhead")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([_format_dataset_label(ds) for ds in datasets])
    
    ax_recall.set_ylabel("Recall")
    ax_recall.set_title("(b) Search Recall")
    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels([_format_dataset_label(ds) for ds in datasets])
    ax_recall.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_recall.legend(frameon=True, edgecolor="#cccccc", fontsize=8)
    
    fig.tight_layout(w_pad=2.5)
    _save(fig, os.path.join(out_dir, "batching_overhead_10m.pdf"))


# ═══════════════════════════════════════════════════════════════
#  Figure 5 – All 10M Data Comparison
# ═══════════════════════════════════════════════════════════════
def plot_all_10m(df, out_dir):
    if df.empty: return
    
    datasets = sorted(df["dataset"].unique())
    methods = ["brute_force", "centroid_batch", "cuvs_ivf_flat", "centroid_join", "centroid_centroid"]
    n_methods = len(methods)
    
    fig, (ax_time, ax_recall) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(datasets))
    width = 0.8 / n_methods
    
    for i, m in enumerate(methods):
        times = []
        recalls = []
        c, _, lbl = _style(m)
        
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["method"] == m)]
            if not sub.empty:
                times.append(sub["time_s"].iloc[0])
                recalls.append(sub["recall"].iloc[0])
            else:
                times.append(0)
                recalls.append(np.nan)
        
        offset = (i - n_methods/2 + 0.5) * width
        ax_time.bar(x + offset, times, width, label=lbl, 
                    color=c, edgecolor="white", linewidth=0.5)
        ax_recall.bar(x + offset, recalls, width, label=lbl, 
                      color=c, edgecolor="white", linewidth=0.5)
        
    _log_time_axis(ax_time)
    ax_time.set_ylabel("Execution Time (s) [log scale]")
    ax_time.set_title("(a) End-to-End Search Times")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([_format_dataset_label(ds) for ds in datasets])
    ax_time.legend(frameon=True, edgecolor="#cccccc", fontsize=8, loc="lower right")
    
    ax_recall.set_ylabel("Recall")
    ax_recall.set_title("(b) Search Recall")
    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels([_format_dataset_label(ds) for ds in datasets])
    ax_recall.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_recall.legend(frameon=True, edgecolor="#cccccc", fontsize=8, loc="lower left")
    
    fig.tight_layout(w_pad=2.5)
    _save(fig, os.path.join(out_dir, "all_10m_comparison.pdf"))


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Plot sweep results from sweep_summary.csv")
    parser.add_argument("csv", help="Path to sweep_summary.csv")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same dir as CSV)")
    parser.add_argument("--results-dir", default=None,
                        help="Path to the overall 'results' directory for 10M data")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    
    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.csv)))), "")
    if "results" not in results_dir:
        # Fallback to prototype/results
        results_dir = os.path.join(os.getcwd(), "results")

    df_sweep = load_sweep_data(args.csv)
    print(f"Loaded {len(df_sweep)} sweep rows from {args.csv}")
    print(f"Methods: {df_sweep['base_method'].unique().tolist()}")

    plot_scatter(df_sweep, out_dir)
    plot_pareto(df_sweep, out_dir)
    plot_centroid_kdb_scaling(df_sweep, out_dir)
    plot_centroid_nc_scaling(df_sweep, out_dir)
    
    df_10m = load_10m_data(results_dir)
    if not df_10m.empty:
        print(f"Loaded {len(df_10m)} 10M baseline runs from {results_dir}")
        plot_batching_overhead(df_10m, out_dir)
        plot_all_10m(df_10m, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
