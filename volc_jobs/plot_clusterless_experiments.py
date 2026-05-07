import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


TIMING_KEYS = [
    ("LESS train grads", "timing_sum_train_gradient_time_sec"),
    ("Cluster rep grads", "timing_sum_representative_gradient_time_sec"),
    ("Eval grads", "timing_sum_eval_gradient_time_sec"),
    ("Embeddings", "timing_sum_embedding_time_sec"),
    ("Clustering", "timing_sum_clustering_time_sec"),
    ("Rep sampling", "timing_sum_representative_sampling_time_sec"),
    ("Scoring", "timing_sum_scoring_time_sec"),
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_aggregate(run_dir):
    return load_json(Path(run_dir) / "aggregate.json")


def infer_train_size(run_dir):
    config_files = sorted((Path(run_dir) / "configs").glob("**/*.yaml"))
    for path in config_files:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and "max_samples" in cfg:
            return int(cfg["max_samples"])
    return None


def grouped_bar(rows, key, ylabel, output_path):
    datasets = sorted({r["dataset"] for r in rows})
    selectors = sorted({r["selector"] for r in rows})

    def value(dataset, selector):
        for row in rows:
            if row["dataset"] == dataset and row["selector"] == selector:
                return row.get(key)
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.3), 5))
    x = list(range(len(datasets)))
    width = min(0.8 / max(1, len(selectors)), 0.22)
    for i, selector in enumerate(selectors):
        vals = [value(ds, selector) for ds in datasets]
        offset = (i - (len(selectors) - 1) / 2) * width
        ax.bar([p + offset for p in x], vals, width=width, label=selector)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_timing(run_dir, output_dir):
    rows = load_aggregate(run_dir)
    selectors = sorted({r["selector"] for r in rows})
    means = {}
    for selector in selectors:
        selector_rows = [r for r in rows if r["selector"] == selector]
        means[selector] = []
        for _, key in TIMING_KEYS:
            vals = [r.get(key) for r in selector_rows if isinstance(r.get(key), (int, float))]
            means[selector].append(sum(vals) / len(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0.0] * len(selectors)
    for idx, (label, _) in enumerate(TIMING_KEYS):
        vals = [means[s][idx] for s in selectors]
        ax.bar(selectors, vals, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Selector time seconds, summed over selection steps")
    ax.set_title("Selector Timing Breakdown")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "selector_timing_breakdown.png", dpi=180)
    plt.close(fig)

    grouped_bar(rows, "timing_sum_total_select_time_sec", "Total selector time seconds", Path(output_dir) / "selector_total_time_by_dataset.png")


def plot_rep_strategy(run_dir, output_dir):
    rows = load_aggregate(run_dir)
    grouped_bar(rows, "eval_loss", "Eval loss (lower is better)", Path(output_dir) / "rep_strategy_eval_loss.png")
    grouped_bar(rows, "wall_time_sec", "Wall time seconds (lower is better)", Path(output_dir) / "rep_strategy_wall_time.png")
    grouped_bar(rows, "merged_cache_mib", "Merged cache MiB (lower is better)", Path(output_dir) / "rep_strategy_cache.png")


def plot_scale(run_dirs, output_dir):
    points = []
    for run_dir in run_dirs:
        train_size = infer_train_size(run_dir)
        if train_size is None:
            continue
        for row in load_aggregate(run_dir):
            points.append(
                {
                    "train_size": train_size,
                    "selector": row["selector"],
                    "eval_loss": row.get("eval_loss"),
                    "wall_time_sec": row.get("wall_time_sec"),
                    "merged_cache_mib": row.get("merged_cache_mib"),
                    "selector_time_sec": row.get("timing_sum_total_select_time_sec"),
                }
            )

    selectors = sorted({p["selector"] for p in points})
    for key, ylabel, filename in [
        ("eval_loss", "Eval loss (lower is better)", "scale_eval_loss.png"),
        ("wall_time_sec", "Wall time seconds (lower is better)", "scale_wall_time.png"),
        ("merged_cache_mib", "Merged cache MiB (lower is better)", "scale_cache.png"),
        ("selector_time_sec", "Selector time seconds (lower is better)", "scale_selector_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for selector in selectors:
            rows = sorted([p for p in points if p["selector"] == selector and p.get(key) is not None], key=lambda p: p["train_size"])
            ax.plot([p["train_size"] for p in rows], [p[key] for p in rows], marker="o", label=selector)
        ax.set_xlabel("Train size")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / filename, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-run", type=Path)
    parser.add_argument("--rep-run", type=Path)
    parser.add_argument("--scale-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.timing_run:
        plot_timing(args.timing_run, args.output_dir)
    if args.rep_run:
        plot_rep_strategy(args.rep_run, args.output_dir)
    if args.scale_runs:
        plot_scale(args.scale_runs, args.output_dir)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
