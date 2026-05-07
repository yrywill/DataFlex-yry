import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path("/home/vepfs/MIXjyx/DataFlex-main")
DATA_ROOT = Path(os.environ.get("DATAFLEX_BENCH_DATA_ROOT", "/home/vepfs/data/dataflex_benchmark"))
RUN_ROOT = Path(os.environ.get("DATAFLEX_BENCH_RUN_ROOT", "/home/vepfs/MIXjyx/dataflex_volc_runs"))
RUN_ID = os.environ.get("RUN_ID") or time.strftime("multidata_bench_%Y%m%d_%H%M%S")
WORK_DIR = RUN_ROOT / RUN_ID
CONFIG_DIR = WORK_DIR / "configs"
RESULTS_DIR = WORK_DIR / "results"
PLOTS_DIR = WORK_DIR / "plots"


DATASET_SPECS = {
    "redpajama_arxiv": {
        "redpajama_subset": "arxiv",
        "text_column": "text",
    },
    "redpajama_c4": {
        "redpajama_subset": "c4",
        "text_column": "text",
    },
    "redpajama_common_crawl": {
        "redpajama_subset": "common_crawl",
        "text_column": "text",
    },
    "redpajama_github": {
        "redpajama_subset": "github",
        "text_column": "text",
    },
    "redpajama_stackexchange": {
        "redpajama_subset": "stackexchange",
        "text_column": "text",
    },
    "redpajama_wikipedia": {
        "redpajama_subset": "wikipedia",
        "text_column": "text",
    },
    "fineweb": {
        "repo": "HuggingFaceFW/fineweb",
        "parquet_prefix": "sample/10BT/",
        "text_column": "text",
    },
    "fineweb_edu": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "parquet_prefix": "sample/10BT/",
        "text_column": "text",
    },
    "slimpajama": {
        "local_jsonl_dir": "/home/vepfs/data/SlimPajama-6B-jsonl",
        "text_column": "text",
    },
}


def run(cmd, cwd=ROOT, env=None):
    print("+", " ".join(cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=True)


def sanitize_name(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text in rows:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def stream_dataset(spec):
    from datasets import load_dataset

    names = [spec.get("name")] + list(spec.get("fallback_names", []))
    last_error = None
    for name in names:
        kwargs = {
            "path": spec["repo"],
            "split": "train",
            "streaming": True,
        }
        if name:
            kwargs["name"] = name
        if spec.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
        try:
            print(f"[data] loading {spec['repo']} name={name}", flush=True)
            return load_dataset(**kwargs)
        except Exception as exc:
            print(f"[data] failed {spec['repo']} name={name}: {exc}", flush=True)
            last_error = exc
    raise last_error


def redpajama_urls(subset):
    import requests

    url = f"https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/raw/main/urls/{subset}.txt"
    print(f"[data] fetching RedPajama url list {url}", flush=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return [line.strip() for line in response.text.splitlines() if line.strip()]


def iter_redpajama_file(url):
    import requests

    print(f"[data] streaming {url}", flush=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    raw = response.raw
    if url.endswith(".zst"):
        import io
        import zstandard as zstd

        reader = zstd.ZstdDecompressor().stream_reader(raw)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        raw.decode_content = True
        text_stream = raw

    for line in text_stream:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = item.get("text")
        if isinstance(text, str):
            yield text


def iter_redpajama_texts(subset):
    for url in redpajama_urls(subset):
        try:
            yield from iter_redpajama_file(url)
        except Exception as exc:
            print(f"[data] skipping failed RedPajama file {url}: {exc}", flush=True)


def collect_parquet_texts(spec, total_needed):
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = spec["repo"]
    prefix = spec["parquet_prefix"]
    text_column = spec.get("text_column", "text")
    min_chars = int(os.environ.get("DATAFLEX_BENCH_MIN_CHARS", "80"))
    max_chars = int(os.environ.get("DATAFLEX_BENCH_MAX_CHARS", "12000"))
    local_dir = DATA_ROOT / "_hf_cache" / sanitize_name(repo)
    files = [
        f for f in list_repo_files(repo_id=repo, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    ]
    if not files:
        raise RuntimeError(f"No parquet files found for {repo}/{prefix}")

    rows = []
    for filename in sorted(files):
        print(f"[data] downloading {repo}/{filename}", flush=True)
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
        df = pd.read_parquet(path, columns=[text_column])
        for text in df[text_column].dropna().astype(str):
            text = text.strip()
            if len(text) < min_chars:
                continue
            if max_chars > 0 and len(text) > max_chars:
                text = text[:max_chars]
            rows.append(text)
            if len(rows) >= total_needed:
                return rows
    if len(rows) < total_needed:
        raise RuntimeError(f"Only collected {len(rows)} rows for {repo}, need {total_needed}")
    return rows


def collect_texts(dataset_key, total_needed):
    spec = DATASET_SPECS[dataset_key]
    min_chars = int(os.environ.get("DATAFLEX_BENCH_MIN_CHARS", "80"))
    max_chars = int(os.environ.get("DATAFLEX_BENCH_MAX_CHARS", "12000"))
    text_column = spec.get("text_column", "text")
    if "local_jsonl_dir" in spec:
        rows = []
        local_dir = Path(spec["local_jsonl_dir"])
        for path in sorted(local_dir.glob("train-*.jsonl")):
            print(f"[data] reading {path}", flush=True)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = item.get(text_column)
                    if not isinstance(text, str):
                        continue
                    text = text.strip()
                    if len(text) < min_chars:
                        continue
                    if max_chars > 0 and len(text) > max_chars:
                        text = text[:max_chars]
                    rows.append(text)
                    if len(rows) >= total_needed:
                        return rows
        if len(rows) < total_needed:
            raise RuntimeError(f"Only collected {len(rows)} rows for {dataset_key}, need {total_needed}")
        return rows
    if "parquet_prefix" in spec:
        return collect_parquet_texts(spec, total_needed)
    if "redpajama_subset" in spec:
        iterator = iter_redpajama_texts(spec["redpajama_subset"])
    else:
        ds = stream_dataset(spec)
        iterator = (item.get(text_column) for item in ds)
    rows = []
    for text in iterator:
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < min_chars:
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        rows.append(text)
        if len(rows) >= total_needed:
            break
    if len(rows) < total_needed:
        raise RuntimeError(f"Only collected {len(rows)} rows for {dataset_key}, need {total_needed}")
    return rows


def prepare_dataset(dataset_key):
    train_size = int(os.environ.get("DATAFLEX_BENCH_TRAIN_SIZE", "2048"))
    eval_size = int(os.environ.get("DATAFLEX_BENCH_EVAL_SIZE", "256"))
    data_dir = DATA_ROOT / dataset_key
    train_path = data_dir / f"train_{train_size}.jsonl"
    eval_path = data_dir / f"eval_{eval_size}.jsonl"
    if train_path.exists() and eval_path.exists():
        return train_path, eval_path

    rows = collect_texts(dataset_key, train_size + eval_size)
    write_jsonl(train_path, rows[:train_size])
    write_jsonl(eval_path, rows[train_size:train_size + eval_size])
    print(f"[data] wrote {dataset_key}: {train_path}, {eval_path}", flush=True)
    return train_path, eval_path


def upsert_dataset_info(dataset_key, train_path, eval_path):
    dataset_info_path = ROOT / "data/dataset_info.json"
    info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    info[f"bench_{dataset_key}_train"] = {
        "file_name": str(train_path),
        "columns": {"prompt": "text"},
    }
    info[f"bench_{dataset_key}_eval"] = {
        "file_name": str(eval_path),
        "columns": {"prompt": "text"},
    }
    dataset_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_components(dataset_key, selector_name, seed):
    components_path = ROOT / "src/dataflex/configs/components.yaml"
    with components_path.open("r", encoding="utf-8") as f:
        components = yaml.safe_load(f)

    selectors = components.setdefault("selectors", {})
    cache_dir = str(WORK_DIR / "selector_cache" / dataset_key / selector_name / f"seed{seed}")
    common = {
        "cache_dir": cache_dir,
        "gradient_type": "adam",
        "proj_dim": int(os.environ.get("DATAFLEX_BENCH_PROJ_DIM", "1024")),
        "seed": seed,
        "save_interval": int(os.environ.get("DATAFLEX_BENCH_SAVE_INTERVAL", "16")),
    }
    if selector_name == "less":
        selectors[selector_name] = {"name": "less", "params": common}
    elif selector_name.startswith("cluster_less"):
        params = dict(common)
        samples_per_cluster = int(os.environ.get("DATAFLEX_BENCH_SAMPLES_PER_CLUSTER", "3"))
        representative_strategy = os.environ.get("DATAFLEX_BENCH_REPRESENTATIVE_STRATEGY", "random")
        if selector_name.startswith("cluster_less_spc"):
            samples_per_cluster = int(selector_name.replace("cluster_less_spc", ""))
        if selector_name.startswith("cluster_less_rep_"):
            representative_strategy = selector_name.replace("cluster_less_rep_", "")
        params.update(
            {
                "cluster_size": int(os.environ.get("DATAFLEX_BENCH_CLUSTER_SIZE", "64")),
                "samples_per_cluster": samples_per_cluster,
                "clustering_batch_size": int(os.environ.get("DATAFLEX_BENCH_CLUSTER_BATCH", "8")),
                "clustering_max_iter": int(os.environ.get("DATAFLEX_BENCH_CLUSTER_ITER", "10")),
                "clustering_method": "kmeans",
                "representative_strategy": representative_strategy,
            }
        )
        selectors[selector_name] = {"name": "cluster_less", "params": params}
    else:
        raise ValueError(f"Unsupported selector {selector_name}")

    with components_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(components, f, allow_unicode=True, sort_keys=False)


def write_train_config(dataset_key, selector_name, seed):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = os.environ.get("DATAFLEX_BENCH_MODEL", "/home/vepfs/models/qwen/Qwen2___5-0___5B-Instruct")
    max_steps = int(os.environ.get("DATAFLEX_BENCH_MAX_STEPS", "80"))
    update_step = int(os.environ.get("DATAFLEX_BENCH_UPDATE_STEP", "20"))
    warmup_step = int(os.environ.get("DATAFLEX_BENCH_WARMUP_STEP", "20"))
    update_times = int(os.environ.get("DATAFLEX_BENCH_UPDATE_TIMES", "3"))
    cutoff_len = int(os.environ.get("DATAFLEX_BENCH_CUTOFF_LEN", "1024"))
    grad_accum = int(os.environ.get("DATAFLEX_BENCH_GRAD_ACCUM", "4"))
    output_dir = RESULTS_DIR / dataset_key / f"{selector_name}_seed{seed}"

    cfg = {
        "model_name_or_path": model_path,
        "trust_remote_code": True,
        "stage": "pt",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": int(os.environ.get("DATAFLEX_BENCH_LORA_RANK", "8")),
        "lora_alpha": int(os.environ.get("DATAFLEX_BENCH_LORA_ALPHA", "16")),
        "dataset": f"bench_{dataset_key}_train",
        "template": os.environ.get("DATAFLEX_BENCH_TEMPLATE", "qwen"),
        "cutoff_len": cutoff_len,
        "max_samples": int(os.environ.get("DATAFLEX_BENCH_TRAIN_SIZE", "2048")),
        "overwrite_cache": True,
        "preprocessing_num_workers": int(os.environ.get("DATAFLEX_BENCH_PREPROCESS_WORKERS", "8")),
        "dataloader_num_workers": int(os.environ.get("DATAFLEX_BENCH_DATALOADER_WORKERS", "0")),
        "seed": seed,
        "output_dir": str(output_dir),
        "logging_steps": 5,
        "save_steps": max_steps,
        "plot_loss": False,
        "save_only_model": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": float(os.environ.get("DATAFLEX_BENCH_LR", "1.0e-4")),
        "max_steps": max_steps,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "ddp_timeout": 180000000,
        "train_type": "dynamic_select",
        "components_cfg_file": "src/dataflex/configs/components.yaml",
        "component_name": selector_name,
        "warmup_step": warmup_step,
        "update_step": update_step,
        "update_times": update_times,
        "eval_dataset": f"bench_{dataset_key}_eval",
        "per_device_eval_batch_size": 1,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": False,
        "eval_strategy": "steps",
        "eval_steps": update_step,
    }
    ds_config = os.environ.get("DATAFLEX_BENCH_DEEPSPEED", "")
    if ds_config:
        cfg["deepspeed"] = ds_config

    path = CONFIG_DIR / dataset_key / f"{selector_name}_seed{seed}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def dir_size_bytes(path):
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def merged_cache_bytes(path):
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("all_projected_grads.pt"))


def read_state(output_dir):
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    eval_losses = [item.get("eval_loss") for item in state.get("log_history", []) if "eval_loss" in item]
    return {
        "global_step": state.get("global_step"),
        "best_metric": state.get("best_metric"),
        "last_eval_loss": eval_losses[-1] if eval_losses else None,
    }


def read_selection_timing(cache_dir):
    timing_rows = []
    for path in sorted(cache_dir.glob("step_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        timing = payload.get("metric", {}).get("timing", {})
        if isinstance(timing, dict):
            timing_rows.append(timing)

    out = {
        "selection_steps": len(timing_rows),
    }
    keys = sorted({key for row in timing_rows for key in row if isinstance(row.get(key), (int, float, bool))})
    for key in keys:
        values = [float(row[key]) for row in timing_rows if isinstance(row.get(key), (int, float, bool))]
        if values:
            out[f"timing_sum_{key}"] = sum(values)
            out[f"timing_mean_{key}"] = sum(values) / len(values)
    return out


def summarize(experiment_rows):
    rows = []
    for row in experiment_rows:
        dataset_key = row["dataset"]
        selector = row["selector"]
        seed = row["seed"]
        out_dir = RESULTS_DIR / dataset_key / f"{selector}_seed{seed}"
        cache_dir = WORK_DIR / "selector_cache" / dataset_key / selector / f"seed{seed}"
        state = read_state(out_dir)
        last_eval = state.get("last_eval_loss")
        timing = read_selection_timing(cache_dir)
        result = {
            **row,
            **state,
            **timing,
            "perplexity": math.exp(last_eval) if isinstance(last_eval, (int, float)) else None,
            "output_dir": str(out_dir),
            "cache_dir": str(cache_dir),
            "cache_bytes": dir_size_bytes(cache_dir),
            "merged_cache_bytes": merged_cache_bytes(cache_dir),
        }
        rows.append(result)

    summary_path = WORK_DIR / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = WORK_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row}) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("SUMMARY_PATH", summary_path, flush=True)
    print("SUMMARY_CSV", csv_path, flush=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    create_plots(rows)
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["selector"]), []).append(row)
    out = []
    for (dataset, selector), vals in sorted(groups.items()):
        def mean(key):
            xs = [v[key] for v in vals if isinstance(v.get(key), (int, float))]
            return sum(xs) / len(xs) if xs else None

        out.append(
            {
                "dataset": dataset,
                "selector": selector,
                "eval_loss": mean("last_eval_loss"),
                "perplexity": mean("perplexity"),
                "wall_time_sec": mean("wall_time_sec"),
                "cache_mib": mean("cache_bytes") / 1024 / 1024 if mean("cache_bytes") is not None else None,
                "merged_cache_mib": mean("merged_cache_bytes") / 1024 / 1024 if mean("merged_cache_bytes") is not None else None,
                "selection_steps": mean("selection_steps"),
                "timing_sum_total_select_time_sec": mean("timing_sum_total_select_time_sec"),
                "timing_sum_train_gradient_time_sec": mean("timing_sum_train_gradient_time_sec"),
                "timing_sum_representative_gradient_time_sec": mean("timing_sum_representative_gradient_time_sec"),
                "timing_sum_eval_gradient_time_sec": mean("timing_sum_eval_gradient_time_sec"),
                "timing_sum_embedding_time_sec": mean("timing_sum_embedding_time_sec"),
                "timing_sum_clustering_time_sec": mean("timing_sum_clustering_time_sec"),
                "timing_sum_representative_sampling_time_sec": mean("timing_sum_representative_sampling_time_sec"),
                "timing_sum_representative_feature_load_time_sec": mean("timing_sum_representative_feature_load_time_sec"),
                "timing_sum_scoring_time_sec": mean("timing_sum_scoring_time_sec"),
            }
        )
    return out


def create_plots(rows):
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = aggregate(rows)
    datasets = sorted({r["dataset"] for r in agg})
    selectors = sorted({r["selector"] for r in rows})

    def value(dataset, selector, key):
        for r in agg:
            if r["dataset"] == dataset and r["selector"] == selector:
                return r.get(key)
        return None

    def grouped_bar(key, ylabel, filename):
        x = list(range(len(datasets)))
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.2), 5))
        if len(selectors) == 1:
            width = 0.5
        else:
            width = min(0.8 / len(selectors), 0.22)
        for i, selector in enumerate(selectors):
            vals = [value(ds, selector, key) for ds in datasets]
            offset = (i - (len(selectors) - 1) / 2) * width
            ax.bar([p + offset for p in x], vals, width=width, label=selector)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=180)
        plt.close(fig)

    grouped_bar("eval_loss", "Eval loss (lower is better)", "eval_loss_by_dataset.png")
    grouped_bar("perplexity", "Perplexity (lower is better)", "perplexity_by_dataset.png")
    grouped_bar("wall_time_sec", "Wall time seconds (lower is better)", "wall_time_by_dataset.png")
    grouped_bar("merged_cache_mib", "Merged gradient cache MiB (lower is better)", "cache_by_dataset.png")

    if "less" in selectors and "cluster_less" in selectors:
        ratio_rows = []
        for ds in datasets:
            less_time = value(ds, "less", "wall_time_sec")
            cluster_time = value(ds, "cluster_less", "wall_time_sec")
            less_cache = value(ds, "less", "merged_cache_mib")
            cluster_cache = value(ds, "cluster_less", "merged_cache_mib")
            ratio_rows.append(
                {
                    "dataset": ds,
                    "speedup": less_time / cluster_time if less_time and cluster_time else None,
                    "cache_reduction": less_cache / cluster_cache if less_cache and cluster_cache else None,
                }
            )

        fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.2), 5))
        x = list(range(len(datasets)))
        width = 0.36
        ax.bar([p - width / 2 for p in x], [r["speedup"] for r in ratio_rows], width=width, label="LESS / ClusterLess time")
        ax.bar([p + width / 2 for p in x], [r["cache_reduction"] for r in ratio_rows], width=width, label="LESS / ClusterLess cache")
        ax.axhline(1.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylabel("Ratio (higher favors cluster_less)")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "speedup_and_cache_ratio.png", dpi=180)
        plt.close(fig)

    sweep_selectors = [s for s in selectors if s.startswith("cluster_less_spc")]
    if sweep_selectors:
        sweep_selectors = sorted(sweep_selectors, key=lambda s: int(s.replace("cluster_less_spc", "")))
        for key, ylabel, filename in [
            ("eval_loss", "Eval loss (lower is better)", "samples_per_cluster_eval_loss.png"),
            ("wall_time_sec", "Wall time seconds (lower is better)", "samples_per_cluster_wall_time.png"),
            ("merged_cache_mib", "Merged gradient cache MiB (lower is better)", "samples_per_cluster_cache.png"),
        ]:
            fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.2), 5))
            for ds in datasets:
                xs = [int(s.replace("cluster_less_spc", "")) for s in sweep_selectors]
                ys = [value(ds, s, key) for s in sweep_selectors]
                ax.plot(xs, ys, marker="o", label=ds)
            ax.set_xlabel("samples_per_cluster")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / filename, dpi=180)
            plt.close(fig)

    aggregate_path = WORK_DIR / "aggregate.json"
    aggregate_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PLOTS_DIR", PLOTS_DIR, flush=True)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_keys = [x.strip() for x in os.environ.get("DATAFLEX_BENCH_DATASETS", ",".join(DATASET_SPECS)).split(",") if x.strip()]
    selectors = [x.strip() for x in os.environ.get("DATAFLEX_BENCH_SELECTORS", "less,cluster_less").split(",") if x.strip()]
    spc_values = [x.strip() for x in os.environ.get("DATAFLEX_BENCH_SAMPLES_PER_CLUSTER_VALUES", "").split(",") if x.strip()]
    rep_strategy_values = [x.strip() for x in os.environ.get("DATAFLEX_BENCH_REPRESENTATIVE_STRATEGY_VALUES", "").split(",") if x.strip()]
    if spc_values:
        expanded = []
        for selector in selectors:
            if selector == "cluster_less":
                expanded.extend([f"cluster_less_spc{int(v)}" for v in spc_values])
            else:
                expanded.append(selector)
        selectors = expanded
    if rep_strategy_values:
        expanded = []
        for selector in selectors:
            if selector == "cluster_less":
                expanded.extend([f"cluster_less_rep_{v}" for v in rep_strategy_values])
            else:
                expanded.append(selector)
        selectors = expanded
    seeds = [int(x) for x in os.environ.get("DATAFLEX_BENCH_SEEDS", "42,43,44").split(",") if x.strip()]

    print("RUN_ID", RUN_ID, flush=True)
    print("WORK_DIR", WORK_DIR, flush=True)
    print("DATASETS", dataset_keys, flush=True)
    print("SELECTORS", selectors, flush=True)
    print("SEEDS", seeds, flush=True)

    prepared = {}
    for dataset_key in dataset_keys:
        if dataset_key not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        train_path, eval_path = prepare_dataset(dataset_key)
        upsert_dataset_info(dataset_key, train_path, eval_path)
        prepared[dataset_key] = (train_path, eval_path)

    experiment_rows = []
    for dataset_key in dataset_keys:
        for selector in selectors:
            for seed in seeds:
                upsert_components(dataset_key, selector, seed)
                cfg_path = write_train_config(dataset_key, selector, seed)
                started = time.time()
                status = "success"
                try:
                    run(["dataflex-cli", "train", str(cfg_path)], env={"DISABLE_VERSION_CHECK": "1"})
                except Exception:
                    status = "failed"
                    raise
                finally:
                    elapsed = time.time() - started
                    row = {
                        "dataset": dataset_key,
                        "selector": selector,
                        "seed": seed,
                        "status": status,
                        "wall_time_sec": elapsed,
                    }
                    experiment_rows.append(row)
                    print(
                        f"EXPERIMENT_DONE dataset={dataset_key} selector={selector} seed={seed} "
                        f"status={status} elapsed_sec={elapsed:.1f}",
                        flush=True,
                    )
                    partial_path = WORK_DIR / "partial_experiments.json"
                    partial_path.write_text(json.dumps(experiment_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summarize(experiment_rows)


if __name__ == "__main__":
    main()
