import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path("/home/vepfs/MIXjyx/DataFlex-main")
DATA_ROOT = Path("/home/vepfs/data")
RUN_ROOT = Path(os.environ.get("DATAFLEX_BATCH_RUN_ROOT", "/home/vepfs/MIXjyx/dataflex_volc_runs"))
RUN_ID = os.environ.get("RUN_ID") or time.strftime("selector_batch_%Y%m%d_%H%M%S")
WORK_DIR = RUN_ROOT / RUN_ID
CONFIG_DIR = WORK_DIR / "configs"
RESULTS_DIR = WORK_DIR / "results"


def run(cmd, cwd=ROOT, env=None):
    print("+", " ".join(cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=True)


def prepare_data():
    dataset_name = os.environ.get("DATAFLEX_BATCH_DATASET", "alpaca").strip().lower()
    if dataset_name in {"slimpajama", "slim_pajama"}:
        return prepare_slimpajama_data()
    if dataset_name != "alpaca":
        raise ValueError(f"Unsupported DATAFLEX_BATCH_DATASET={dataset_name!r}")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    parquet_path = DATA_ROOT / "tatsu-lab_alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
    if not parquet_path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="tatsu-lab/alpaca",
            filename="data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
            repo_type="dataset",
            local_dir=str(DATA_ROOT / "tatsu-lab_alpaca"),
        )

    df = pd.read_parquet(parquet_path)[["instruction", "input", "output"]].fillna("")
    train_size = int(os.environ.get("DATAFLEX_BATCH_TRAIN_SIZE", "2048"))
    eval_size = int(os.environ.get("DATAFLEX_BATCH_EVAL_SIZE", "256"))

    train_path = DATA_ROOT / f"alpaca_selector_train_{train_size}.json"
    eval_path = DATA_ROOT / f"alpaca_selector_eval_{eval_size}.json"
    train_path.write_text(
        json.dumps(df.iloc[:train_size].to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(df.iloc[train_size:train_size + eval_size].to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_path, eval_path


def _iter_jsonl_texts(paths):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    yield text.strip()


def _write_text_jsonl(path, texts):
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def prepare_slimpajama_data():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    source_dir = Path(os.environ.get("DATAFLEX_BATCH_SLIMPAJAMA_DIR", DATA_ROOT / "SlimPajama-6B-jsonl"))
    if not source_dir.exists():
        raise FileNotFoundError(f"SlimPajama directory not found: {source_dir}")

    train_size = int(os.environ.get("DATAFLEX_BATCH_TRAIN_SIZE", "2048"))
    eval_size = int(os.environ.get("DATAFLEX_BATCH_EVAL_SIZE", "256"))
    train_path = DATA_ROOT / f"slimpajama_selector_train_{train_size}.jsonl"
    eval_path = DATA_ROOT / f"slimpajama_selector_eval_{eval_size}.jsonl"

    if train_path.exists() and eval_path.exists():
        return train_path, eval_path

    train_files = sorted(source_dir.glob("train-*.jsonl"))
    eval_files = [source_dir / "validation.jsonl"] if (source_dir / "validation.jsonl").exists() else train_files[-1:]
    if not train_files:
        raise FileNotFoundError(f"No train-*.jsonl files found under {source_dir}")

    train_texts = []
    for text in _iter_jsonl_texts(train_files):
        train_texts.append(text)
        if len(train_texts) >= train_size:
            break
    if len(train_texts) < train_size:
        raise ValueError(f"Only collected {len(train_texts)} train texts from {source_dir}, need {train_size}")

    eval_texts = []
    for text in _iter_jsonl_texts(eval_files):
        eval_texts.append(text)
        if len(eval_texts) >= eval_size:
            break
    if len(eval_texts) < eval_size:
        raise ValueError(f"Only collected {len(eval_texts)} eval texts from {eval_files}, need {eval_size}")

    _write_text_jsonl(train_path, train_texts)
    _write_text_jsonl(eval_path, eval_texts)
    return train_path, eval_path


def upsert_dataset_info(train_path, eval_path):
    dataset_info_path = ROOT / "data/dataset_info.json"
    info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_name = os.environ.get("DATAFLEX_BATCH_DATASET", "alpaca").strip().lower()
    if dataset_name in {"slimpajama", "slim_pajama"}:
        info["slimpajama_selector_batch_train"] = {
            "file_name": str(train_path),
            "columns": {"prompt": "text"},
        }
        info["slimpajama_selector_batch_eval"] = {
            "file_name": str(eval_path),
            "columns": {"prompt": "text"},
        }
    else:
        info["alpaca_selector_batch_train"] = {"file_name": str(train_path)}
        info["alpaca_selector_batch_eval"] = {"file_name": str(eval_path)}
    dataset_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def _cluster_less_method(selector_name):
    aliases = {
        "cluster_less": os.environ.get("DATAFLEX_BATCH_CLUSTERING_METHOD", "kmeans"),
        "cluster_less_kmeans": "kmeans",
        "cluster_less_spherical": "spherical_kmeans",
        "cluster_less_spherical_kmeans": "spherical_kmeans",
        "cluster_less_farthest": "farthest_first",
        "cluster_less_farthest_first": "farthest_first",
        "cluster_less_lsh": "random_projection_lsh",
        "cluster_less_random_projection_lsh": "random_projection_lsh",
        "cluster_less_random": "random_partition",
        "cluster_less_random_partition": "random_partition",
    }
    return aliases.get(selector_name)


def upsert_components(selector_name, seed):
    components_path = ROOT / "src/dataflex/configs/components.yaml"
    with components_path.open("r", encoding="utf-8") as f:
        components = yaml.safe_load(f)

    selectors = components.setdefault("selectors", {})
    cache_dir = str(WORK_DIR / "selector_cache" / selector_name / f"seed{seed}")
    cluster_method = _cluster_less_method(selector_name)
    if cluster_method is not None:
        selectors[selector_name] = {
            "name": "cluster_less",
            "params": {
                "cache_dir": cache_dir,
                "gradient_type": "adam",
                "proj_dim": int(os.environ.get("DATAFLEX_BATCH_PROJ_DIM", "1024")),
                "seed": seed,
                "save_interval": int(os.environ.get("DATAFLEX_BATCH_SAVE_INTERVAL", "16")),
                "cluster_size": int(os.environ.get("DATAFLEX_BATCH_CLUSTER_SIZE", "64")),
                "samples_per_cluster": int(os.environ.get("DATAFLEX_BATCH_SAMPLES_PER_CLUSTER", "3")),
                "clustering_batch_size": int(os.environ.get("DATAFLEX_BATCH_CLUSTER_BATCH", "8")),
                "clustering_max_iter": int(os.environ.get("DATAFLEX_BATCH_CLUSTER_ITER", "10")),
                "clustering_method": cluster_method,
                "lsh_num_bits": int(os.environ["DATAFLEX_BATCH_LSH_BITS"]) if os.environ.get("DATAFLEX_BATCH_LSH_BITS") else None,
            },
        }
    elif selector_name == "less":
        selectors[selector_name] = {
            "name": selector_name,
            "params": {
                "cache_dir": cache_dir,
                "gradient_type": "adam",
                "proj_dim": int(os.environ.get("DATAFLEX_BATCH_PROJ_DIM", "1024")),
                "seed": seed,
                "save_interval": int(os.environ.get("DATAFLEX_BATCH_SAVE_INTERVAL", "16")),
            },
        }
    elif selector_name == "random":
        selectors[selector_name] = {
            "name": selector_name,
            "params": {
                "cache_dir": cache_dir,
                "seed": seed,
                "replacement": False,
            },
        }
    else:
        raise ValueError(f"Unsupported selector: {selector_name}")

    with components_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(components, f, allow_unicode=True, sort_keys=False)


def write_train_config(selector_name, seed):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = os.environ.get("DATAFLEX_BATCH_MODEL", "/home/vepfs/models/qwen/Qwen2___5-0___5B-Instruct")
    max_steps = int(os.environ.get("DATAFLEX_BATCH_MAX_STEPS", "80"))
    update_step = int(os.environ.get("DATAFLEX_BATCH_UPDATE_STEP", "20"))
    warmup_step = int(os.environ.get("DATAFLEX_BATCH_WARMUP_STEP", "20"))
    update_times = int(os.environ.get("DATAFLEX_BATCH_UPDATE_TIMES", "3"))
    cutoff_len = int(os.environ.get("DATAFLEX_BATCH_CUTOFF_LEN", "1024"))
    dataset_name = os.environ.get("DATAFLEX_BATCH_DATASET", "alpaca").strip().lower()
    is_slimpajama = dataset_name in {"slimpajama", "slim_pajama"}
    train_dataset = "slimpajama_selector_batch_train" if is_slimpajama else "alpaca_selector_batch_train"
    eval_dataset = "slimpajama_selector_batch_eval" if is_slimpajama else "alpaca_selector_batch_eval"
    stage = os.environ.get("DATAFLEX_BATCH_STAGE") or ("pt" if is_slimpajama else "sft")

    cfg = {
        "model_name_or_path": model_path,
        "trust_remote_code": True,
        "stage": stage,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 8,
        "lora_alpha": 16,
        "dataset": train_dataset,
        "template": "qwen",
        "cutoff_len": cutoff_len,
        "max_samples": int(os.environ.get("DATAFLEX_BATCH_TRAIN_SIZE", "2048")),
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "dataloader_num_workers": 0,
        "seed": seed,
        "output_dir": str(RESULTS_DIR / f"{selector_name}_seed{seed}"),
        "logging_steps": 5,
        "save_steps": max_steps,
        "plot_loss": False,
        "save_only_model": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.0e-4,
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
        "eval_dataset": eval_dataset,
        "per_device_eval_batch_size": 1,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": False,
        "eval_strategy": "steps",
        "eval_steps": update_step,
    }
    path = CONFIG_DIR / f"{selector_name}_seed{seed}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def summarize(experiments):
    rows = []
    for selector_name, seed in experiments:
        out_dir = RESULTS_DIR / f"{selector_name}_seed{seed}"
        state_path = out_dir / "trainer_state.json"
        row = {"selector": selector_name, "seed": seed, "output_dir": str(out_dir)}
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            row["train_loss"] = state.get("train_loss")
            eval_losses = [
                item.get("eval_loss")
                for item in state.get("log_history", [])
                if "eval_loss" in item
            ]
            row["last_eval_loss"] = eval_losses[-1] if eval_losses else None
            row["best_metric"] = state.get("best_metric")
            row["global_step"] = state.get("global_step")
        rows.append(row)

    summary_path = WORK_DIR / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY_PATH", summary_path, flush=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_path, eval_path = prepare_data()
    upsert_dataset_info(train_path, eval_path)

    selectors = os.environ.get("DATAFLEX_BATCH_SELECTORS", "random,cluster_less,less").split(",")
    seeds = [int(x) for x in os.environ.get("DATAFLEX_BATCH_SEEDS", "42,43,44").split(",")]
    experiments = [(selector.strip(), seed) for selector in selectors for seed in seeds]
    print("RUN_ID", RUN_ID, flush=True)
    print("WORK_DIR", WORK_DIR, flush=True)
    print("EXPERIMENTS", experiments, flush=True)

    for selector_name, seed in experiments:
        upsert_components(selector_name, seed)
        cfg_path = write_train_config(selector_name, seed)
        started = time.time()
        try:
            run(["dataflex-cli", "train", str(cfg_path)], env={"DISABLE_VERSION_CHECK": "1"})
        finally:
            elapsed = time.time() - started
            print(f"EXPERIMENT_DONE selector={selector_name} seed={seed} elapsed_sec={elapsed:.1f}", flush=True)

    summarize(experiments)


if __name__ == "__main__":
    main()
