# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlex is a data-centric dynamic training framework for LLMs built on top of LLaMA-Factory. It provides dynamic **Data Selection**, **Data Mixture**, and **Data Reweighting** during training via runtime monkey-patching of LlamaFactory classes.

## Build & Install

```bash
pip install -e .                # editable install (installs llamafactory automatically for Python 3.11+)
pip install -e ".[dev]"         # includes pre-commit, ruff, pytest, build
```

For Python 3.10, install `llamafactory` manually since the auto-dependency requires Python 3.11+.

## Common Commands

```bash
# CLI
dataflex-cli version                          # verify installation
dataflex-cli train <config.yaml>              # run training
dataflex-cli train config.yaml key=value      # OmegaConf-style overrides

# Multi-GPU
FORCE_TORCHRUN=1 dataflex-cli train config.yaml

# Lint & format
ruff check src/                               # lint
ruff format src/                              # format

# Smoke tests (CI runs these)
python -c "from dataflex import __version__; print(__version__)"
python -c "from dataflex.core.registry import Registry"
python -c "from dataflex.train.selector import *"
python -c "from dataflex.train.mixer import *"
python -c "from dataflex.train.weighter import *"

# Minimal training configs for testing (require a GPU + model access)
dataflex-cli train examples/test/test_select.yaml
dataflex-cli train examples/test/test_mix.yaml
dataflex-cli train examples/test/test_weight.yaml
```

## Code Style

- **Ruff** with `line-length = 119`, target Python 3.10
- Import ordering: isort with `known-first-party = ["dataflex"]`
- Quote style: double quotes, space indent

## Architecture

### Monkey-Patching Flow

When `dataflex-cli train config.yaml` executes:

1. `cli.main()` reads YAML via OmegaConf
2. `patch_finetune_params()` — replaces LlamaFactory's `FinetuningArguments` and `DataArguments` with DataFlex versions (`DynamicFinetuningArguments`, extended `DataArguments`)
3. `patch_trainer(train_type)` — based on `train_type` field, injects `SelectTrainer`, `MixTrainer`, or `WeightTrainer` in place of LlamaFactory's `CustomSeq2SeqTrainer`/`CustomTrainer`
4. `patch_get_dataset()` — (dynamic_mix only) replaces `get_dataset` to support multi-source mixture sampling
5. Calls `llamafactory.train.tuner.run_exp()` which now uses patched classes

### Registry System (`core/registry.py`)

All algorithms are registered via decorators and instantiated through a central `REGISTRY`:

```python
@register_selector("my_method")   # registers under kind="selector"
@register_mixer("my_method")      # registers under kind="mixer"
@register_weighter("my_method")   # registers under kind="weighter"
```

`REGISTRY.build(kind, name, runtime=..., cfg=...)` merges runtime deps (from trainer) with config params (from `components.yaml`), inspects `__init__` signature, and passes only accepted kwargs.

### Training Modes (`train_type` values)

| `train_type` | Trainer Class | Component Type |
|---|---|---|
| `dynamic_select` | `SelectTrainer` | Selector |
| `dynamic_mix` | `MixTrainer` | Mixer |
| `dynamic_weight` | `WeightTrainer` | Weighter |
| `static` | Default LlamaFactory trainer | None |

### Key Source Layout

```
src/dataflex/
├── cli.py                    # Entry point, monkey-patching orchestration
├── launcher.py               # torchrun distributed launcher
├── core/registry.py          # Registry + register_* decorators
├── configs/components.yaml   # Algorithm parameter presets
├── train/
│   ├── selector/             # Selector implementations (base + LESS, NICE, Loss, etc.)
│   ├── mixer/                # Mixer implementations (base + DoReMi, ODM, etc.)
│   ├── weighter/             # Weighter implementations (base + LossWeighter, etc.)
│   ├── trainer/              # SelectTrainer, MixTrainer, WeightTrainer
│   ├── hparams/              # DynamicFinetuningArguments, extended DataArguments
│   ├── data/loader.py        # Patched get_dataset for multi-source mixing
│   └── dataset/              # MixedProportionManager
├── offline_selector/         # Standalone preprocessing (TSDS, NEAR)
└── utils/                    # load_component, logging, selector_io
```

### Adding a New Algorithm

1. Create `src/dataflex/train/{selector,mixer,weighter}/my_algo.py`
2. Subclass `Selector`/`Mixer`/`Weighter`, decorate with `@register_{type}("name")`
3. Add `from .my_algo import *` to the package `__init__.py`
4. Add params in `src/dataflex/configs/components.yaml`
5. Create a training YAML with `component_name: name`

### Base Class Interfaces

- **Selector**: `__init__(dataset, accelerator, data_collator, cache_dir)` + abstract `select(model, step_id, num_samples, **kwargs) -> List[int]`
- **Mixer**: `__init__(mixture_manager)` + abstract `mix(model, step_id, **kwargs) -> np.ndarray`
- **Weighter**: `__init__(**kwargs)` + abstract `get_weighted_loss(losses, *, ctx, model, inputs) -> torch.Tensor`

### Training Loop Semantics

- **SelectTrainer**: warmup random → at `warmup_step` and every `update_step`, calls `selector.select()` → rebuilds dataloader with subset indices. Total Flex epoch = `warmup_step + update_step * update_times`. Supports `num_train_epochs` for multi-epoch repetition.
- **MixTrainer**: adjusts domain proportions via `mixer.mix()` at intervals; rebuilds mixed dataset.
- **WeightTrainer**: standard dataloader, applies `weighter.get_weighted_loss()` to per-sample losses after `warmup_step` global steps.

### Distributed Training Notes

- Selectors run selection logic on `is_main_process` only, then `dist.broadcast_object_list` to all ranks.
- DeepSpeed ZeRO-3 is supported for gradient computation.
- The `launcher.py` handles `torchrun` spawning when `FORCE_TORCHRUN=1` or multiple GPUs detected.
