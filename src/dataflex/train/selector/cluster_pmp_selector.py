"""ClusterPMP Selector — Cluster-based data selection with PMP gradient scoring.

Uses a fixed small model for embedding extraction and clustering, then uses the
training model to compute per-cluster gradient contributions via CountSketch
(Perturbation-based Meta-Policy). Samples are drawn proportional to cluster weights.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from dataflex.core.registry import register_selector
from dataflex.train.selector.base_selector import Selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection


# ---------------------------------------------------------------------------
# CountSketchProjector — ported from torchtitan (self-contained)
# ---------------------------------------------------------------------------
class CountSketchProjector:
    """Streaming CountSketch over a model's .grad tensors.

    Mathematical guarantee: E[<sketch(g1), sketch(g2)>] == <g1, g2>
    so the sketch is an unbiased inner-product estimator.

    Args:
        sketch_dim: Output dimension m (default 8192).
        seed: Base RNG seed for hash/sign tables.
    """

    def __init__(self, sketch_dim: int = 8192, seed: int = 42) -> None:
        self.m = int(sketch_dim)
        self.seed = int(seed)
        self._cache: Dict[str, tuple] = {}

    def _get_hash_sign(self, name: str, numel: int, device: torch.device):
        entry = self._cache.get(name)
        if entry is None or entry[0].numel() != numel:
            name_hash = hash(name) & 0xFFFFFFFF
            g = torch.Generator(device="cpu").manual_seed(self.seed + name_hash)
            h = torch.randint(0, self.m, (numel,), generator=g, dtype=torch.int64)
            sign = torch.randint(0, 2, (numel,), generator=g, dtype=torch.float32) * 2 - 1
            self._cache[name] = (h, sign)
        h_cpu, sign_cpu = self._cache[name]
        return h_cpu.to(device, non_blocking=True), sign_cpu.to(device, non_blocking=True)

    def sketch_grad(self, named_params: List[tuple], device: torch.device) -> torch.Tensor:
        """Sketch the current .grad of the supplied (name, param) pairs."""
        s = torch.zeros(self.m, device=device, dtype=torch.float32)
        for name, p in named_params:
            if p.grad is None:
                continue
            g_flat = p.grad.detach().float().reshape(-1)
            h, sign = self._get_hash_sign(name, g_flat.numel(), g_flat.device)
            s.scatter_add_(0, h, g_flat * sign)
        return s

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# ClusterWeightState — ported from torchtitan (self-contained)
# ---------------------------------------------------------------------------
@dataclass
class ClusterWeightState:
    """PMP sampling state: grad_gamma accumulator → softmax weights."""

    num_clusters: int
    temperature: float = 0.5
    min_weight: float = 0.01
    accumulate: bool = True
    drop_bad_clusters: bool = False
    drop_patience: int = 5

    def __post_init__(self) -> None:
        K = int(self.num_clusters)
        self.grad_gamma = np.zeros(K, dtype=np.float64)
        self.weights = np.ones(K, dtype=np.float64) / max(K, 1)
        self.negative_streak = np.zeros(K, dtype=np.int32)
        self.dead = np.zeros(K, dtype=bool)

    def update(self, grad_gamma_delta) -> None:
        """Apply one PMP delta and recompute weights."""
        if isinstance(grad_gamma_delta, torch.Tensor):
            delta = grad_gamma_delta.detach().cpu().double().numpy()
        else:
            delta = np.asarray(grad_gamma_delta, dtype=np.float64)

        if self.accumulate:
            self.grad_gamma += delta
        else:
            self.grad_gamma = delta.copy()

        # Dead-cluster tracking
        if self.drop_bad_clusters:
            for k in range(self.num_clusters):
                if self.dead[k]:
                    continue
                if delta[k] < 0:
                    self.negative_streak[k] += 1
                elif delta[k] > 0:
                    self.negative_streak[k] = 0
                if self.negative_streak[k] >= self.drop_patience:
                    self.dead[k] = True

        # Softmax weights
        logits = -self.grad_gamma / max(self.temperature, 1e-6)
        logits -= logits.max()
        w = np.exp(logits)
        w = np.clip(w, a_min=self.min_weight, a_max=None)
        if self.drop_bad_clusters:
            w[self.dead] = 0.0
        total = w.sum()
        if total <= 0.0:
            alive = (~self.dead).astype(np.float64)
            if alive.sum() == 0:
                alive = np.ones_like(w)
            w = alive / alive.sum()
        else:
            w = w / total
        self.weights = w

    def state_dict(self) -> dict:
        return {
            "grad_gamma": self.grad_gamma.tolist(),
            "weights": self.weights.tolist(),
            "negative_streak": self.negative_streak.tolist(),
            "dead": self.dead.tolist(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        self.grad_gamma = np.asarray(state_dict["grad_gamma"], dtype=np.float64)
        self.weights = np.asarray(state_dict["weights"], dtype=np.float64)
        self.negative_streak = np.asarray(state_dict["negative_streak"], dtype=np.int32)
        self.dead = np.asarray(state_dict["dead"], dtype=bool)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device) for v in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


# ---------------------------------------------------------------------------
# ClusterPmpSelector
# ---------------------------------------------------------------------------
@register_selector("cluster_pmp")
class ClusterPmpSelector(Selector):
    """Cluster-based data selection with PMP gradient scoring.

    1. Extracts embeddings with a small model → clusters training data
    2. Scores clusters via PMP: CountSketch(dev_grad) · CountSketch(cluster_grad)
    3. Converts scores to sampling weights via softmax
    4. Returns weighted-sampled indices
    """

    def __init__(
        self,
        dataset,
        eval_dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        # Small model for embedding extraction
        embedding_model_name_or_path: Optional[str] = None,
        embedding_model_dtype: str = "auto",
        # Clustering parameters
        cluster_size: int = 64,
        num_clusters: Optional[int] = None,
        clustering_batch_size: int = 8,
        clustering_max_iter: int = 20,
        clustering_method: str = "kmeans",
        assignment_chunk_size: int = 4096,
        # PMP scoring parameters
        sketch_dim: int = 8192,
        pmp_lr: float = 0.1,
        temperature: float = 0.5,
        min_weight: float = 0.01,
        n_samples_per_cluster: int = 4,
        accumulate_grad_gamma: bool = True,
        drop_bad_clusters: bool = False,
        drop_patience: int = 5,
        seed: int = 42,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.eval_dataset = eval_dataset
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.embedding_model_dtype = embedding_model_dtype
        self.cluster_size = cluster_size
        self.num_clusters = num_clusters
        self.clustering_batch_size = clustering_batch_size
        self.clustering_max_iter = clustering_max_iter
        self.clustering_method = clustering_method
        self.assignment_chunk_size = assignment_chunk_size
        self.sketch_dim = sketch_dim
        self.pmp_lr = pmp_lr
        self.temperature = temperature
        self.min_weight = min_weight
        self.n_samples_per_cluster = n_samples_per_cluster
        self.accumulate_grad_gamma = accumulate_grad_gamma
        self.drop_bad_clusters = drop_bad_clusters
        self.drop_patience = drop_patience
        self.seed = seed

        self.device = self.accelerator.device
        self._embedding_model = None
        self._sketcher = None
        self._weight_state = None

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(
            f"[ClusterPmpSelector] initialized: embedding_model={embedding_model_name_or_path}, "
            f"cluster_size={cluster_size}, sketch_dim={sketch_dim}, pmp_lr={pmp_lr}"
        )

    # ------------------------------------------------------------------
    # Small model management
    # ------------------------------------------------------------------
    def _get_embedding_model(self):
        """Lazily load the small embedding model."""
        if self._embedding_model is not None:
            return self._embedding_model
        if self.embedding_model_name_or_path is None:
            raise ValueError(
                "[ClusterPmpSelector] embedding_model_name_or_path must be set "
                "to use a small model for clustering."
            )
        dtype_map = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map.get(self.embedding_model_dtype, "auto")
        self._embedding_model = AutoModelForCausalLM.from_pretrained(
            self.embedding_model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self._embedding_model.eval()
        logger.info(f"[ClusterPmpSelector] Loaded embedding model from {self.embedding_model_name_or_path}")
        return self._embedding_model

    def _get_sketcher(self) -> CountSketchProjector:
        if self._sketcher is None:
            self._sketcher = CountSketchProjector(sketch_dim=self.sketch_dim, seed=self.seed)
        return self._sketcher

    def _get_weight_state(self, num_clusters: int) -> ClusterWeightState:
        if self._weight_state is None or self._weight_state.num_clusters != num_clusters:
            self._weight_state = ClusterWeightState(
                num_clusters=num_clusters,
                temperature=self.temperature,
                min_weight=self.min_weight,
                accumulate=self.accumulate_grad_gamma,
                drop_bad_clusters=self.drop_bad_clusters,
                drop_patience=self.drop_patience,
            )
        return self._weight_state

    # ------------------------------------------------------------------
    # Embedding extraction (uses small model)
    # ------------------------------------------------------------------
    def _extract_embedding_features(self, step_id: int) -> torch.Tensor:
        """Extract embeddings using the small model (distributed)."""
        feature_path = os.path.join(self.cache_dir, "cluster", str(step_id), "train_embeddings.pt")

        cached = os.path.exists(feature_path) if self.accelerator.is_main_process else False
        if dist.is_available() and dist.is_initialized():
            obj = [cached]
            dist.broadcast_object_list(obj, src=0)
            cached = obj[0]

        if cached:
            if self.accelerator.is_main_process:
                logger.info(f"[ClusterPmpSelector] Loading cached embeddings from {feature_path}")
            self.accelerator.wait_for_everyone()
            return torch.load(feature_path, map_location="cpu")

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        emb_model = self._get_embedding_model()
        indexed_dataset = list(range(len(self.dataset)))

        def collate_indices(indices):
            examples = [self.dataset[int(i)] for i in indices]
            return torch.tensor(indices, dtype=torch.long), self.data_collator(examples)

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=self.clustering_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_indices,
        )
        dataloader = self.accelerator.prepare(dataloader)

        features = []
        indices_seen = []
        with torch.no_grad():
            for indices, batch in tqdm(
                dataloader,
                desc=f"[Process {self.accelerator.process_index}] PMP embeddings",
                disable=not self.accelerator.is_local_main_process,
                dynamic_ncols=True,
            ):
                batch = _move_to_device(batch, self.device)
                outputs = emb_model(**batch, output_hidden_states=True, return_dict=True)
                hidden = outputs.hidden_states[-1]
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                features.append(pooled.detach().float().cpu())
                indices_seen.append(indices.cpu())

        features = torch.cat(features, dim=0)
        indices_seen = torch.cat(indices_seen, dim=0)

        if dist.is_available() and dist.is_initialized():
            all_features_list = [None] * dist.get_world_size()
            all_indices_list = [None] * dist.get_world_size()
            dist.all_gather_object(all_features_list, features)
            dist.all_gather_object(all_indices_list, indices_seen)
            features = torch.cat(all_features_list, dim=0)
            indices_seen = torch.cat(all_indices_list, dim=0)

        n_samples = len(self.dataset)
        ordered = torch.empty(n_samples, features.shape[1], dtype=features.dtype)
        ordered[indices_seen] = features
        norms = ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)
        ordered = ordered / norms

        if self.accelerator.is_main_process:
            torch.save(ordered, feature_path)
            logger.info(f"[ClusterPmpSelector] Saved embeddings to {feature_path}")
        self.accelerator.wait_for_everyone()
        return ordered

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def _resolve_num_clusters(self, n_samples: int) -> int:
        if n_samples == 0:
            raise ValueError("Cannot cluster an empty training dataset.")
        if self.num_clusters is not None:
            k = int(self.num_clusters)
        else:
            k = int(math.ceil(n_samples / max(1, self.cluster_size)))
        return max(1, min(k, n_samples))

    def _assign_to_centers(self, features: torch.Tensor, centers: torch.Tensor, spherical: bool = False):
        assignments = []
        for start in range(0, len(features), self.assignment_chunk_size):
            chunk = features[start:start + self.assignment_chunk_size]
            if spherical:
                assignments.append((chunk @ centers.T).argmax(dim=1))
            else:
                assignments.append(torch.cdist(chunk, centers).argmin(dim=1))
        return torch.cat(assignments, dim=0)

    def _run_kmeans(self, features: torch.Tensor, step_id: int, spherical: bool = False) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id))
        init_indices = torch.randperm(n_samples, generator=generator)[:k]
        centers = features[init_indices].clone()
        if spherical:
            centers = centers / centers.norm(dim=1, keepdim=True).clamp(min=1e-12)

        for _ in range(max(1, self.clustering_max_iter)):
            assignments = self._assign_to_centers(features, centers, spherical=spherical)
            new_centers = torch.zeros_like(centers)
            counts = torch.bincount(assignments, minlength=k).float().unsqueeze(1)
            new_centers.index_add_(0, assignments, features)
            empty = counts.squeeze(1) == 0
            counts = counts.clamp(min=1.0)
            new_centers = new_centers / counts
            if empty.any():
                replacement = torch.randperm(n_samples, generator=generator)[:int(empty.sum().item())]
                new_centers[empty] = features[replacement]
            if spherical:
                new_centers = new_centers / new_centers.norm(dim=1, keepdim=True).clamp(min=1e-12)
            centers = new_centers

        return self._assign_to_centers(features, centers, spherical=spherical).to(torch.long)

    def _run_random_partition(self, features: torch.Tensor, step_id: int) -> torch.Tensor:
        n_samples = len(features)
        k = self._resolve_num_clusters(n_samples)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id) + 47)
        order = torch.randperm(n_samples, generator=generator)
        assignments = torch.empty(n_samples, dtype=torch.long)
        assignments[order] = torch.arange(n_samples, dtype=torch.long) % k
        return assignments

    def _run_clustering(self, features: torch.Tensor, step_id: int) -> torch.Tensor:
        method = self.clustering_method.lower()
        if method in {"kmeans", "euclidean_kmeans"}:
            return self._run_kmeans(features, step_id, spherical=False)
        if method in {"spherical_kmeans", "cosine_kmeans"}:
            return self._run_kmeans(features, step_id, spherical=True)
        if method in {"random_partition", "random"}:
            return self._run_random_partition(features, step_id)
        raise ValueError(f"Unknown clustering_method: {self.clustering_method}")

    def _get_or_create_clusters(self, step_id: int):
        """Returns (cluster_ids, timing, features_or_None)."""
        cluster_dir = os.path.join(self.cache_dir, "cluster", str(step_id))
        cluster_path = os.path.join(cluster_dir, "train_cluster_ids.pt")
        timing = {"embedding_time_sec": 0.0, "clustering_time_sec": 0.0}

        clusters_cached = os.path.exists(cluster_path) if self.accelerator.is_main_process else False
        if dist.is_available() and dist.is_initialized():
            obj = [clusters_cached]
            dist.broadcast_object_list(obj, src=0)
            clusters_cached = obj[0]

        if not clusters_cached:
            started = time.perf_counter()
            features = self._extract_embedding_features(step_id)
            timing["embedding_time_sec"] = time.perf_counter() - started
        else:
            features = None

        if self.accelerator.is_main_process and clusters_cached:
            cluster_ids = torch.load(cluster_path, map_location="cpu")
            logger.info(f"[ClusterPmpSelector] Loading cached clusters from {cluster_path}")
        elif self.accelerator.is_main_process:
            started = time.perf_counter()
            cluster_ids = self._run_clustering(features, step_id)
            timing["clustering_time_sec"] = time.perf_counter() - started
            os.makedirs(cluster_dir, exist_ok=True)
            torch.save(cluster_ids, cluster_path)
            logger.info(
                f"[ClusterPmpSelector] Built {int(cluster_ids.max().item()) + 1} clusters "
                f"for {len(cluster_ids)} samples with method={self.clustering_method}."
            )
        else:
            cluster_ids = None

        obj = [cluster_ids.tolist() if cluster_ids is not None else None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        cluster_ids = torch.tensor(obj[0], dtype=torch.long)
        return cluster_ids, timing, features

    # ------------------------------------------------------------------
    # PMP Scoring (uses training model)
    # ------------------------------------------------------------------
    def _compute_sketch_for_batch(self, model, batch, sketcher, trainable):
        """Forward + backward + sketch for a single batch."""
        for _, p in trainable:
            if p.grad is not None:
                p.grad.zero_()
        batch = _move_to_device(batch, self.device)
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        sketch = sketcher.sketch_grad(trainable, device=self.device)
        for _, p in trainable:
            if p.grad is not None:
                p.grad.zero_()
        return sketch

    def _compute_pmp_contributions(self, model, cluster_ids: torch.Tensor, step_id: int):
        """Compute grad_gamma_delta for each cluster using training model.

        Returns grad_gamma_delta tensor of shape [num_clusters].
        """
        sketcher = self._get_sketcher()
        num_clusters = int(cluster_ids.max().item()) + 1
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        was_training = model.training
        model.eval()

        # Stage 1: Compute dev gradient sketch q
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.n_samples_per_cluster,
            shuffle=False,
            num_workers=0,
            collate_fn=self.data_collator,
        )
        q = torch.zeros(self.sketch_dim, device=self.device, dtype=torch.float32)
        n_dev = 0
        model.train()  # Need gradients for backward
        for batch in eval_loader:
            q = q + self._compute_sketch_for_batch(model, batch, sketcher, trainable)
            n_dev += 1
            if n_dev >= 4:  # Limit dev batches for efficiency
                break
        if n_dev > 0:
            q = q / float(n_dev)

        # Stage 2: Per-cluster contribution
        grad_gamma_delta = torch.zeros(num_clusters, device=self.device, dtype=torch.float32)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(step_id) + 99)

        for k in range(num_clusters):
            members = torch.where(cluster_ids == k)[0]
            if len(members) == 0:
                continue
            # Sample representatives from cluster k
            n_take = min(self.n_samples_per_cluster, len(members))
            perm = torch.randperm(len(members), generator=generator)[:n_take]
            rep_indices = members[perm].tolist()
            rep_dataset = Subset(self.dataset, rep_indices)

            rep_loader = DataLoader(
                rep_dataset,
                batch_size=n_take,
                shuffle=False,
                num_workers=0,
                collate_fn=self.data_collator,
            )
            for batch in rep_loader:
                v_k = self._compute_sketch_for_batch(model, batch, sketcher, trainable)
                ct_k = torch.dot(q, v_k)
                grad_gamma_delta[k] = self.pmp_lr * ct_k
                break  # One batch per cluster

        if was_training:
            model.train()
        else:
            model.eval()

        # All-reduce across ranks
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_gamma_delta, op=dist.ReduceOp.SUM)
            grad_gamma_delta = grad_gamma_delta / dist.get_world_size()

        return grad_gamma_delta

    # ------------------------------------------------------------------
    # Weighted sampling
    # ------------------------------------------------------------------
    def _weighted_sample(self, cluster_ids: torch.Tensor, weights: np.ndarray, num_samples: int) -> List[int]:
        """Sample indices proportional to cluster weights."""
        num_clusters = len(weights)
        generator = np.random.default_rng(self.seed + 777)

        # Allocate samples per cluster proportional to weight
        raw_alloc = weights * num_samples
        samples_per_cluster = np.floor(raw_alloc).astype(int)
        # Distribute remainder by highest fractional part
        remainder = num_samples - samples_per_cluster.sum()
        if remainder > 0:
            fractions = raw_alloc - samples_per_cluster
            top_clusters = np.argsort(fractions)[::-1][:int(remainder)]
            samples_per_cluster[top_clusters] += 1

        selected = []
        for k in range(num_clusters):
            if samples_per_cluster[k] <= 0:
                continue
            members = torch.where(cluster_ids == k)[0].numpy()
            if len(members) == 0:
                continue
            n_take = min(samples_per_cluster[k], len(members))
            chosen = generator.choice(members, size=n_take, replace=False)
            selected.extend(chosen.tolist())

        return selected

    # ------------------------------------------------------------------
    # Main select
    # ------------------------------------------------------------------
    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """Select samples using cluster-PMP weighted sampling."""
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        # Check cache
        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            obj = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        select_started = time.perf_counter()
        timing = {}

        # Step 1: Cluster (using small model embeddings)
        cluster_ids, cluster_timing, _ = self._get_or_create_clusters(step_id)
        timing.update(cluster_timing)
        num_clusters = int(cluster_ids.max().item()) + 1

        # Step 2: PMP scoring (using training model)
        started = time.perf_counter()
        grad_gamma_delta = self._compute_pmp_contributions(model, cluster_ids, step_id)
        timing["pmp_scoring_time_sec"] = time.perf_counter() - started

        # Step 3: Update weights
        weight_state = self._get_weight_state(num_clusters)
        weight_state.update(grad_gamma_delta)
        weights = weight_state.weights

        # Step 4: Weighted sampling (main process)
        if self.accelerator.is_main_process:
            started = time.perf_counter()
            selected_indices = self._weighted_sample(cluster_ids, weights, num_samples)
            timing["sampling_time_sec"] = time.perf_counter() - started
            timing["total_select_time_sec"] = time.perf_counter() - select_started

            metric_payload = {
                "cluster_weights": weights.tolist(),
                "grad_gamma": weight_state.grad_gamma.tolist(),
                "num_clusters": num_clusters,
                "num_dead_clusters": int(weight_state.dead.sum()),
                "timing": timing,
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(
                f"[ClusterPmpSelector] Selected {len(selected_indices)} samples from "
                f"{num_clusters} clusters. Weight range: [{weights.min():.4f}, {weights.max():.4f}]"
            )
        else:
            selected_indices = None

        obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        return obj[0] or []
