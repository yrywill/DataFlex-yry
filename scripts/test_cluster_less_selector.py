import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from dataflex.train.selector.cluster_less_selector import ClusterLessSelector


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, size, seq_len=6, vocab_size=32):
        self.examples = []
        for idx in range(size):
            ids = torch.tensor(
                [((idx * 3 + pos) % (vocab_size - 1)) + 1 for pos in range(seq_len)],
                dtype=torch.long,
            )
            self.examples.append(
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "labels": torch.tensor(idx % 2, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


class ToyModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8, num_labels=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True):
        hidden = self.embed(input_ids)
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(pooled)
        loss = nn.functional.cross_entropy(logits, labels)
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)


class FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.process_index = 0
        self.is_main_process = True
        self.is_local_main_process = True
        self.state = SimpleNamespace(deepspeed_plugin=None)

    @contextmanager
    def no_sync(self, model):
        yield

    def backward(self, loss):
        loss.backward()

    def prepare(self, obj):
        return obj

    def wait_for_everyone(self):
        return None


def main():
    torch.manual_seed(7)
    cache_dir = Path("/tmp/dataflex_cluster_less_smoke")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    train_dataset = ToyDataset(size=12)
    eval_dataset = ToyDataset(size=4)
    model = ToyModel()

    selector = ClusterLessSelector(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        accelerator=FakeAccelerator(),
        data_collator=collate,
        cache_dir=str(cache_dir),
        gradient_type="sgd",
        proj_dim=16,
        save_interval=2,
        seed=123,
        cluster_size=4,
        samples_per_cluster=2,
        clustering_batch_size=3,
        clustering_max_iter=3,
    )

    selected = selector.select(model=model, step_id=10, num_samples=5)
    assert len(selected) == 5, selected
    assert len(set(selected)) == 5, selected
    assert all(0 <= idx < len(train_dataset) for idx in selected), selected
    assert (cache_dir / "step_10.json").exists()
    assert (cache_dir / "cluster" / "10" / "train_cluster_ids.pt").exists()
    print("cluster_less smoke ok")
    print("selected:", selected)


if __name__ == "__main__":
    main()
