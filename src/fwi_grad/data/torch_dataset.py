"""Lazy CPU dataset and shot-padding collate helper for FWI trajectories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, default_collate

from ..models.conditioning import encode_condition
from ..models.normalization import SignPreservingNormalizer

_IMAGES = ("v_k", "v_init", "v_true", "illumination", "g_sparse", "g_ref")
_GATHERS = ("r_wave", "r_env", "r_phase")


class FWIGradientDataset(Dataset):
    """Load one split of multi-modal FWI trajectory samples from a manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        scenarios: Sequence[str] | None = None,
        max_iters: int = 20,
        normalizer: SignPreservingNormalizer | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if split not in manifest["splits"]:
            raise ValueError(f"unknown manifest split {split!r}")
        wanted = None if scenarios is None else set(scenarios)
        self.entries = [
            row
            for row in manifest["samples"]
            if row["split"] == split and (wanted is None or row["scenario"] in wanted)
        ]
        self.directory = manifest_path.parent
        self.max_iters = max_iters
        self.normalizer = normalizer if normalizer is not None else SignPreservingNormalizer()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        row = self.entries[index]
        sample = torch.load(self.directory / row["path"], map_location="cpu", weights_only=True)
        result = {key: sample[key].float().unsqueeze(0) for key in _IMAGES}
        result.update({key: sample[key].float() for key in _GATHERS})
        norm, scale = self.normalizer.normalize(result["g_ref"])
        meta = sample["meta"]
        shot_fraction = float(meta.get("shot_fraction", meta["sparse_ratio"]))
        result.update(
            g_ref_norm=norm,
            g_ref_scale=scale.reshape(1),
            scenario=row["scenario"],
            f_cut=float(meta["f_cut"]),
            q_k=encode_condition(meta["f_cut"], meta["iter_k"], shot_fraction),
        )
        return result


def collate_fwi_batch(batch: list[dict]) -> dict:
    """Collate trajectory samples and pad variable-length shot gathers."""
    result = {
        key: default_collate([sample[key] for sample in batch])
        for key in batch[0]
        if key not in _GATHERS
    }
    lengths = torch.tensor([len(sample["r_wave"]) for sample in batch])
    for key in _GATHERS:
        result[key] = pad_sequence([sample[key] for sample in batch], batch_first=True)
    result["shot_mask"] = torch.arange(result["r_wave"].size(1))[None, :] < lengths[:, None]
    return result
