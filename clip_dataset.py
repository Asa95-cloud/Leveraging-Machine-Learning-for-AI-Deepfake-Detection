"""
ClipDataset -- turns cached, aligned face crops (see face_cache.py) into
fixed-length clip tensors of shape (T, C, H, W) for
`stage6_classification.TemporalDeepfakeClassifier`.

Kept separate from face_cache.py so caching (slow, I/O + MTCNN bound,
done once) and sampling (fast, done every epoch) can vary independently
-- e.g. re-running training with a different `frames_per_clip` doesn't
require rebuilding the cache.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from face_cache import CacheEntry


def _sample_frame_indices(n_available: int, n_wanted: int, rng: np.random.Generator, deterministic: bool) -> List[int]:
    """Pick `n_wanted` indices into a clip's cached frames.

    If the clip has fewer cached frames than requested, indices repeat
    (sampled with replacement) rather than raising -- short clips are
    common enough in this dataset that padding beats dropping them.
    `deterministic=True` (used for val/test) always returns the same,
    evenly-spaced indices so evaluation metrics are reproducible epoch
    to epoch; `deterministic=False` (train) samples randomly for a
    cheap form of temporal augmentation.
    """
    if n_available <= 0:
        return []
    if deterministic:
        if n_available >= n_wanted:
            return [int(round(i)) for i in np.linspace(0, n_available - 1, num=n_wanted)]
        # pad by cycling through the available frames in order
        return [i % n_available for i in range(n_wanted)]
    if n_available >= n_wanted:
        return sorted(rng.choice(n_available, size=n_wanted, replace=False).tolist())
    return sorted(rng.choice(n_available, size=n_wanted, replace=True).tolist())


class ClipDataset(Dataset):
    """Wraps a list of `face_cache.CacheEntry` objects. Each `__getitem__`
    returns (clip_tensor, label) where clip_tensor has shape
    (frames_per_clip, 3, H, W) and label is 0 (real) or 1 (fake).
    """

    def __init__(self, entries: List[CacheEntry], frames_per_clip: int, transform, train: bool, seed: int = 0):
        if not entries:
            raise ValueError("ClipDataset received an empty entry list -- nothing to train/evaluate on.")
        self.entries = entries
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.train = train
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        indices = _sample_frame_indices(
            len(entry.frame_paths), self.frames_per_clip, self._rng, deterministic=not self.train
        )
        frames = []
        for i in indices:
            img = Image.open(entry.frame_paths[i]).convert("RGB")
            arr = np.asarray(img)
            frames.append(self.transform(arr))
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        label = 1 if entry.sample.label == "fake" else 0
        return clip, label

    def sample_ref(self, idx: int):
        """Return the underlying Sample for `idx`, useful when reporting
        per-video predictions (e.g. selecting a video for Grad-CAM)."""
        return self.entries[idx].sample
