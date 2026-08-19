"""
Shared device selection for every stage that uses torch.

Without this, everything silently runs on CPU even when a GPU is
available -- on Apple Silicon (M-series) that means never using the
MPS backend, which is a large, easy-to-miss speed difference once you
move from single-file inference to actually training on the full
FaceForensics++/DeepFakeDetection dataset.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Best available device: CUDA > Apple Silicon MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")
    if device.type == "cpu":
        print("No GPU acceleration available -- training will be slow. "
              "If you're on Apple Silicon and expected MPS, check that your "
              "torch install is recent enough (torch>=2.2) and that you're "
              "not running under Rosetta (x86) emulation.")
