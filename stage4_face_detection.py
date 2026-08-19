"""
Stage 4 -- Face detection and alignment.

The manipulation methods represented in FaceForensics++, DFDC, and
Celeb-DF are predominantly face-centric (Rossler et al., 2019;
Dolhansky et al., 2020; Li et al., 2020), so every frame is passed
through a face detector and the detected face is aligned to a
canonical pose before features are extracted. This implementation
uses MTCNN (Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint
face detection and alignment using multi-task cascaded convolutional
networks. IEEE Signal Processing Letters, 23(10), 1499-1503), via the
`facenet-pytorch` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from facenet_pytorch import MTCNN

# MTCNN is deliberately pinned to CPU, NOT device.get_device(). facenet-pytorch's
# MTCNN hits a documented PyTorch MPS bug on Apple Silicon --
# "Adaptive pool MPS: input sizes must be divisible by output sizes"
# (https://github.com/pytorch/pytorch/issues/96056) -- which makes detection fail
# on essentially every frame when MTCNN runs on the "mps" device. MTCNN's PNet/RNet/ONet
# are small relative to the Xception+LSTM classifier, so running detection on CPU while
# training/inference still use MPS/CUDA (via device.get_device() elsewhere) costs little
# and avoids a hard, whole-video failure on every Mac with Apple Silicon.
_detector = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,          # keep only the highest-confidence face per frame
    post_process=False,
    select_largest=True,
    device=torch.device("cpu"),
)


@dataclass
class DetectedFace:
    frame_index: int
    aligned_face: np.ndarray     # HxWx3, aligned + cropped
    confidence: float
    box: Optional[List[float]]   # [x1, y1, x2, y2] in original frame coordinates


def detect_and_align(frames: List[np.ndarray]) -> List[DetectedFace]:
    """Run MTCNN over a list of RGB frames, returning aligned face crops.

    Frames with no detected face are skipped so downstream stages never
    see empty input; the caller can compare `len(result)` against
    `len(frames)` to see how many frames were unusable (relevant to the
    per-frame confidence weighting used in Stage 7).
    """
    results: List[DetectedFace] = []

    for idx, frame in enumerate(frames):
        boxes, probs = _detector.detect(frame)
        if boxes is None or probs is None or probs[0] is None:
            continue

        aligned = _detector(frame)  # returns the aligned face tensor/array or None
        if aligned is None:
            continue

        aligned_np = aligned.permute(1, 2, 0).cpu().numpy() if hasattr(aligned, "permute") else aligned

        results.append(DetectedFace(
            frame_index=idx,
            aligned_face=aligned_np,
            confidence=float(probs[0]),
            box=boxes[0].tolist() if boxes is not None else None,
        ))

    return results


if __name__ == "__main__":
    import sys

    from stage2_preprocessing import preprocess

    frames = preprocess("video", sys.argv[1])
    faces = detect_and_align(frames)
    print(f"{len(faces)}/{len(frames)} frames yielded a usable aligned face")
