"""
Stage 2 -- Preprocessing.

Video is decomposed into frames at a fixed sampling interval. This
stage hands RAW (uint8, full-resolution) frames to Stage 4 for face
detection -- resizing to the classifier's input resolution and scaling
to [0, 1] happens AFTER face detection/alignment, applied to the
aligned face crop (via `inference_transform`/`train_augmentations`
below), not to the full frame beforehand. Training-time augmentation
simulates the recompression media typically undergoes after upload to
social platforms, per Section 3.5 of the methodology.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from torchvision import transforms

TARGET_SIZE = (224, 224)      # Xception / EfficientNet input resolution
FRAME_SAMPLE_INTERVAL = 5     # extract every Nth frame


def extract_frames(video_path: str, interval: int = FRAME_SAMPLE_INTERVAL) -> List[np.ndarray]:
    """Sample frames from a video at a fixed interval using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def normalize_image(image: np.ndarray, size=TARGET_SIZE) -> np.ndarray:
    """Resize to the model's expected input size and scale to [0, 1]."""
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# Training-time augmentation pipeline (Section 3.2): random crop, flip,
# compression-level jitter (via JPEG-style random quality resave upstream),
# and brightness/contrast variation.
train_augmentations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
])

# Deterministic pipeline used at inference time -- no randomness.
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
])


def preprocess(media_type: str, path: str) -> List[np.ndarray]:
    """Unified entry point: returns a list of RAW, full-resolution RGB
    frames (uint8, original size) ready for Stage 4 (face detection).

    Deliberately does NOT resize/normalize here: Stage 4's face
    detector (MTCNN) expects uint8 pixels in [0, 255] at (close to)
    original resolution -- shrinking to the classifier's 224x224 input
    size and scaling to [0, 1] before detection makes faces too small
    to find reliably, and washes out the contrast MTCNN's internal
    normalization assumes, causing near-total detection failure. Once
    Stage 4 produces an aligned face crop, THAT is what gets resized/
    normalized for the classifier (see `inference_transform` /
    `train_augmentations` below, applied to `DetectedFace.aligned_face`
    in Stage 6, not to these raw frames).

    A still image yields a list of length 1 so downstream stages can
    treat images and videos identically.
    """
    if media_type == "video":
        return extract_frames(path)
    return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)]


if __name__ == "__main__":
    import sys

    frames = preprocess("video", sys.argv[1])
    print(f"extracted {len(frames)} raw frame(s), "
          f"shape={frames[0].shape if frames else None}, dtype={frames[0].dtype if frames else None}")
