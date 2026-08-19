"""
Face cache -- extracts aligned faces once per video and writes them to
disk as JPEGs, so a multi-epoch training run doesn't re-decode the
video and re-run MTCNN on every epoch (Stage 2 + Stage 4, but memoized).

Design goals, in order:

1. Never let one bad video (corrupt file, zero detected faces, an
   OpenCV decode failure) crash the whole run. Every failure is caught,
   logged once, and the video is excluded from the returned sample list
   -- callers see a clean, usable dataset rather than a stack trace
   partway through a multi-hour job.
2. Idempotent: re-running the same command after an interrupted run
   only (re)processes videos that don't already have a complete cache
   entry, marked by a `_done.json` file written last.
3. Bounded size: at most `max_frames_per_video` aligned faces are kept
   per video (evenly subsampled if more were detected), so cache size
   and later epoch time stay predictable regardless of clip length.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from datasets import Sample
from stage2_preprocessing import extract_frames, FRAME_SAMPLE_INTERVAL
from stage4_face_detection import detect_and_align

DEFAULT_CACHE_ROOT = "face_cache"
DONE_MARKER = "_done.json"


@dataclass
class CacheEntry:
    sample: Sample
    frame_paths: List[str]        # cached aligned-face JPEGs, evenly spaced through the clip
    confidences: List[float]      # MTCNN detection confidence per cached frame, same order


def _cache_dir_for(sample: Sample, cache_root: str) -> str:
    """Stable, filesystem-safe cache directory for a given video, keyed by
    a hash of its absolute path so identical filenames across different
    manipulation methods (e.g. "000.mp4" appearing under both Deepfakes
    and DeepFakeDetection) never collide."""
    key = hashlib.sha1(os.path.abspath(sample.path).encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(sample.path))[0]
    safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:40]
    return os.path.join(cache_root, sample.collection, sample.manipulation_method, f"{safe_stem}_{key}")


def _evenly_subsample(items: list, k: int) -> list:
    """Pick k items evenly spaced across items (keeps first/last), used to
    cap how many aligned faces are cached per video."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, num=k)
    return [items[int(round(i))] for i in idx]


def build_or_load_cache_entry(
    sample: Sample,
    cache_root: str = DEFAULT_CACHE_ROOT,
    frame_interval: int = FRAME_SAMPLE_INTERVAL,
    max_frames_per_video: int = 32,
    jpeg_quality: int = 95,
) -> Optional[CacheEntry]:
    """Returns a CacheEntry for `sample`, building it if not already
    cached. Returns None (after printing a one-line warning) if the
    video can't be read at all or yields zero usable faces -- callers
    should skip such samples rather than treat this as fatal.
    """
    cache_dir = _cache_dir_for(sample, cache_root)
    done_path = os.path.join(cache_dir, DONE_MARKER)

    if os.path.exists(done_path):
        try:
            with open(done_path) as f:
                meta = json.load(f)
            frame_paths = [os.path.join(cache_dir, name) for name in meta["frames"]]
            if frame_paths and all(os.path.exists(p) for p in frame_paths):
                return CacheEntry(sample=sample, frame_paths=frame_paths, confidences=meta["confidences"])
            # marker exists but files are missing (e.g. cache partially deleted) -- rebuild below
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # corrupt marker -- rebuild below

    try:
        frames = extract_frames(sample.path, interval=frame_interval)
    except Exception as exc:  # noqa: BLE001 -- OpenCV can raise a range of backend-specific errors
        warnings.warn(f"skipping {sample.path!r}: could not read video ({exc})")
        return None

    if not frames:
        warnings.warn(f"skipping {sample.path!r}: 0 frames extracted (corrupt or empty video)")
        return None

    try:
        faces = detect_and_align(frames)
    except Exception as exc:  # noqa: BLE001 -- MTCNN can fail on unusual frame shapes/corrupt data
        warnings.warn(f"skipping {sample.path!r}: face detection failed ({exc})")
        return None

    if not faces:
        warnings.warn(f"skipping {sample.path!r}: 0/{len(frames)} frames had a detectable face")
        return None

    faces = _evenly_subsample(faces, max_frames_per_video)

    os.makedirs(cache_dir, exist_ok=True)
    frame_names, confidences = [], []
    for i, face in enumerate(faces):
        name = f"frame_{i:03d}.jpg"
        out_path = os.path.join(cache_dir, name)
        try:
            Image.fromarray(np.clip(face.aligned_face, 0, 255).astype(np.uint8)).save(
                out_path, "JPEG", quality=jpeg_quality
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"skipping frame {i} of {sample.path!r}: could not write cache image ({exc})")
            continue
        frame_names.append(name)
        confidences.append(float(face.confidence))

    if not frame_names:
        warnings.warn(f"skipping {sample.path!r}: faces detected but none could be cached to disk")
        return None

    with open(done_path, "w") as f:
        json.dump({"frames": frame_names, "confidences": confidences, "source": sample.path}, f)

    frame_paths = [os.path.join(cache_dir, n) for n in frame_names]
    return CacheEntry(sample=sample, frame_paths=frame_paths, confidences=confidences)


def build_cache_for_samples(
    samples: List[Sample],
    cache_root: str = DEFAULT_CACHE_ROOT,
    frame_interval: int = FRAME_SAMPLE_INTERVAL,
    max_frames_per_video: int = 32,
    progress: bool = True,
) -> List[CacheEntry]:
    """Builds/loads cache entries for every sample, skipping (and
    reporting) any that fail rather than aborting the whole batch.
    Returns only the successfully cached entries.
    """
    entries: List[CacheEntry] = []
    skipped = 0
    total = len(samples)
    for i, sample in enumerate(samples, start=1):
        entry = build_or_load_cache_entry(
            sample, cache_root=cache_root, frame_interval=frame_interval,
            max_frames_per_video=max_frames_per_video,
        )
        if entry is None:
            skipped += 1
        else:
            entries.append(entry)
        if progress and (i % 25 == 0 or i == total):
            print(f"  face cache: {i}/{total} videos processed, {skipped} skipped so far")

    if skipped:
        print(f"face cache: {skipped}/{total} video(s) skipped (see warnings above for reasons)")
    if not entries:
        raise RuntimeError(
            "face cache: every video was skipped -- check that root_dir points at a valid "
            "FaceForensics++ download and that videos actually contain a detectable face."
        )
    return entries


if __name__ == "__main__":
    import sys

    from datasets import load_faceforensics

    root_dir = sys.argv[1] if len(sys.argv) > 1 else None
    samples = load_faceforensics(root_dir=root_dir) if root_dir else load_faceforensics()
    print(f"Building face cache for {len(samples)} videos ...")
    entries = build_cache_for_samples(samples)
    print(f"Cached {len(entries)} videos successfully.")
