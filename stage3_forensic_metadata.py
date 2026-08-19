"""
Stage 3 -- Forensic metadata capture.

This is the stage that distinguishes a forensic tool from a bare
classifier (Section 3.3 / 3.6 of the methodology). Before any pixel is
touched by the model, the pipeline fixes an evidentiary baseline:

  * a SHA-256 hash of the untouched original file (chain of custody),
  * embedded EXIF / container metadata (absence or inconsistency is
    itself a forensic signal),
  * an Error Level Analysis (ELA) map flagging regions with
    inconsistent JPEG compression history, a common by-product of
    splicing or face-swapping.

Digital forensic literature (Qureshi et al., 2024) emphasizes exactly
this kind of verifiable evidence-handling alongside the detection
model itself, rather than relying on the classifier's output alone.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PIL import Image, ExifTags


@dataclass
class ForensicRecord:
    file_path: str
    sha256: str
    captured_at: str
    exif: dict = field(default_factory=dict)
    ela_mean_error: Optional[float] = None
    ela_max_error: Optional[float] = None
    notes: str = ""


def compute_sha256(file_path: str, chunk_size: int = 8192) -> str:
    """Hash the file exactly as uploaded -- before any resizing or recompression."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_exif(image_path: str) -> dict:
    """Pull embedded EXIF tags, if any. Missing/blank EXIF is itself notable."""
    try:
        img = Image.open(image_path)
        raw = img.getexif()  # public API; the older private _getexif() was removed in modern Pillow
        if not raw:
            return {}
        return {ExifTags.TAGS.get(tag, tag): value for tag, value in raw.items()}
    except Exception:
        return {}


def error_level_analysis(image_path: str, quality: int = 90) -> tuple[float, float]:
    """Resave the image at a known JPEG quality and diff against the original.

    Regions that were spliced or re-rendered typically show a different
    compression signature than the rest of the frame, producing a
    visibly higher error in the ELA difference map. Returns
    (mean_error, max_error) as summary statistics; the full difference
    map can be kept for visual inspection in the forensic report.
    """
    original = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = np.abs(np.asarray(original, dtype=np.int16) - np.asarray(resaved, dtype=np.int16))
    return float(diff.mean()), float(diff.max())


def capture_forensic_record(file_path: str, is_image: bool) -> ForensicRecord:
    record = ForensicRecord(
        file_path=file_path,
        sha256=compute_sha256(file_path),
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
    if is_image:
        record.exif = extract_exif(file_path)
        try:
            record.ela_mean_error, record.ela_max_error = error_level_analysis(file_path)
        except Exception as exc:  # noqa: BLE001
            record.notes = f"ELA failed: {exc}"
    else:
        record.notes = "ELA is defined for still images; per-frame ELA can be run downstream."
    return record


if __name__ == "__main__":
    import sys

    rec = capture_forensic_record(sys.argv[1], is_image=True)
    print(rec)
