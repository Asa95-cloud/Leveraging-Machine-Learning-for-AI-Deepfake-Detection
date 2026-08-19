"""
Stage 1 -- Media upload and validation.

Accepts a user-supplied image or video, verifies that its declared
extension matches its true file type (not just the extension string),
and rejects anything malformed or oversized before it ever reaches the
rest of the pipeline. This is the entry point described in Section 3
(Methodology, Stage 1) of the accompanying research paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import magic  # python-magic: reads the file's actual signature, not its extension

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
MAX_FILE_SIZE_MB = 500


@dataclass
class UploadResult:
    path: str
    media_type: str          # "image" or "video"
    mime_type: str
    size_mb: float
    valid: bool
    reason: str = ""


def validate_media(file_path: str) -> UploadResult:
    """Validate a single uploaded file and classify it as image or video.

    Raises no exceptions for "bad" files -- callers should check
    `UploadResult.valid` and act on `UploadResult.reason`.
    """
    if not os.path.isfile(file_path):
        return UploadResult(file_path, "unknown", "", 0.0, False, "file not found")

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return UploadResult(file_path, "unknown", "", size_mb, False,
                             f"file exceeds {MAX_FILE_SIZE_MB} MB limit")

    mime_type = magic.from_file(file_path, mime=True)

    if mime_type in ALLOWED_IMAGE_TYPES:
        media_type = "image"
    elif mime_type in ALLOWED_VIDEO_TYPES:
        media_type = "video"
    else:
        return UploadResult(file_path, "unknown", mime_type, size_mb, False,
                             f"unsupported mime type: {mime_type}")

    return UploadResult(file_path, media_type, mime_type, size_mb, True)


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        result = validate_media(path)
        status = "OK" if result.valid else f"REJECTED ({result.reason})"
        print(f"{path}: {result.media_type} / {result.mime_type} "
              f"({result.size_mb:.2f} MB) -> {status}")
