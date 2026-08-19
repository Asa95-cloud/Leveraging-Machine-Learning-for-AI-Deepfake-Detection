"""
Stage 5 -- Feature extraction.

Two complementary feature families are computed per aligned face
(Section 3.5 of the methodology):

  1. Spatial features -- the penultimate-layer activations of the CNN
     backbone (Stage 6's Xception network), capturing texture and
     blending inconsistencies typical of face-swapped or reenacted
     regions.
  2. Frequency-domain features -- a radially-averaged 2D FFT power
     spectrum, capturing the spectral artifacts that GAN- and
     diffusion-based generators tend to leave behind and that are
     often invisible in the spatial domain alone.

Only the frequency-domain half is implemented here as a standalone,
inspectable function; the spatial half is produced as a side effect of
the forward pass in Stage 6 and is imported from there to avoid
running the backbone twice.
"""

from __future__ import annotations

import numpy as np


def radial_fft_spectrum(gray_image: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Compute a radially-averaged FFT power spectrum for a grayscale image.

    Returns a 1D feature vector of length `n_bins`, low frequencies
    first. This is a lightweight, model-free signal that can be fed to
    the classifier alongside CNN features, or inspected on its own as
    a sanity check independent of the learned model.
    """
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    r_max = r.max()
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    spectrum = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        spectrum[i] = magnitude[mask].mean() if mask.any() else 0.0

    # normalize so the feature is comparable across images of different scale
    norm = np.linalg.norm(spectrum)
    return spectrum / norm if norm > 0 else spectrum


def to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """Standard luma conversion, avoiding an extra OpenCV/PIL dependency here."""
    return (0.299 * rgb_image[..., 0] + 0.587 * rgb_image[..., 1] + 0.114 * rgb_image[..., 2])


def extract_frequency_features(aligned_face_rgb: np.ndarray) -> np.ndarray:
    gray = to_grayscale(aligned_face_rgb)
    return radial_fft_spectrum(gray)


if __name__ == "__main__":
    import sys

    from stage2_preprocessing import preprocess
    from stage4_face_detection import detect_and_align

    frames = preprocess("video", sys.argv[1])
    faces = detect_and_align(frames)
    if faces:
        feats = extract_frequency_features(faces[0].aligned_face)
        print(f"frequency feature vector: shape={feats.shape}, "
              f"first 5 bins={np.round(feats[:5], 4)}")
    else:
        print("no faces detected")
