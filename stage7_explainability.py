"""
Stage 7 -- Explainability and confidence scoring.

Grad-CAM (Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R.,
Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from
deep networks via gradient-based localization. Proceedings of the
IEEE International Conference on Computer Vision (ICCV), 618-626) is
applied to the trained Xception backbone to produce a heatmap showing
which facial regions drove each prediction -- the piece that keeps the
verdict inspectable rather than a black box (Section 3.6).

For video, per-frame logits are aggregated into one video-level
confidence score, down-weighting frames where Stage 4's face detector
was itself uncertain (Section 3.6 / 3.8).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class _LogitOnlyWrapper(torch.nn.Module):
    """pytorch-grad-cam requires `model(x)` to return a single (batch,
    num_classes)-shaped tensor so it can index into it for the target
    class. `DeepfakeClassifier.forward` returns `(logits, features)` --
    convenient for training, but incompatible with Grad-CAM as-is (it
    would otherwise try to index a tuple and fail). This wrapper exposes
    just the logit, reshaped to (batch, 1) since it's a single binary
    logit rather than a multi-class row, which Grad-CAM treats correctly
    as "class 0" (the only class) via its default target selection.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.unsqueeze(-1) if logits.dim() == 1 else logits


def _find_last_conv_layer(module: torch.nn.Module) -> torch.nn.Module:
    """Walk the module tree and return the last nn.Conv2d encountered.

    Grad-CAM needs a late convolutional layer to hook into. Rather than
    hardcoding a specific attribute path (e.g. `backbone.conv4`), which
    would silently break if timm ever renames internals, this walks the
    actual module graph and picks the last conv layer it finds -- correct
    for any CNN backbone, not just the exact Xception variant in use.
    """
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No nn.Conv2d layer found in the model; Grad-CAM needs a conv layer to target.")
    return last_conv


def build_gradcam(model: torch.nn.Module) -> GradCAM:
    """Targets the last convolutional layer of the Xception backbone
    (found automatically via `_find_last_conv_layer`), wrapping `model`
    so its (logits, features) tuple output is Grad-CAM-compatible (see
    `_LogitOnlyWrapper`). The conv layer is located on the original,
    unwrapped model -- wrapping doesn't clone parameters, so the same
    layer object is reachable (and hookable) either way.
    """
    target_layer = _find_last_conv_layer(model.extractor.backbone)
    wrapped = _LogitOnlyWrapper(model)
    return GradCAM(model=wrapped, target_layers=[target_layer])


def explain_prediction(cam: GradCAM, input_tensor: torch.Tensor,
                        rgb_image_float: np.ndarray) -> np.ndarray:
    """Returns an RGB heatmap overlay (H, W, 3) in [0, 1], ready to save/embed
    in the Stage 8 forensic report."""
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W) in [0, 1]
    overlay = show_cam_on_image(rgb_image_float, grayscale_cam, use_rgb=True)
    return overlay


def aggregate_video_confidence(
    frame_logits: List[float],
    detection_confidences: List[float],
) -> Tuple[float, str]:
    """Weighted-average frame-level fake-probabilities into one verdict.

    Frames where Stage 4's face detector was itself low-confidence are
    down-weighted, since a poorly detected face produces a noisy,
    unreliable classification for that frame.
    """
    if not frame_logits:
        return 0.0, "no usable frames"

    probs = 1 / (1 + np.exp(-np.array(frame_logits)))     # sigmoid
    weights = np.clip(np.array(detection_confidences), 1e-3, 1.0)
    weighted_score = float(np.average(probs, weights=weights))

    verdict = "likely manipulated" if weighted_score >= 0.5 else "likely authentic"
    return weighted_score, verdict


if __name__ == "__main__":
    # smoke test with synthetic numbers -- no model/image I/O required
    logits = [2.1, 1.8, -0.4, 2.4]
    det_conf = [0.98, 0.95, 0.40, 0.99]
    score, verdict = aggregate_video_confidence(logits, det_conf)
    print(f"video confidence={score:.3f} -> {verdict}")
