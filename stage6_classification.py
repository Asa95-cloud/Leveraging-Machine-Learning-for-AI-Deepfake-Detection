"""
Stage 6 -- Deepfake classification.

Backbone: Xception (Chollet, F. (2017). Xception: Deep learning with
depthwise separable convolutions. Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), 1251-1258),
pretrained on ImageNet and fine-tuned on the FaceForensics++ /
DeepFakeDetection training set (see datasets.py).

For video, per-frame Xception embeddings are passed through an LSTM to
model inter-frame consistency, since single-frame classification alone
cannot capture the flicker/temporal artifacts that distinguish many
deepfakes at the video level (Section 3.8 of the methodology).
"""

from __future__ import annotations

from typing import List, Tuple

import timm
import torch
import torch.nn as nn


class XceptionFeatureExtractor(nn.Module):
    """Wraps a pretrained Xception, exposing pooled penultimate features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # timm renamed "xception" -> "legacy_xception" (same architecture/weights,
        # implementing Chollet (2017)); using the canonical name directly avoids
        # depending on timm's deprecated-name redirect.
        self.backbone = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (batch, feature_dim)


class DeepfakeClassifier(nn.Module):
    """Frame-level classifier head on top of Xception features.

    Used directly for still images. For video, `TemporalDeepfakeClassifier`
    below consumes a sequence of these per-frame feature vectors.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.extractor = XceptionFeatureExtractor(pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.extractor.feature_dim, 1),  # binary logit: fake probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        return self.head(features).squeeze(-1), features


class TemporalDeepfakeClassifier(nn.Module):
    """Adds an LSTM over per-frame Xception features for video input."""

    def __init__(self, pretrained: bool = True, hidden_size: int = 256, dropout: float = 0.3):
        super().__init__()
        self.extractor = XceptionFeatureExtractor(pretrained=pretrained)
        self.lstm = nn.LSTM(
            input_size=self.extractor.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, frame_batch: torch.Tensor) -> torch.Tensor:
        """frame_batch: (batch, seq_len, C, H, W) -> per-video logit."""
        b, t, c, h, w = frame_batch.shape
        flat = frame_batch.view(b * t, c, h, w)
        features = self.extractor(flat).view(b, t, -1)
        _, (h_n, _) = self.lstm(features)
        return self.head(h_n[-1]).squeeze(-1)


def train_step(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor],
               optimizer: torch.optim.Optimizer, pos_weight: torch.Tensor) -> float:
    """One optimization step with class-weighted binary cross-entropy (Section 3.5)."""
    inputs, labels = batch
    optimizer.zero_grad()
    logits = model(inputs)
    if isinstance(logits, tuple):
        logits = logits[0]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = loss_fn(logits, labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def build_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


if __name__ == "__main__":
    model = DeepfakeClassifier(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    logits, feats = model(dummy)
    print(f"logits shape={logits.shape}, feature dim={feats.shape[-1]}")
