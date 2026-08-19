"""
train.py -- Complete training and evaluation with 8 visuals and 4 tables
for the deepfake detection report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report

import datasets
import evaluation
from clip_dataset import ClipDataset
from device import get_device
from face_cache import build_cache_for_samples, DEFAULT_CACHE_ROOT
from stage2_preprocessing import inference_transform, train_augmentations
from stage6_classification import TemporalDeepfakeClassifier, build_optimizer, train_step
from stage7_explainability import build_gradcam, explain_prediction

from PIL import Image


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-dir", default=datasets.DEFAULT_ROOT_DIR)
    p.add_argument("--collections", default="actors,youtube")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--max-videos", type=int, default=100, help="max videos to use")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--frames-per-clip", type=int, default=8)
    p.add_argument("--max-frames-per-video", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradcam-examples", type=int, default=4)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def balanced_video_split(
    samples: List[datasets.Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[datasets.Sample], List[datasets.Sample], List[datasets.Sample]]:
    """Robust split ensuring both classes in each split."""
    labels = [1 if s.label == "fake" else 0 for s in samples]
    
    # Try identity-based grouping first
    by_identity: Dict[str, List[datasets.Sample]] = {}
    for s in samples:
        by_identity.setdefault(s.identity, []).append(s)
    
    identities = list(by_identity.keys())
    if len(identities) >= 8:
        rng = random.Random(seed)
        rng.shuffle(identities)
        n = len(identities)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_ids = set(identities[:n_train])
        val_ids = set(identities[n_train:n_train + n_val])
        
        train_set, val_set, test_set = [], [], []
        for identity, items in by_identity.items():
            bucket = train_set if identity in train_ids else (val_set if identity in val_ids else test_set)
            bucket.extend(items)
        
        # Check if splits are valid
        val_labels = [1 if s.label == "fake" else 0 for s in val_set]
        test_labels = [1 if s.label == "fake" else 0 for s in test_set]
        if (len(set(val_labels)) > 1 and len(set(test_labels)) > 1 and
            len(train_set) > 0 and len(val_set) > 0 and len(test_set) > 0):
            return train_set, val_set, test_set
    
    # Fallback: sample-level stratified split
    print("  Using fallback: sample-level stratified split")
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio - val_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(np.zeros(len(samples)), labels))
    train_val_samples = [samples[i] for i in train_val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    train_val_labels = [1 if s.label == "fake" else 0 for s in train_val_samples]
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=seed)
    train_idx, val_idx = next(sss2.split(np.zeros(len(train_val_samples)), train_val_labels))
    
    train_samples = [train_val_samples[i] for i in train_idx]
    val_samples = [train_val_samples[i] for i in val_idx]
    
    return train_samples, val_samples, test_samples


def make_loader(entries, frames_per_clip, transform, train: bool, batch_size: int, num_workers: int, seed: int):
    ds = ClipDataset(entries, frames_per_clip=frames_per_clip, transform=transform, train=train, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=False), ds


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, pos_weight: torch.Tensor):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss, n_batches = 0.0, 0
    y_true, y_pred, y_scores = [], [], []
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = loss_fn(logits, labels.float())
        total_loss += loss.item()
        n_batches += 1
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_scores.extend(probs.tolist())
        y_pred.extend((probs >= 0.5).astype(int).tolist())
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, y_true, y_pred, y_scores


def compute_pos_weight(samples, device: torch.device) -> torch.Tensor:
    counts = datasets.class_balance(samples)
    n_real, n_fake = counts.get("real", 0), counts.get("fake", 0)
    ratio = (n_real / n_fake) if n_fake > 0 else 1.0
    return torch.tensor(ratio, dtype=torch.float32, device=device)


def train_with_early_stopping(train_entries, val_entries, args, device, label: str = ""):
    prefix = f"[{label}] " if label else ""
    
    train_loader, _ = make_loader(
        train_entries, args.frames_per_clip, train_augmentations,
        train=True, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed
    )
    val_loader, _ = make_loader(
        val_entries, args.frames_per_clip, inference_transform,
        train=False, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed
    )

    train_samples = [e.sample for e in train_entries]
    pos_weight = compute_pos_weight(train_samples, device)

    model = TemporalDeepfakeClassifier(pretrained=True).to(device)
    optimizer = build_optimizer(model, lr=args.lr)

    best_val_loss = float("inf")
    best_state, best_epoch = None, 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            loss = train_step(model, (clips, labels), optimizer, pos_weight)
            train_losses.append(loss)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_loss, y_true, y_pred, y_scores = evaluate_loader(model, val_loader, device, pos_weight)
        val_metrics = evaluation.evaluate(y_true, y_pred, y_scores)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        })
        print(f"{prefix}epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']:.3f}  val_f1={val_metrics['f1']:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"{prefix}early stopping at epoch {epoch}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = len(history)

    model.load_state_dict(best_state)
    return model, best_epoch, history


# ============================================================================
# 8 VISUALIZATIONS
# ============================================================================

def visualize_1_training_history(history, out_path: str) -> None:
    """Visual 1: Training and validation loss curves with metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], 'b-o', label="Train Loss", linewidth=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], 'r-o', label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy & F1
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], 'g-o', label="Accuracy", linewidth=2)
    axes[1].plot(epochs, [h["val_f1"] for h in history], 'm-o', label="F1-Score", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Precision & Recall
    axes[2].plot(epochs, [h["val_precision"] for h in history], 'c-o', label="Precision", linewidth=2)
    axes[2].plot(epochs, [h["val_recall"] for h in history], 'y-o', label="Recall", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Precision & Recall")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    fig.suptitle("Figure 1: Training Progress Over Epochs", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_2_confusion_matrix(cm, out_path: str) -> None:
    """Visual 2: Confusion matrix with percentages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    cm = np.array(cm)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    im1 = ax1.imshow(cm, cmap="Blues", aspect='auto')
    labels = ["Real", "Fake"]
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix (Counts)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Percentages
    im2 = ax2.imshow(cm_percent, cmap="Reds", aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labels)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix (%)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, f"{cm_percent[i, j]:.1f}%", ha="center", va="center",
                     color="white" if cm_percent[i, j] > 50 else "black", fontsize=14)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    fig.suptitle("Figure 2: Confusion Matrix Analysis", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_3_roc_curve(fpr, tpr, auc, out_path: str) -> None:
    """Visual 3: ROC curve with AUC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    
    if fpr and tpr:
        ax.plot(fpr, tpr, 'b-', linewidth=3, label=f"AUC = {auc:.4f}")
        # Fill area under curve
        ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Figure 3: ROC Curve", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_4_precision_recall_curve(y_true, y_scores, out_path: str) -> None:
    """Visual 4: Precision-Recall curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, 'g-', linewidth=3, label=f"AP = {ap_score:.4f}")
    ax.fill_between(recall, precision, alpha=0.2, color='green')
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Figure 4: Precision-Recall Curve", fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_5_performance_metrics(metrics, out_path: str) -> None:
    """Visual 5: Bar chart of all performance metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
              metrics['f1'], metrics['auc_roc']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Figure 5: Overall Performance Metrics", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_6_class_distribution(train_s, val_s, test_s, out_path: str) -> None:
    """Visual 6: Class distribution across splits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ['Train', 'Validation', 'Test']
    real_counts = [
        datasets.class_balance(train_s)['real'],
        datasets.class_balance(val_s)['real'],
        datasets.class_balance(test_s)['real']
    ]
    fake_counts = [
        datasets.class_balance(train_s)['fake'],
        datasets.class_balance(val_s)['fake'],
        datasets.class_balance(test_s)['fake']
    ]
    
    x = np.arange(len(splits))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, real_counts, width, label='Real', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, fake_counts, width, label='Fake', color='#e74c3c', edgecolor='black')
    
    # Add labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel("Dataset Split", fontsize=12)
    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title("Figure 6: Class Distribution Across Splits", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_7_sample_predictions(entries, y_true, y_pred, y_scores, out_path: str) -> None:
    """Visual 7: Grid of sample predictions with confidence scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    n = min(8, len(entries))
    indices = np.random.choice(len(entries), n, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        entry = entries[idx]
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        score = y_scores[idx]
        
        # Load middle frame
        frame_path = entry.frame_paths[len(entry.frame_paths) // 2]
        img = Image.open(frame_path)
        
        ax = axes[i]
        ax.imshow(img)
        
        # Color coding
        if pred_label == true_label:
            color = '#27ae60' if true_label == 1 else '#2980b9'
            status = "✓ Correct"
        else:
            color = '#c0392b'
            status = "✗ Misclassified"
        
        true_label_str = "Fake" if true_label else "Real"
        pred_label_str = "Fake" if pred_label else "Real"
        
        title = f"{status}\nTrue: {true_label_str} | Pred: {pred_label_str}\nConfidence: {score:.3f}"
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        ax.axis("off")
    
    # Hide empty subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis("off")
    
    fig.suptitle("Figure 7: Sample Predictions with Confidence Scores", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def visualize_8_gradcam(model, test_entries, y_true, y_pred, output_dir, device, n_examples: int = 4) -> List:
    """Visual 8: Grad-CAM heatmaps showing model attention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Select examples: correct fakes, correct reals, misclassifications
    buckets = {"correct_fake": [], "correct_real": [], "misclassified": []}
    for entry, yt, yp in zip(test_entries, y_true, y_pred):
        if yt == yp:
            buckets["correct_fake" if yt == 1 else "correct_real"].append(entry)
        else:
            buckets["misclassified"].append(entry)
    
    # Pick diverse examples
    picks = []
    for key in ["correct_fake", "correct_real", "misclassified"]:
        if buckets[key]:
            picks.append((key, buckets[key][0]))
    # Add more from correct predictions
    for key in ["correct_fake", "correct_real"]:
        if len(picks) < n_examples and len(buckets[key]) > 1:
            picks.extend([(key, e) for e in buckets[key][1:4]])
    picks = picks[:n_examples]
    
    if not picks:
        print("  No examples for Grad-CAM visualization")
        return []
    
    try:
        cam = build_gradcam(model)
    except Exception as exc:
        print(f"  Grad-CAM build failed: {exc}")
        return []
    
    results = []
    n_cols = min(3, len(picks))
    n_rows = (len(picks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, (kind, entry) in enumerate(picks):
        try:
            frame_path = entry.frame_paths[len(entry.frame_paths) // 2]
            
            # Load and prepare image
            img = Image.open(frame_path).convert("RGB")
            rgb_float = np.asarray(img).astype(np.float32) / 255.0
            tensor = inference_transform(np.asarray(img)).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(device)
            tensor.requires_grad_(True)
            
            # Get Grad-CAM
            grayscale_cam = cam(input_tensor=tensor)[0]
            
            # Overlay
            from pytorch_grad_cam.utils.image import show_cam_on_image
            overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
            
            ax = axes[i]
            ax.imshow(overlay)
            
            # Add label
            label = entry.sample.label
            method = entry.sample.manipulation_method
            kind_label = kind.replace('_', ' ')
            ax.set_title(f"{kind_label}\nLabel: {label} | Method: {method}", fontsize=10)
            ax.axis("off")
            
            # Save individual for report
            ind_path = os.path.join(output_dir, f"gradcam_{i+1}_{kind}.png")
            Image.fromarray((overlay * 255).astype(np.uint8)).save(ind_path)
            results.append((ind_path, kind, entry))
            
        except Exception as exc:
            print(f"  Grad-CAM example {i+1} failed: {exc}")
            if i < len(axes):
                axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(len(picks), len(axes)):
        axes[i].axis("off")
    
    fig.suptitle("Figure 8: Grad-CAM Attention Maps", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gradcam_grid.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return results


# ============================================================================
# 4 TABLES
# ============================================================================

def generate_table_1_dataset_summary(samples, train_s, val_s, test_s) -> str:
    """Table 1: Dataset composition summary."""
    lines = []
    lines.append("### Table 1: Dataset Composition Summary")
    lines.append("")
    lines.append("| Split | Real Videos | Fake Videos | Total |")
    lines.append("|-------|-------------|-------------|-------|")
    
    for name, bucket in [("Train", train_s), ("Validation", val_s), ("Test", test_s)]:
        counts = datasets.class_balance(bucket)
        lines.append(f"| {name} | {counts['real']} | {counts['fake']} | {len(bucket)} |")
    
    total_counts = datasets.class_balance(samples)
    lines.append(f"| **Total** | **{total_counts['real']}** | **{total_counts['fake']}** | **{len(samples)}** |")
    
    # Add collection breakdown
    lines.append("")
    lines.append("**Collection Breakdown:**")
    for collection, counts in datasets.collection_summary(samples).items():
        methods = [f"{m}: {c}" for m, c in counts.items() if m != "real"]
        lines.append(f"- {collection}: {counts.get('real', 0)} real, {', '.join(methods) if methods else 'no fake'}")
    
    return "\n".join(lines)


def generate_table_2_performance_metrics(metrics) -> str:
    """Table 2: Detailed performance metrics."""
    lines = []
    lines.append("### Table 2: Classification Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")
    lines.append(f"| **Accuracy** | {metrics['accuracy']:.4f} | Overall correctness |")
    lines.append(f"| **Precision** | {metrics['precision']:.4f} | When predicting fake, how often correct |")
    lines.append(f"| **Recall** | {metrics['recall']:.4f} | Ability to find all fake videos |")
    lines.append(f"| **F1-Score** | {metrics['f1']:.4f} | Harmonic mean of precision & recall |")
    lines.append(f"| **AUC-ROC** | {metrics['auc_roc']:.4f} | Ability to distinguish classes |")
    
    cm = metrics['confusion_matrix']
    lines.append("")
    lines.append("**Confusion Matrix:**")
    lines.append(f"- True Real: {cm[0][0]}")
    lines.append(f"- False Fake: {cm[0][1]} (False Positives)")
    lines.append(f"- False Real: {cm[1][0]} (False Negatives)")
    lines.append(f"- True Fake: {cm[1][1]}")
    
    return "\n".join(lines)


def generate_table_3_classification_report(y_true, y_pred) -> str:
    """Table 3: Per-class classification report."""
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True)
    
    lines = []
    lines.append("### Table 3: Per-Class Classification Report")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1-Score | Support |")
    lines.append("|-------|-----------|--------|----------|---------|")
    
    for class_name in ['Real', 'Fake']:
        stats = report[class_name.lower()]
        lines.append(f"| {class_name} | {stats['precision']:.4f} | {stats['recall']:.4f} | "
                     f"{stats['f1-score']:.4f} | {stats['support']} |")
    
    lines.append(f"| **Macro Avg** | {report['macro avg']['precision']:.4f} | "
                 f"{report['macro avg']['recall']:.4f} | {report['macro avg']['f1-score']:.4f} | - |")
    lines.append(f"| **Weighted Avg** | {report['weighted avg']['precision']:.4f} | "
                 f"{report['weighted avg']['recall']:.4f} | {report['weighted avg']['f1-score']:.4f} | - |")
    
    return "\n".join(lines)


def generate_table_4_cross_validation_stats(history, best_epoch) -> str:
    """Table 4: Training and validation statistics."""
    lines = []
    lines.append("### Table 4: Training and Validation Statistics")
    lines.append("")
    lines.append("| Metric | Best Value | Epoch |")
    lines.append("|--------|------------|-------|")
    
    # Find best values
    best_val_acc = max(history, key=lambda x: x['val_accuracy'])
    best_val_f1 = max(history, key=lambda x: x['val_f1'])
    best_val_loss = min(history, key=lambda x: x['val_loss'])
    
    lines.append(f"| Validation Accuracy | {best_val_acc['val_accuracy']:.4f} | {best_val_acc['epoch']} |")
    lines.append(f"| Validation F1-Score | {best_val_f1['val_f1']:.4f} | {best_val_f1['epoch']} |")
    lines.append(f"| Validation Loss | {best_val_loss['val_loss']:.4f} | {best_val_loss['epoch']} |")
    lines.append(f"| **Best Model Epoch** | - | **{best_epoch}** |")
    
    # Training summary
    lines.append("")
    lines.append("**Training Summary:**")
    lines.append(f"- Total epochs trained: {len(history)}")
    lines.append(f"- Early stopping at epoch: {best_epoch}")
    lines.append(f"- Initial learning rate: {args.lr}")
    lines.append(f"- Batch size: {args.batch_size}")
    lines.append(f"- Frames per clip: {args.frames_per_clip}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global args
    args = parse_args()
    
    if args.smoke_test:
        args.max_videos = min(args.max_videos, 20)
        args.epochs = min(args.epochs, 3)
        args.frames_per_clip = min(args.frames_per_clip, 4)
        args.max_frames_per_video = min(args.max_frames_per_video, 6)
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    
    print("="*80)
    print("DEEPFAKE DETECTION - COMPLETE TRAINING PIPELINE")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Max videos: {args.max_videos}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)

    # Load samples
    collections = tuple(c.strip() for c in args.collections.split(",") if c.strip())
    print(f"\n[1/6] Loading samples from {args.root_dir}...")
    samples = datasets.load_faceforensics(root_dir=args.root_dir, collections=collections)
    
    if not samples:
        print("ERROR: No videos found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Found {len(samples)} videos total")
    for collection, counts in datasets.collection_summary(samples).items():
        print(f"    {collection}: {counts}")

    # Sample down
    if args.max_videos and args.max_videos > 0 and len(samples) > args.max_videos:
        samples = datasets.sample_for_quick_run(samples, max_total=args.max_videos, seed=args.seed)
        print(f"  Sampled down to {len(samples)} videos")
        for collection, counts in datasets.collection_summary(samples).items():
            print(f"    {collection}: {counts}")

    # Build cache
    print(f"\n[2/6] Building face cache (slow step, runs once)...")
    entries = build_cache_for_samples(
        samples, cache_root=args.cache_dir,
        max_frames_per_video=args.max_frames_per_video,
        progress=True
    )
    entries_by_path = {e.sample.path: e for e in entries}
    samples_cached = [e.sample for e in entries]
    print(f"  Cached {len(samples_cached)}/{len(samples)} videos")

    # Split
    print(f"\n[3/6] Splitting data...")
    train_s, val_s, test_s = balanced_video_split(samples_cached, seed=args.seed)
    
    for name, bucket in [("Train", train_s), ("Validation", val_s), ("Test", test_s)]:
        counts = datasets.class_balance(bucket)
        print(f"  {name}: {len(bucket)} clips ({counts})")
    
    train_entries = [entries_by_path[s.path] for s in train_s if s.path in entries_by_path]
    val_entries = [entries_by_path[s.path] for s in val_s if s.path in entries_by_path]
    test_entries = [entries_by_path[s.path] for s in test_s if s.path in entries_by_path]

    # Train
    device = get_device()
    print(f"\n[4/6] Training model on {device}...")
    
    model, best_epoch, history = train_with_early_stopping(
        train_entries, val_entries, args, device, label="main"
    )
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

    # Evaluate
    print(f"\n[5/6] Evaluating model...")
    test_loader, _ = make_loader(
        test_entries, args.frames_per_clip, inference_transform,
        train=False, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed
    )
    pos_weight = compute_pos_weight(train_s, device)
    _, y_true, y_pred, y_scores = evaluate_loader(model, test_loader, device, pos_weight)
    test_metrics = evaluation.evaluate(y_true, y_pred, y_scores)
    roc_points = evaluation.roc_curve_points(y_true, y_scores)
    
    print(f"\n  Test Results:")
    print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:    {test_metrics['recall']:.4f}")
    print(f"    F1-Score:  {test_metrics['f1']:.4f}")
    print(f"    AUC-ROC:   {test_metrics['auc_roc']:.4f}")

    # Generate 8 Visualizations
    print(f"\n[6/6] Generating 8 visualizations and 4 tables...")
    
    vis_paths = {}
    
    # Visual 1: Training History
    print("  Generating Visual 1: Training History...")
    visualize_1_training_history(history, os.path.join(args.output_dir, "figure1_training_history.png"))
    vis_paths['figure1'] = 'figure1_training_history.png'
    
    # Visual 2: Confusion Matrix
    print("  Generating Visual 2: Confusion Matrix...")
    visualize_2_confusion_matrix(test_metrics['confusion_matrix'], 
                                os.path.join(args.output_dir, "figure2_confusion_matrix.png"))
    vis_paths['figure2'] = 'figure2_confusion_matrix.png'
    
    # Visual 3: ROC Curve
    print("  Generating Visual 3: ROC Curve...")
    visualize_3_roc_curve(roc_points['fpr'], roc_points['tpr'], test_metrics['auc_roc'],
                          os.path.join(args.output_dir, "figure3_roc_curve.png"))
    vis_paths['figure3'] = 'figure3_roc_curve.png'
    
    # Visual 4: Precision-Recall Curve
    print("  Generating Visual 4: Precision-Recall Curve...")
    visualize_4_precision_recall_curve(y_true, y_scores,
                                       os.path.join(args.output_dir, "figure4_precision_recall.png"))
    vis_paths['figure4'] = 'figure4_precision_recall.png'
    
    # Visual 5: Performance Metrics Bar Chart
    print("  Generating Visual 5: Performance Metrics...")
    visualize_5_performance_metrics(test_metrics,
                                    os.path.join(args.output_dir, "figure5_performance_metrics.png"))
    vis_paths['figure5'] = 'figure5_performance_metrics.png'
    
    # Visual 6: Class Distribution
    print("  Generating Visual 6: Class Distribution...")
    visualize_6_class_distribution(train_s, val_s, test_s,
                                   os.path.join(args.output_dir, "figure6_class_distribution.png"))
    vis_paths['figure6'] = 'figure6_class_distribution.png'
    
    # Visual 7: Sample Predictions
    print("  Generating Visual 7: Sample Predictions...")
    visualize_7_sample_predictions(test_entries, y_true, y_pred, y_scores,
                                   os.path.join(args.output_dir, "figure7_sample_predictions.png"))
    vis_paths['figure7'] = 'figure7_sample_predictions.png'
    
    # Visual 8: Grad-CAM
    print("  Generating Visual 8: Grad-CAM Attention Maps...")
    gradcam_results = visualize_8_gradcam(
        model, test_entries, y_true, y_pred,
        args.output_dir, device, args.gradcam_examples
    )
    vis_paths['figure8'] = 'gradcam_grid.png'
    
    # Generate 4 Tables
    print("\n  Generating Tables...")
    
    tables = {}
    tables['table1'] = generate_table_1_dataset_summary(samples_cached, train_s, val_s, test_s)
    tables['table2'] = generate_table_2_performance_metrics(test_metrics)
    tables['table3'] = generate_table_3_classification_report(y_true, y_pred)
    tables['table4'] = generate_table_4_cross_validation_stats(history, best_epoch)

    # Save all tables to a single markdown file
    with open(os.path.join(args.output_dir, "tables.md"), "w") as f:
        f.write("# Experimental Results - Tables\n\n")
        for table_name, table_content in tables.items():
            f.write(table_content)
            f.write("\n\n")
            f.write("---\n\n")
    
    print("  Tables saved to tables.md")

    # Save metrics
    results = {
        "args": vars(args),
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "history": history,
        "dataset_counts": {
            "train": datasets.class_balance(train_s),
            "val": datasets.class_balance(val_s),
            "test": datasets.class_balance(test_s),
        },
        "visualizations": vis_paths,
    }
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE! Results saved to:", args.output_dir)
    print("="*80)
    print("\n8 Visualizations:")
    for i in range(1, 9):
        print(f"  Figure {i}: {vis_paths.get(f'figure{i}', 'N/A')}")
    print("\n4 Tables saved in: tables.md")
    print("\nModel weights: best_model.pt")
    print("Full metrics: metrics.json")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERROR:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)