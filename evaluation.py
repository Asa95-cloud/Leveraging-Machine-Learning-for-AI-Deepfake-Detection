"""
Evaluation metrics (Section 3.11 of the methodology).

Accuracy alone is a poor summary for a forensic tool: a false positive
(flagging authentic evidence as fake) and a false negative (missing
real manipulation) carry different, serious consequences. Precision,
recall, F1, AUC-ROC, and the confusion matrix are reported alongside
accuracy for exactly that reason.
"""

from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def evaluate(y_true: List[int], y_pred: List[int], y_scores: List[float]) -> Dict:
    """y_true/y_pred are 0/1 (0 = real, 1 = fake); y_scores are the
    continuous fake-probability outputs used for AUC-ROC."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def cross_manipulation_evaluate(results_by_held_out_method: Dict[str, Dict]) -> Dict:
    """Section 3.4/3.11: for each manipulation method in scope, the model
    is trained on the other methods plus pristine footage (via
    `datasets.leave_one_method_out_split`) and evaluated on the held-out
    method.

    Expects **raw predictions**, not pre-computed metrics:
        {"Deepfakes": {"y_true": [...], "y_pred": [...], "y_scores": [...]},
         "DeepFakeDetection": {...}}

    A common mistake is to call `evaluate(...)` yourself for each held-out
    method first and pass the *resulting metrics dict* (with keys like
    "accuracy", "f1", ...) in here -- this function calls `evaluate()`
    internally, so that would double-evaluate and fail with a confusing
    `TypeError: evaluate() got an unexpected keyword argument 'accuracy'`
    deep inside this function. To catch that early with a clear message,
    every value's keys are checked before any evaluation runs.
    """
    required = {"y_true", "y_pred", "y_scores"}
    for method, data in results_by_held_out_method.items():
        if not isinstance(data, dict) or not required.issubset(data.keys()):
            got = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
            raise ValueError(
                f"cross_manipulation_evaluate expects raw predictions for {method!r} -- a dict "
                f"with keys {sorted(required)} -- but got {got}. If you already called evaluate() "
                f"for this method, pass its y_true/y_pred/y_scores inputs here instead of its "
                f"output; don't evaluate twice."
            )
    return {method: evaluate(**data) for method, data in results_by_held_out_method.items()}


def summarize_generalisation_gap(in_distribution: Dict, cross_manipulation: Dict[str, Dict]) -> Dict:
    """Reports the accuracy/F1 drop between the in-distribution test split
    (same manipulation methods seen in training, Section 3.4's 70/15/15
    identity-stratified split) and each leave-one-method-out result --
    the generalisation gap the dissertation's Findings/Discussion
    chapters report on (Section 3.11)."""
    gaps = {}
    for method, metrics in cross_manipulation.items():
        gaps[method] = {
            "accuracy_drop": in_distribution["accuracy"] - metrics["accuracy"],
            "f1_drop": in_distribution["f1"] - metrics["f1"],
        }
    return gaps


def roc_curve_points(y_true: List[int], y_scores: List[float]) -> Dict:
    """FPR/TPR points for plotting the ROC curve (Figure 5), kept separate
    from `evaluate()` so its return shape -- and the offline test that
    checks it -- stays unchanged. Returns empty lists if `y_true` is
    single-class (ROC is undefined in that case)."""
    if len(set(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0]
    y_scores = [0.1, 0.55, 0.9, 0.8, 0.4, 0.2]
    print(evaluate(y_true, y_pred, y_scores))
