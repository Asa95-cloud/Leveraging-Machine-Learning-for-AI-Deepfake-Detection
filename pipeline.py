"""
pipeline.py -- End-to-end orchestration of all eight stages for a
single uploaded file. This is the runnable counterpart to Table 1 and
Section 3 of the accompanying research paper.

Usage:
    python pipeline.py /path/to/media_file.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from stage1_upload import validate_media
from stage2_preprocessing import preprocess, inference_transform
from stage3_forensic_metadata import capture_forensic_record
from stage4_face_detection import detect_and_align
from stage6_classification import DeepfakeClassifier, TemporalDeepfakeClassifier
from stage7_explainability import build_gradcam, explain_prediction, aggregate_video_confidence
from stage8_report import ReportInputs, generate_forensic_report
from device import get_device


def run_pipeline(file_path: str, output_dir: str = "output", weights_path: str | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # --- Stage 1: upload & validation -------------------------------------
    upload = validate_media(file_path)
    if not upload.valid:
        raise ValueError(f"Stage 1 rejected the file: {upload.reason}")
    print(f"[1/8] validated {upload.media_type} ({upload.mime_type}, {upload.size_mb:.2f} MB)")

    # --- Stage 2: preprocessing --------------------------------------------
    frames = preprocess(upload.media_type, file_path)
    print(f"[2/8] extracted {len(frames)} raw frame(s)")

    # --- Stage 3: forensic metadata capture --------------------------------
    forensic_record = capture_forensic_record(file_path, is_image=(upload.media_type == "image"))
    print(f"[3/8] sha256={forensic_record.sha256[:16]}...  exif_fields={len(forensic_record.exif)}")

    # --- Stage 4: face detection & alignment -------------------------------
    faces = detect_and_align(frames)
    if not faces:
        raise RuntimeError("Stage 4 found no usable face in the input; cannot continue.")
    print(f"[4/8] {len(faces)}/{len(frames)} frame(s) yielded a usable aligned face")

    # --- Stage 5 & 6: feature extraction + classification -------------------
    device = get_device()
    model = DeepfakeClassifier(pretrained=weights_path is None)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.to(device)
    model.eval()

    frame_logits, det_confidences, last_input_tensor, last_rgb_float = [], [], None, None
    with torch.no_grad():
        for face in faces:
            tensor = inference_transform(face.aligned_face).unsqueeze(0).to(device)
            logit, _ = model(tensor)
            frame_logits.append(logit.item())
            det_confidences.append(face.confidence)
            last_input_tensor, last_rgb_float = tensor, face.aligned_face.astype(np.float32) / 255.0
    print(f"[5-6/8] scored {len(frame_logits)} frame(s) with the Xception classifier on {device}")

    # --- Stage 7: explainability & confidence aggregation --------------------
    confidence, verdict = aggregate_video_confidence(frame_logits, det_confidences)
    heatmap_path = None
    try:
        cam = build_gradcam(model)
        last_input_tensor.requires_grad_(True)
        overlay = explain_prediction(cam, last_input_tensor, last_rgb_float)
        heatmap_path = os.path.join(output_dir, "gradcam_overlay.png")
        from PIL import Image
        Image.fromarray((overlay * 255).astype(np.uint8)).save(heatmap_path)
    except Exception as exc:  # noqa: BLE001
        print(f"    (Grad-CAM overlay skipped: {exc})")
    print(f"[7/8] verdict='{verdict}', confidence={confidence:.3f}")

    # --- Stage 8: forensic report -------------------------------------------
    report_inputs = ReportInputs(
        file_path=file_path,
        sha256=forensic_record.sha256,
        captured_at=forensic_record.captured_at,
        media_type=upload.media_type,
        verdict=verdict,
        confidence=confidence,
        exif=forensic_record.exif,
        ela_mean_error=forensic_record.ela_mean_error,
        heatmap_path=heatmap_path,
    )
    report_path = os.path.join(output_dir, "forensic_report.pdf")
    generate_forensic_report(report_inputs, report_path)
    print(f"[8/8] wrote forensic report -> {report_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Run the 8-stage deepfake detection pipeline.")
    parser.add_argument("file_path", help="path to an image or video file")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--weights", default=None, help="path to fine-tuned model weights (.pt)")
    args = parser.parse_args()

    try:
        run_pipeline(args.file_path, args.output_dir, args.weights)
    except Exception as exc:  # noqa: BLE001
        print(f"pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
