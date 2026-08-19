"""
Stage 8 -- Forensic report generation.

Compiles everything the earlier stages produced -- verdict, confidence,
Grad-CAM heatmap, original file hash, metadata, timestamp -- into a
single structured PDF designed to be reviewed by a forensic examiner,
not just consumed by another program (Section 3.7 of the methodology).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


@dataclass
class ReportInputs:
    file_path: str
    sha256: str
    captured_at: str
    media_type: str
    verdict: str
    confidence: float
    exif: dict
    ela_mean_error: Optional[float]
    heatmap_path: Optional[str]  # PNG written by Stage 7's overlay, if available


def generate_forensic_report(data: ReportInputs, output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=LETTER,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Deepfake Detection - Forensic Report", styles["Title"]))
    story.append(Spacer(1, 12))

    summary_rows = [
        ["File", os.path.basename(data.file_path)],
        ["Media type", data.media_type],
        ["SHA-256 (original file)", data.sha256],
        ["Processed at (UTC)", data.captured_at],
        ["Verdict", data.verdict],
        ["Confidence (fake probability)", f"{data.confidence:.3f}"],
    ]
    if data.ela_mean_error is not None:
        summary_rows.append(["ELA mean error", f"{data.ela_mean_error:.3f}"])

    table = Table(summary_rows, colWidths=[2.2 * inch, 3.8 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    if data.exif:
        story.append(Paragraph("Embedded metadata (EXIF)", styles["Heading2"]))
        exif_rows = [[str(k), str(v)] for k, v in list(data.exif.items())[:20]]
        exif_table = Table(exif_rows, colWidths=[2.2 * inch, 3.8 * inch])
        exif_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(exif_table)
    else:
        story.append(Paragraph("No embedded EXIF metadata found "
                                "(absence of expected metadata can itself be a forensic signal).",
                                styles["Normal"]))
    story.append(Spacer(1, 16))

    if data.heatmap_path and os.path.exists(data.heatmap_path):
        story.append(Paragraph("Grad-CAM explanation", styles["Heading2"]))
        story.append(Paragraph(
            "Highlighted regions indicate where the classifier's attention "
            "was concentrated when producing the verdict above.", styles["Normal"]))
        story.append(Spacer(1, 8))
        story.append(RLImage(data.heatmap_path, width=3 * inch, height=3 * inch))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "This report documents an automated analysis and is intended to support, "
        "not replace, review by a qualified forensic examiner.", styles["Italic"]))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    data = ReportInputs(
        file_path="sample.mp4",
        sha256="a" * 64,
        captured_at="2026-07-11T12:00:00Z",
        media_type="video",
        verdict="likely manipulated",
        confidence=0.87,
        exif={},
        ela_mean_error=None,
        heatmap_path=None,
    )
    out = generate_forensic_report(data, "sample_report.pdf")
    print(f"wrote {out}")
