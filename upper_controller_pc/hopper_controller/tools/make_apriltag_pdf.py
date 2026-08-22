#!/usr/bin/env python3
"""Print tag36h11 AprilTags at a true millimetre size on A4.

Print at 100% / actual size. Do not 'fit to page'. After printing, the
black square must measure the stated millimetres with a ruler.

Usage:
  python3 tools/make_apriltag_pdf.py
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent / "apriltags_100mm_id1_id2_id3_a4.pdf"
FAMILY = "tag36h11"
TAG_MM = 100.0
QUIET_MM = 16.0  # white margin around the black square (>= 1 module)


def tag_image(tag_id: int, px: int = 1600) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    img = cv2.aruco.generateImageMarker(dictionary, int(tag_id), int(px))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Force a binary black square; OpenCV already fills the marker.
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img


def page(
    c: canvas.Canvas,
    *,
    tag_id: int,
    role: str,
    note: str,
    tag_mm: float = TAG_MM,
) -> None:
    page_w, page_h = A4
    img = tag_image(tag_id)
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    buf.seek(0)
    reader = ImageReader(buf)

    tag_pt = tag_mm * mm
    quiet_pt = QUIET_MM * mm
    cx = page_w / 2.0
    cy = page_h / 2.0 + 8 * mm
    x0 = cx - tag_pt / 2.0
    y0 = cy - tag_pt / 2.0

    # White quiet zone (required around AprilTags).
    c.setFillColorRGB(1, 1, 1)
    c.setStrokeColorRGB(0.75, 0.75, 0.75)
    c.setDash(2, 2)
    c.rect(
        x0 - quiet_pt, y0 - quiet_pt,
        tag_pt + 2 * quiet_pt, tag_pt + 2 * quiet_pt,
        fill=1, stroke=1,
    )
    c.setDash()
    c.drawImage(
        reader, x0, y0, width=tag_pt, height=tag_pt,
        preserveAspectRatio=True, mask="auto",
    )

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(cx, page_h - 18 * mm, f"{FAMILY}   ID {tag_id}")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(cx, page_h - 26 * mm, role)
    c.setFont("Helvetica", 11)
    c.drawCentredString(
        cx, page_h - 33 * mm,
        f"black square = {tag_mm:.0f} mm    print 100% / actual size",
    )

    # Dimension line under the tag.
    y_dim = y0 - quiet_pt - 8 * mm
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.6)
    c.line(x0, y_dim, x0 + tag_pt, y_dim)
    c.line(x0, y_dim - 2 * mm, x0, y_dim + 2 * mm)
    c.line(x0 + tag_pt, y_dim - 2 * mm, x0 + tag_pt, y_dim + 2 * mm)
    c.setFont("Helvetica", 10)
    c.drawCentredString(cx, y_dim - 6 * mm, f"{tag_mm:.0f} mm")

    c.setFont("Helvetica", 10)
    for i, line in enumerate(note.split("\n")):
        c.drawCentredString(cx, 22 * mm - i * 5 * mm, line)

    c.setFont("Helvetica", 8)
    c.drawCentredString(
        cx, 10 * mm,
        "Do not scale. Measure the black square after printing.",
    )
    c.showPage()


def main() -> None:
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("Hopper tag36h11 100 mm — ID 1 / 2 / 3")
    page(
        c, tag_id=1, role="BUTTON  (not the box)",
        note="Tape this on the wall for the button.\n"
             "Controller / perception: button = ID 1.",
    )
    page(
        c, tag_id=2, role="BOX LEFT  (not the button)",
        note="Box pair LEFT as you face the box.\n"
             "Software box size is 36 mm — this 100 mm page is ID only.\n"
             "For pushing, print box_apriltags_id2_id3_a4.pdf instead.",
    )
    page(
        c, tag_id=3, role="BOX RIGHT  (not the button)",
        note="Box pair RIGHT as you face the box.\n"
             "Software box size is 36 mm — this 100 mm page is ID only.\n"
             "For pushing, print box_apriltags_id2_id3_a4.pdf instead.",
    )
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
