"""
Subject-aware framing.

Decides which part of a landscape source (image still or video frame) should
stay in focus when it is cropped to a vertical 9:16 short-form frame.

Cropping a 16:9 source (e.g. 3840x2160) to 9:16 (1080x1920) keeps only the
center ~28% of the width, so a naive center-crop throws characters standing
left or right of center straight out of frame. This module finds the subject
and returns a normalized focus point so the crop window can be aimed at it.

Strategy (highest priority first):
  1. Faces / characters  -> Haar frontal + profile cascades.
  2. Visual saliency      -> spectral-residual attention map.
  3. Center               -> last-resort fallback.

`detect_focus_point` returns (fx, fy) in 0..1 source coordinates, where the
subject sits. Detection is comparatively expensive, so callers should run it
once per source (not once per output frame).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Cascades ship with opencv; load once at import.
_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
_SALIENCY = cv2.saliency.StaticSaliencySpectralResidual_create()

# Work at a reduced size for speed; detection scale is generous since we only
# need the centroid, not pixel-accurate boxes.
_DETECT_MAX_DIM = 960


def _downscale(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = min(1.0, _DETECT_MAX_DIM / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray, scale


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces via frontal + profile (and mirrored profile) cascades."""
    boxes: list[tuple[int, int, int, int]] = []
    params = dict(scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))

    for (x, y, w, h) in _FRONTAL.detectMultiScale(gray, **params):
        boxes.append((x, y, w, h))
    for (x, y, w, h) in _PROFILE.detectMultiScale(gray, **params):
        boxes.append((x, y, w, h))
    # Right-facing profiles: mirror the image, detect, flip x back.
    flipped = cv2.flip(gray, 1)
    gw = gray.shape[1]
    for (x, y, w, h) in _PROFILE.detectMultiScale(flipped, **params):
        boxes.append((gw - x - w, y, w, h))

    return boxes


def _focus_from_faces(boxes: list[tuple[int, int, int, int]], w: int, h: int) -> tuple[float, float] | None:
    if not boxes:
        return None
    # Area-weighted centroid: bigger / closer faces dominate the framing.
    total = 0.0
    sx = sy = 0.0
    for (x, y, bw, bh) in boxes:
        area = float(bw * bh)
        cx = x + bw / 2.0
        # Bias slightly toward the upper part of the face box so eyes/face
        # land in frame rather than the chin.
        cy = y + bh * 0.40
        sx += cx * area
        sy += cy * area
        total += area
    if total <= 0:
        return None
    return (sx / total / w, sy / total / h)


def _focus_from_saliency(gray: np.ndarray) -> tuple[float, float] | None:
    ok, smap = _SALIENCY.computeSaliency(gray)
    if not ok or smap is None:
        return None
    smap = (smap * 255).astype("uint8")
    # Keep the most salient mass; threshold then take its centroid.
    _, thresh = cv2.threshold(smap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    m = cv2.moments(thresh, binaryImage=True)
    h, w = gray.shape[:2]
    if m["m00"] <= 0:
        # Fall back to the single brightest saliency pixel.
        _, _, _, maxloc = cv2.minMaxLoc(smap)
        return (maxloc[0] / w, maxloc[1] / h)
    return (m["m10"] / m["m00"] / w, m["m01"] / m["m00"] / h)


def detect_focus_point(rgb: np.ndarray) -> tuple[float, float]:
    """
    Return the normalized (fx, fy) focus point (0..1) of `rgb` (H, W, 3).

    Faces win over saliency; saliency wins over center. The result is clamped
    to a safe interior margin so the crop window never has to clamp hard
    against an edge and lose the subject.
    """
    if rgb is None or rgb.size == 0:
        return (0.5, 0.5)

    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray, _ = _downscale(gray_full)
    gh, gw = gray.shape[:2]

    method = "center"
    focus: tuple[float, float] | None = None

    faces = _detect_faces(gray)
    if faces:
        focus = _focus_from_faces(faces, gw, gh)
        method = f"faces({len(faces)})"

    if focus is None:
        focus = _focus_from_saliency(gray)
        if focus is not None:
            method = "saliency"

    if focus is None:
        focus = (0.5, 0.5)

    fx = float(min(0.85, max(0.15, focus[0])))
    fy = float(min(0.80, max(0.20, focus[1])))
    logger.debug("focus=%s via %s", (round(fx, 3), round(fy, 3)), method)
    return (fx, fy)
