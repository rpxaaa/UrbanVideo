"""Category-specific key-frame selection strategies.

Each strategy maps a temporal region of interest to a concrete list of
frame indices.  Strategies are dispatched by question category and return
index lists — frame reading / encoding stays in video_utils.
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Callable

# ── Category → strategy dispatch ─────────────────────────────────────────
CATEGORY_SAMPLING_MODE: dict[str, str] = {
    "Trajectory Captioning": "head_tail_balance",
    "Start/End Position":    "head_tail_focus",
    "Progress Evaluation":   "uniform_tail_dense",
    "Action Generation":     "action_tail_heavy",
    "Landmark Position":     "landmark_focus",
    "Goal Detection":        "goal_detection",
    "High-level Planning":   "uniform_tail_dense",
    "Cognitive Map":         "uniform_tail_dense",
    "Association Reasoning": "goal_context",
    "Proximity":             "change_focus",
    "Causal":                "cause_effect",
    "Counterfactual":        "decision_focus",
    "Sequence Recall":       "sequence_focus",
    "Duration":              "duration_compare",
}

# BASELINE_CATEGORIES always use "uniform" — this is enforced in reasoner_node,
# not here, so that the dispatch table is the single source of truth.


# ── Strategy implementations ─────────────────────────────────────────────

def _uniform(total_frames: int, num_frames: int) -> list[int]:
    step = max(total_frames // num_frames, 1)
    indices = [i * step for i in range(num_frames) if i * step < total_frames]
    return indices


def _uniform_range(start: int, end: int, n: int) -> list[int]:
    """n indices evenly spaced in [start, end)."""
    if n <= 0 or end <= start:
        return []
    step = max((end - start) // n, 1)
    return [start + i * step for i in range(n) if start + i * step < end]


def _head_tail_balance(total_frames: int, num_frames: int) -> list[int]:
    """First 20% + mid 35-70% + last 20%."""
    head_end = int(total_frames * 0.20)
    mid_start = int(total_frames * 0.35)
    mid_end = int(total_frames * 0.70)
    tail_start = int(total_frames * 0.80)

    n_each = max(num_frames // 3, 1)
    head_n, mid_n, tail_n = n_each, n_each, n_each
    # Distribute remainder
    remainder = num_frames - n_each * 3
    if remainder >= 2:
        head_n += 1
        tail_n += 1
        remainder -= 2
    if remainder == 1:
        mid_n += 1

    head = _uniform_range(0, head_end, head_n)
    mid = _uniform_range(mid_start, mid_end, mid_n)
    tail = _uniform_range(tail_start, total_frames, tail_n)
    return sorted(set(head + mid + tail))


def _head_tail_focus(total_frames: int, num_frames: int) -> list[int]:
    """First 18% (40% budget) + mid 42-58% + last 82-100% (40% budget)."""
    tail_start = int(total_frames * 0.82)
    head_end = int(total_frames * 0.18)
    mid_start = int(total_frames * 0.42)
    mid_end = int(total_frames * 0.58)

    head_n = max(int(num_frames * 0.4), 2)
    tail_n = max(int(num_frames * 0.4), 2)
    mid_n = num_frames - head_n - tail_n
    if mid_n < 1:
        mid_n = 1
        head_n = max(int((num_frames - 1) * 0.5), 2)
        tail_n = num_frames - head_n - mid_n

    head = _uniform_range(0, head_end, head_n)
    mid = _uniform_range(mid_start, mid_end, mid_n)
    tail = _uniform_range(tail_start, total_frames, tail_n)
    return sorted(set(head + mid + tail))


def _uniform_tail_dense(total_frames: int, num_frames: int) -> list[int]:
    """Uniform 60% + last 70-100% dense (40% budget)."""
    tail_n = max(int(num_frames * 0.4), 2)
    uniform_n = num_frames - tail_n
    tail_start = int(total_frames * 0.70)

    uniform = _uniform_range(0, total_frames, uniform_n)
    tail = _uniform_range(tail_start, total_frames, tail_n)
    return sorted(set(uniform + tail))


def _action_tail_heavy(total_frames: int, num_frames: int) -> list[int]:
    """Uniform 50% + last 25% dense (50% budget) — for Action Generation.

    Action questions need heavy tail concentration to capture the current
    drone state while retaining enough uniform coverage to audit completed steps.
    """
    tail_n = max(int(num_frames * 0.50), 4)
    uniform_n = num_frames - tail_n
    tail_start = int(total_frames * 0.75)

    uniform = _uniform_range(0, total_frames, uniform_n)
    tail = _uniform_range(tail_start, total_frames, tail_n)
    return sorted(set(uniform + tail))


def _landmark_focus(total_frames: int, num_frames: int) -> list[int]:
    """15-40% + 55-100% — focus on mid-to-late frames where landmarks appear."""
    seg1_start = int(total_frames * 0.15)
    seg1_end = int(total_frames * 0.40)
    seg2_start = int(total_frames * 0.55)

    seg1_n = max(int(num_frames * 0.35), 2)
    seg2_n = num_frames - seg1_n

    seg1 = _uniform_range(seg1_start, seg1_end, seg1_n)
    seg2 = _uniform_range(seg2_start, total_frames, seg2_n)
    return sorted(set(seg1 + seg2))


def _goal_detection(total_frames: int, num_frames: int) -> list[int]:
    """0-30% dense (start view) + 30-70% sparse + 70-100% (approach).

    Goal Detection questions ask about visibility from the initial position,
    so the first 30% of frames must be well-covered — unlike landmark_focus
    which skips 0-15%.
    """
    start_end = int(total_frames * 0.30)
    mid_start = int(total_frames * 0.30)
    mid_end = int(total_frames * 0.70)
    tail_start = int(total_frames * 0.70)

    start_n = max(int(num_frames * 0.38), 6)   # ~6 frames for starting view
    tail_n = max(int(num_frames * 0.31), 5)     # ~5 frames for approach
    mid_n = num_frames - start_n - tail_n        # ~5 frames for mid-flight

    start_frames = _uniform_range(0, start_end, start_n)
    mid_frames = _uniform_range(mid_start, mid_end, mid_n)
    tail_frames = _uniform_range(tail_start, total_frames, tail_n)
    return sorted(set(start_frames + mid_frames + tail_frames))


def _goal_context(total_frames: int, num_frames: int) -> list[int]:
    """0-20% + 55-85% (50% budget) + 85-100%."""
    seg1_end = int(total_frames * 0.20)
    seg2_start = int(total_frames * 0.55)
    seg2_end = int(total_frames * 0.85)
    seg3_start = int(total_frames * 0.85)

    seg2_n = max(int(num_frames * 0.50), 2)
    seg1_n = max(int((num_frames - seg2_n) * 0.5), 1)
    seg3_n = num_frames - seg1_n - seg2_n

    seg1 = _uniform_range(0, seg1_end, seg1_n)
    seg2 = _uniform_range(seg2_start, seg2_end, seg2_n)
    seg3 = _uniform_range(seg3_start, total_frames, seg3_n)
    return sorted(set(seg1 + seg2 + seg3))


def _decision_focus(total_frames: int, num_frames: int) -> list[int]:
    """0-25% + 45-70% + 80-100% — covers pre-decision, decision, post-decision."""
    seg1_end = int(total_frames * 0.25)
    seg2_start = int(total_frames * 0.45)
    seg2_end = int(total_frames * 0.70)
    seg3_start = int(total_frames * 0.80)

    n_each = max(num_frames // 3, 1)
    seg1_n, seg2_n, seg3_n = n_each, n_each, n_each
    remainder = num_frames - n_each * 3
    if remainder >= 2:
        seg1_n += 1
        seg3_n += 1
        remainder -= 2
    if remainder == 1:
        seg2_n += 1

    seg1 = _uniform_range(0, seg1_end, seg1_n)
    seg2 = _uniform_range(seg2_start, seg2_end, seg2_n)
    seg3 = _uniform_range(seg3_start, total_frames, seg3_n)
    return sorted(set(seg1 + seg2 + seg3))


def _find_anchor_frame(
    video_path: str, total_frames: int, top_n: int = 1
) -> list[int]:
    """Use visual-change detection to find anchor frames (highest scene novelty).

    Samples 48 evenly-spaced frames, computes histogram correlation between
    consecutive frames, returns indices with the lowest correlation (most change).
    """
    if not video_path:
        return [total_frames // 2]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [total_frames // 2]

    probe_n = min(48, total_frames)
    step = max(total_frames // probe_n, 1)

    prev_hist = None
    change_scores: list[tuple[int, float]] = []

    for i in range(probe_n):
        idx = i * step
        if idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            novelty = 1.0 - (corr + 1.0) / 2.0  # corr∈[-1,1] → novelty∈[0,1]
            change_scores.append((idx, novelty))
        prev_hist = hist

    cap.release()

    if not change_scores:
        return [total_frames // 2]

    change_scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in change_scores[:top_n]]


def _change_focus(total_frames: int, num_frames: int,
                  video_path: str = "", **_kw) -> list[int]:
    """Anchor ±3 neighbourhood + uniform base."""
    anchors = _find_anchor_frame(video_path, total_frames, top_n=1)
    anchor = anchors[0] if anchors else total_frames // 2

    window = max(total_frames // 8, 3)
    anchor_start = max(0, anchor - window)
    anchor_end = min(total_frames, anchor + window + 1)

    anchor_n = max(int(num_frames * 0.45), 3)
    uniform_n = num_frames - anchor_n

    anchor_frames = _uniform_range(anchor_start, anchor_end, anchor_n)
    uniform_frames = _uniform(total_frames, uniform_n)
    return sorted(set(anchor_frames + uniform_frames))


def _sequence_focus(total_frames: int, num_frames: int) -> list[int]:
    """Dense early+mid coverage (50% budget in 0-65%) + uniform rest.

    Sequence questions ask "what happens NEXT after event X". The model needs
    dense early+mid frames to locate event X, then the remaining frames show
    what follows. Avoiding tail-heavy bias ensures the model doesn't confuse
    "what happened next" with "what happened at the end."
    """
    early_end = int(total_frames * 0.65)
    early_n = max(int(num_frames * 0.50), 6)
    late_n = num_frames - early_n

    early = _uniform_range(0, early_end, early_n)
    late = _uniform_range(early_end, total_frames, late_n)
    return sorted(set(early + late))


def _duration_compare(total_frames: int, num_frames: int) -> list[int]:
    """Balanced coverage across 4 quarters — for Duration comparison.

    Duration questions compare time lengths of different segments. Need equal
    coverage across the whole video so the model can count frames for each
    segment being compared. Avoids over-concentration in any one region.
    """
    q1_end = int(total_frames * 0.25)
    q2_end = int(total_frames * 0.50)
    q3_end = int(total_frames * 0.75)

    n_each = max(num_frames // 4, 1)
    q1 = _uniform_range(0, q1_end, n_each)
    q2 = _uniform_range(q1_end, q2_end, n_each)
    q3 = _uniform_range(q2_end, q3_end, n_each)
    q4 = _uniform_range(q3_end, total_frames, n_each)

    indices = sorted(set(q1 + q2 + q3 + q4))
    # Fill to budget if quarters under-filled
    if len(indices) < num_frames:
        fill = _uniform(total_frames, num_frames)
        existing = set(indices)
        for f in fill:
            if len(indices) >= num_frames:
                break
            if f not in existing:
                indices.append(f)
                existing.add(f)
    return sorted(indices)[:num_frames]


def _cause_effect(total_frames: int, num_frames: int,
                  video_path: str = "", **_kw) -> list[int]:
    """Two anchors (±4 / ±2 windows) + uniform fill."""
    anchors = _find_anchor_frame(video_path, total_frames, top_n=2)
    anchors.sort()  # temporal order

    if len(anchors) < 2:
        # Fallback: split video in thirds, sample mid-points
        anchors = [int(total_frames * 0.33), int(total_frames * 0.66)]

    win1 = max(total_frames // 12, 2)  # ±4 equivalent
    win2 = max(total_frames // 24, 1)  # ±2 equivalent

    a1_start = max(0, anchors[0] - win1)
    a1_end = min(total_frames, anchors[0] + win1 + 1)
    a2_start = max(0, anchors[1] - win2)
    a2_end = min(total_frames, anchors[1] + win2 + 1)

    anchor1_n = max(int(num_frames * 0.25), 2)
    anchor2_n = max(int(num_frames * 0.20), 2)
    uniform_n = num_frames - anchor1_n - anchor2_n

    a1 = _uniform_range(a1_start, a1_end, anchor1_n)
    a2 = _uniform_range(a2_start, a2_end, anchor2_n)
    uniform = _uniform(total_frames, uniform_n)
    return sorted(set(a1 + a2 + uniform))


# ── Strategy dispatch ────────────────────────────────────────────────────

_STRATEGIES: dict[str, Callable[..., list[int]]] = {
    "uniform":             _uniform,
    "head_tail_balance":   _head_tail_balance,
    "head_tail_focus":     _head_tail_focus,
    "uniform_tail_dense":  _uniform_tail_dense,
    "action_tail_heavy":   _action_tail_heavy,
    "landmark_focus":      _landmark_focus,
    "goal_detection":      _goal_detection,
    "goal_context":        _goal_context,
    "decision_focus":      _decision_focus,
    "change_focus":        _change_focus,
    "cause_effect":        _cause_effect,
    "sequence_focus":      _sequence_focus,
    "duration_compare":    _duration_compare,
}


def get_frame_indices(
    total_frames: int,
    num_frames: int,
    category: str = "",
    question: str = "",
    video_path: str = "",
) -> list[int]:
    """Return frame indices using the category-specific sampling strategy.

    Falls back to ``"uniform"`` for unknown categories or BASELINE_CATEGORIES.
    """
    mode = CATEGORY_SAMPLING_MODE.get(category, "uniform")
    strategy_fn = _STRATEGIES.get(mode, _uniform)

    try:
        indices = strategy_fn(total_frames, num_frames)
    except Exception:
        indices = _uniform(total_frames, num_frames)

    # Pass video_path for anchor-based strategies
    if mode in ("change_focus", "cause_effect") and video_path:
        try:
            indices = strategy_fn(
                total_frames, num_frames, video_path=video_path
            )
        except Exception:
            indices = _uniform(total_frames, num_frames)

    # Ensure exact budget
    if len(indices) > num_frames:
        step = max(len(indices) // num_frames, 1)
        indices = [indices[i] for i in range(0, len(indices), step)][:num_frames]
    elif len(indices) < num_frames:
        # Fill gaps with uniform interpolation
        existing = set(indices)
        uniform_candidates = _uniform(total_frames, num_frames)
        for c in uniform_candidates:
            if len(indices) >= num_frames:
                break
            if c not in existing:
                indices.append(c)
                existing.add(c)

    return sorted(indices)[:num_frames]
