"""Programmatic MemoryBuilder — computes structured spatial-temporal metadata
from video properties and frame indices. No LLM calls.

Produces 6 memory blocks matching the DESIGN_DOC architecture:
  1. scene_summary   — total frames, fps, duration, sampled span
  2. motion_summary  — early/mid/late frame indices + timestamps
  3. route_memory    — time-interval coverage (0-25%, 25-50%, 50-75%, 75-100%)
  4. landmark_summary— sampling strategy description + main evidence window
  5. progress_memory — neutral evidence-matching instruction (Progress Eval only)
  6. event_memory    — average inter-frame gap, dense-sampling regions
"""

from __future__ import annotations

import cv2
from typing import Optional


def _get_video_meta(video_path: str) -> dict:
    """Read video metadata: total_frames, fps, duration."""
    if not video_path:
        return {"total_frames": 0, "fps": 0.0, "duration": 0.0}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"total_frames": 0, "fps": 0.0, "duration": 0.0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    duration = total_frames / fps if fps > 0 else 0.0
    return {"total_frames": total_frames, "fps": fps, "duration": duration}


def _compute_visual_signals(
    video_path: str, frame_indices: list[int]
) -> dict:
    """Compute lightweight CV-based visual signals from sampled frames.

    Returns:
        dict with:
          - horizon_trend: sky ratio per frame (high=more sky=higher altitude)
          - edge_density_trend: edge complexity per frame (high=dense urban)
          - scene_changes: frames where visual content shifts significantly
          - summary: one-line natural-language summary of visual trends
    """
    import numpy as np

    if not video_path:
        return {"horizon_trend": [], "edge_density_trend": [],
                "scene_changes": [], "summary": "[No visual data]"}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"horizon_trend": [], "edge_density_trend": [],
                "scene_changes": [], "summary": "[No visual data]"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return {"horizon_trend": [], "edge_density_trend": [],
                "scene_changes": [], "summary": "[No visual data]"}

    # Resize target for consistent analysis
    analyze_w, analyze_h = 320, 240

    horizon_trend: list[tuple[int, float]] = []
    edge_density_trend: list[tuple[int, float]] = []
    prev_hist = None
    scene_changes: list[int] = []

    # ── Define sky detection: upper strips with low edge count ──
    num_strips = 10
    sky_strip_threshold = 0.02  # edge pixels / total pixels below this ≈ sky

    for idx in frame_indices:
        if idx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (analyze_w, analyze_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Edge density ──
        edges = cv2.Canny(gray, 50, 150)
        edge_pct = np.count_nonzero(edges) / (analyze_w * analyze_h)

        # ── Horizon / sky ratio ──
        strip_h = analyze_h // num_strips
        sky_strips = 0
        for s in range(num_strips):
            y0 = s * strip_h
            y1 = y0 + strip_h if s < num_strips - 1 else analyze_h
            strip_edges = edges[y0:y1, :]
            strip_edge_pct = np.count_nonzero(strip_edges) / (analyze_w * (y1 - y0))
            # High luminance + low edge count = sky indicator
            strip_mean = np.mean(gray[y0:y1, :])
            if strip_edge_pct < sky_strip_threshold and strip_mean > 100:
                sky_strips += 1
        sky_ratio = sky_strips / num_strips

        horizon_trend.append((idx, round(sky_ratio, 3)))
        edge_density_trend.append((idx, round(edge_pct, 4)))

        # ── Scene change detection (histogram correlation) ──
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < 0.7:  # significant visual change
                scene_changes.append(idx)
        prev_hist = hist

    cap.release()

    # ── Build summary ──
    if len(horizon_trend) >= 2:
        first_sky = horizon_trend[0][1]
        mid_sky = horizon_trend[len(horizon_trend) // 2][1]
        last_sky = horizon_trend[-1][1]
        sky_change = last_sky - first_sky
        if sky_change > 0.10:
            alt_trend = "rising (more sky visible → ascending)"
        elif sky_change < -0.10:
            alt_trend = "falling (less sky visible → descending)"
        else:
            alt_trend = "stable altitude"

        first_edge = edge_density_trend[0][1]
        last_edge = edge_density_trend[-1][1]
        if last_edge > first_edge * 1.3:
            env_trend = "entering denser area (more edges)"
        elif last_edge < first_edge * 0.7:
            env_trend = "entering more open area (fewer edges)"
        else:
            env_trend = "consistent environment density"

        scene_str = f"{len(scene_changes)} major scene change(s)" if scene_changes else "no major scene changes"
        summary = (
            f"Visual trend: {alt_trend} | {env_trend} | {scene_str}"
        )
    else:
        summary = "[Insufficient frames for visual trend analysis]"

    return {
        "horizon_trend": horizon_trend,
        "edge_density_trend": edge_density_trend,
        "scene_changes": scene_changes,
        "summary": summary,
    }


def _pct_str(frame_idx: int, total_frames: int) -> str:
    if total_frames == 0:
        return "0%"
    return f"{frame_idx / total_frames * 100:.0f}%"


def _time_str(frame_idx: int, fps: float) -> str:
    if fps <= 0:
        return "?s"
    t = frame_idx / fps
    return f"{t:.1f}s"


def build_memory(
    video_path: str,
    frame_indices: list[int],
    question_category: str,
    sampling_mode: str = "uniform",
    question: str = "",
) -> dict:
    """Build all 6 memory blocks and a formatted prompt-ready context string.

    Args:
        video_path: Path to the video file.
        frame_indices: Selected frame indices (0-based).
        question_category: Category name for progress_memory gating.
        sampling_mode: Strategy name (e.g. "uniform_tail_dense") for landmark_summary.
        question: Original question text (unused currently; reserved for future
                  keyword-aware memory augmentation).

    Returns:
        dict with keys matching GraphState.spatial_memory plus ``formatted_context``.
    """
    meta = _get_video_meta(video_path)
    total_frames = meta["total_frames"]
    fps = meta["fps"]
    num_frames = len(frame_indices)

    # ── CV-based visual signals (disabled — caused verifier regression) ──
    # Keeping function for future refinement. Sky-ratio unreliable in dense urban scenes.
    # visual_signals = _compute_visual_signals(video_path, frame_indices)
    visual_signals = {}

    if total_frames == 0 or num_frames == 0:
        return {
            "scene_summary": "[No video metadata]",
            "motion_summary": "",
            "route_memory": "",
            "landmark_summary": "",
            "progress_memory": "",
            "event_memory": "",
            "visual_signals": {},
            "formatted_context": "",
        }

    first_idx = frame_indices[0]
    last_idx = frame_indices[-1]
    mid_idx = frame_indices[num_frames // 2]

    # ── 1. scene_summary ──
    scene_summary = (
        f"Video: {total_frames} frames, {fps:.1f} fps, "
        f"{meta['duration']:.1f}s total | "
        f"Sampled: {num_frames} frames spanning frame {first_idx}–{last_idx} "
        f"({_time_str(first_idx, fps)} – {_time_str(last_idx, fps)})"
    )

    # ── 2. motion_summary ──
    early_block = frame_indices[: max(num_frames // 3, 1)]
    late_block = frame_indices[-max(num_frames // 3, 1):]
    motion_summary = (
        f"Early frames: {early_block[0]}–{early_block[-1]} "
        f"({_time_str(early_block[0], fps)} – {_time_str(early_block[-1], fps)}) | "
        f"Mid frame: {mid_idx} ({_time_str(mid_idx, fps)}) | "
        f"Late frames: {late_block[0]}–{late_block[-1]} "
        f"({_time_str(late_block[0], fps)} – {_time_str(late_block[-1], fps)})"
    )

    # ── 3. route_memory ──
    intervals = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}
    for idx in frame_indices:
        pct = idx / total_frames
        if pct < 0.25:
            intervals["0-25%"] += 1
        elif pct < 0.5:
            intervals["25-50%"] += 1
        elif pct < 0.75:
            intervals["50-75%"] += 1
        else:
            intervals["75-100%"] += 1
    route_parts = [f"{k}: {v}f" for k, v in intervals.items() if v > 0]
    route_memory = "Frame coverage: " + " | ".join(route_parts)

    # ── 4. landmark_summary ──
    recent_n = min(4, num_frames)
    recent_frames = frame_indices[-recent_n:]
    recent_str = ", ".join(
        f"#{idx} ({_time_str(idx, fps)})" for idx in recent_frames
    )
    landmark_summary = (
        f"Sampling strategy: {sampling_mode} | "
        f"Main evidence window (last {recent_n} frames): {recent_str}"
    )

    # ── 5. progress_memory ──
    if question_category == "Progress Evaluation":
        progress_memory = (
            "Progress determination: Compare the visual evidence across the sampled "
            "frames with each navigation instruction step. Determine which step's "
            "completion is best supported by the visible scene state, landmarks, and "
            "agent position. Do NOT infer progress from frame position or timestamp "
            "alone — the last frame does not necessarily mean the task is complete."
        )
    else:
        progress_memory = ""

    # ── 6. event_memory ──
    if num_frames >= 2:
        gaps = [
            frame_indices[i] - frame_indices[i - 1]
            for i in range(1, num_frames)
        ]
        avg_gap_frames = sum(gaps) / len(gaps)
        avg_gap_sec = avg_gap_frames / fps if fps > 0 else 0
    else:
        avg_gap_frames = 0
        avg_gap_sec = 0.0

    # Identify dense regions (consecutive frames with below-average gap)
    dense_regions = []
    if num_frames >= 3:
        region_start = frame_indices[0]
        for i in range(1, num_frames):
            gap = frame_indices[i] - frame_indices[i - 1]
            if gap > avg_gap_frames * 1.5:  # gap break
                if frame_indices[i - 1] - region_start >= avg_gap_frames:
                    dense_regions.append(f"{region_start}–{frame_indices[i-1]}")
                region_start = frame_indices[i]
        # last region
        if frame_indices[-1] - region_start >= avg_gap_frames:
            dense_regions.append(f"{region_start}–{frame_indices[-1]}")

    event_memory = (
        f"Avg inter-frame gap: {avg_gap_frames:.0f} frames ({avg_gap_sec:.1f}s)"
    )
    if dense_regions:
        event_memory += f" | Dense regions: {', '.join(dense_regions[:3])}"

    # ── Build formatted_context (prompt-ready injection) ──
    blocks = []
    if scene_summary:
        blocks.append(f"Scene: {scene_summary}")
    if motion_summary:
        blocks.append(f"Motion: {motion_summary}")
    if route_memory:
        blocks.append(f"Route: {route_memory}")
    if landmark_summary:
        blocks.append(f"Sampling: {landmark_summary}")
    if progress_memory:
        blocks.append(f"Progress: {progress_memory}")
    if event_memory:
        blocks.append(f"Events: {event_memory}")
    # CV visual signals disabled — caused verifier regression (see _compute_visual_signals)

    formatted_context = "[Spatial Memory Context]\n" + "\n".join(
        f"  {b}" for b in blocks
    ) if blocks else ""

    return {
        "scene_summary": scene_summary,
        "motion_summary": motion_summary,
        "route_memory": route_memory,
        "landmark_summary": landmark_summary,
        "progress_memory": progress_memory,
        "event_memory": event_memory,
        "visual_signals": visual_signals,
        "formatted_context": formatted_context,
    }


def format_memory_for_prompt(structured: dict) -> str:
    """Format structured memory dict into a concise prompt-ready string.

    Compatible with the old perception output format for verifier consumption.
    Also works with MemoryBuilder output.
    """
    if structured.get("formatted_context"):
        return structured["formatted_context"]

    # Fallback: build from individual fields (backward-compatible)
    parts = []
    if structured.get("scene_summary"):
        parts.append(f"Scene: {structured['scene_summary']}")
    if structured.get("motion_summary"):
        parts.append(f"Motion: {structured['motion_summary']}")
    if structured.get("route_memory"):
        parts.append(f"Route: {structured['route_memory']}")
    if structured.get("landmark_summary"):
        parts.append(f"Sampling: {structured['landmark_summary']}")
    if structured.get("progress_memory"):
        parts.append(f"Progress: {structured['progress_memory']}")
    if structured.get("event_memory"):
        parts.append(f"Events: {structured['event_memory']}")

    return "[Spatial Memory Context]\n" + "\n".join(
        f"  {p}" for p in parts
    ) if parts else ""


def format_path_timeline(spatial_memory: dict) -> str:
    """Format a movement timeline for Progress Evaluation.

    With MemoryBuilder output this produces a temporal-position summary
    since we don't have parsed path_segments (no LLM perception).
    """
    if spatial_memory.get("motion_summary"):
        return (
            "[Movement Timeline — temporal position of sampled frames]:\n"
            f"  {spatial_memory['motion_summary']}\n"
            f"  {spatial_memory.get('route_memory', '')}\n"
            "  (Base visual evidence decisions on scene content, not frame position.)"
        )
    return ""
