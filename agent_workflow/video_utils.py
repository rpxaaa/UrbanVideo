"""Shared video frame extraction utility.

Uses category-specific sampling strategies from frame_selector to pick
frame indices, then reads and encodes them as base64 JPEGs with watermarks.
"""

import cv2
import base64
from .frame_selector import get_frame_indices


def extract_frames(video_path: str, num_frames: int = 16, max_size: int = 768,
                   question: str = "", question_category: str = "") -> list[str]:
    """Extract frames using category-specific sampling.

    Returns list of base64-encoded JPEG strings with frame-number watermarks.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    # Get frame indices from category-specific strategy
    frame_indices = get_frame_indices(
        total_frames=total_frames,
        num_frames=num_frames,
        category=question_category,
        question=question,
        video_path=video_path,
    )

    frames_base64 = []
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Frame index watermark for temporal reference
        text = f"Frame {i+1}/{len(frame_indices)}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frames_base64.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    return frames_base64


def get_sampled_indices(video_path: str, num_frames: int = 16,
                        question: str = "", question_category: str = "") -> list[int]:
    """Return the frame indices that would be sampled (without encoding).

    Useful for MemoryBuilder which needs to know which frames were selected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames == 0:
        return []

    return get_frame_indices(
        total_frames=total_frames,
        num_frames=num_frames,
        category=question_category,
        question=question,
        video_path=video_path,
    )
