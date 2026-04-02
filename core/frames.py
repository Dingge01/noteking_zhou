"""Key frame extraction: scene detection + subtitle time alignment."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .subtitle import SubtitleResult


@dataclass
class ExtractedFrame:
    path: Path
    timestamp: float
    scene_score: float = 0.0
    description: str = ""

    @property
    def timestamp_str(self) -> str:
        h = int(self.timestamp // 3600)
        m = int((self.timestamp % 3600) // 60)
        s = int(self.timestamp % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    max_frames: int = 20,
    interval_seconds: float = 0,
    threshold: float = 27.0,
) -> list[ExtractedFrame]:
    """Extract key frames using scene detection or fixed intervals.

    Uses PySceneDetect when available, falls back to ffmpeg interval extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _extract_with_scenedetect(
            video_path, output_dir, max_frames, threshold
        )
    except ImportError:
        pass

    if interval_seconds <= 0:
        duration = _get_duration(video_path)
        if duration > 0:
            interval_seconds = max(duration / max_frames, 10)
        else:
            interval_seconds = 30

    return _extract_with_ffmpeg(video_path, output_dir, interval_seconds, max_frames)


def _extract_with_scenedetect(
    video_path: Path,
    output_dir: Path,
    max_frames: int,
    threshold: float,
) -> list[ExtractedFrame]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    frames: list[ExtractedFrame] = []
    for i, (start, end) in enumerate(scene_list[:max_frames]):
        mid_time = (start.get_seconds() + end.get_seconds()) / 2
        frame_path = output_dir / f"frame_{i:04d}_{mid_time:.1f}s.jpg"

        _extract_frame_at(video_path, mid_time, frame_path)

        if frame_path.exists():
            frames.append(ExtractedFrame(
                path=frame_path,
                timestamp=mid_time,
                scene_score=threshold,
            ))

    return frames


def _extract_with_ffmpeg(
    video_path: Path,
    output_dir: Path,
    interval: float,
    max_frames: int,
) -> list[ExtractedFrame]:
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(output_dir / "frame_%04d.jpg"),
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)

    frames: list[ExtractedFrame] = []
    for i, fp in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
        ts = i * interval
        frames.append(ExtractedFrame(path=fp, timestamp=ts))
    return frames


def _extract_frame_at(video_path: Path, timestamp: float, output: Path) -> None:
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output),
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)


def _get_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def align_frames_to_subtitles(
    frames: list[ExtractedFrame],
    subtitles: SubtitleResult,
    tolerance: float = 5.0,
) -> list[tuple[ExtractedFrame, str]]:
    """Align extracted frames to the nearest subtitle text."""
    aligned = []
    for frame in frames:
        best_text = ""
        best_dist = float("inf")
        for seg in subtitles.segments:
            mid = (seg.start + seg.end) / 2
            dist = abs(frame.timestamp - mid)
            if dist < best_dist and dist <= tolerance + (seg.end - seg.start):
                best_dist = dist
                best_text = seg.text
        aligned.append((frame, best_text))
    return aligned
