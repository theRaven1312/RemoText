"""FFmpeg wrapper utilities for video I/O operations."""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    width: int
    height: int
    fps: float
    duration: float  # seconds
    total_frames: int
    video_codec: str
    video_bitrate: int | None  # bits/s
    audio_codec: str | None
    audio_bitrate: int | None
    audio_sample_rate: int | None
    pixel_format: str
    file_size: int  # bytes

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


def probe_video(video_path: str) -> VideoMetadata:
    """Extract comprehensive metadata from a video file using ffprobe.

    Args:
        video_path: Path to the input video.

    Returns:
        VideoMetadata with all relevant properties.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If ffprobe fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    audio_stream = None
    for stream in info.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError(f"No video stream found in: {video_path}")

    # Parse FPS from r_frame_rate (e.g., "30000/1001")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    # Parse duration
    duration = float(video_stream.get("duration", info.get("format", {}).get("duration", 0)))

    # Parse total frames
    total_frames = int(video_stream.get("nb_frames", 0))
    if total_frames == 0:
        total_frames = int(fps * duration)

    # Parse bitrate
    video_bitrate = int(video_stream.get("bit_rate", 0)) or None
    if video_bitrate is None:
        format_bitrate = info.get("format", {}).get("bit_rate")
        if format_bitrate:
            video_bitrate = int(format_bitrate)

    # Audio info
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    audio_bitrate = int(audio_stream.get("bit_rate", 0)) or None if audio_stream else None
    audio_sample_rate = int(audio_stream.get("sample_rate", 0)) or None if audio_stream else None

    return VideoMetadata(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        duration=duration,
        total_frames=total_frames,
        video_codec=video_stream.get("codec_name", "unknown"),
        video_bitrate=video_bitrate,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        audio_sample_rate=audio_sample_rate,
        pixel_format=video_stream.get("pix_fmt", "yuv420p"),
        file_size=int(info.get("format", {}).get("size", 0)),
    )


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_format: str = "png",
) -> list[str]:
    """Extract all frames from a video as lossless images.

    Args:
        video_path: Path to the input video.
        output_dir: Directory to save extracted frames.
        frame_format: Image format ('png' for lossless, 'jpg' for faster).

    Returns:
        Sorted list of frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, f"frame_%06d.{frame_format}")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:v", "0",  # Best quality
        pattern,
        "-y",  # Overwrite
    ]

    logger.info(f"Extracting frames to {output_dir}...")
    subprocess.run(cmd, capture_output=True, check=True)

    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(f".{frame_format}")]
    )
    logger.info(f"Extracted {len(frames)} frames")
    return frames


def extract_audio(video_path: str, output_path: str) -> str | None:
    """Extract audio stream from video without re-encoding.

    Args:
        video_path: Path to the input video.
        output_path: Path to save the audio file.

    Returns:
        Path to the extracted audio file, or None if no audio stream.
    """
    # Check if audio stream exists
    metadata = probe_video(video_path)
    if metadata.audio_codec is None:
        logger.info("No audio stream found in video")
        return None

    # Determine audio extension based on codec
    codec_ext_map = {
        "aac": "m4a",  # MP4 container preserves HE-AAC (SBR/PS) AudioSpecificConfig
        "mp3": "mp3",
        "opus": "opus",
        "vorbis": "ogg",
        "flac": "flac",
        "pcm_s16le": "wav",
    }
    ext = codec_ext_map.get(metadata.audio_codec, "m4a")
    if not output_path.endswith(f".{ext}"):
        output_path = f"{output_path}.{ext}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",        # No video
        "-c:a", "copy",  # Copy audio without re-encoding
        output_path,
    ]

    logger.info(f"Extracting audio to {output_path}...")
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def assemble_video(
    frames_dir: str,
    frame_format: str,
    audio_path: str | None,
    output_path: str,
    fps: float,
    codec: str = "libx264",
    crf: int | str = "auto",
    preset: str = "slow",
    pixel_format: str = "yuv420p",
    video_bitrate: int | None = None,
) -> str:
    """Assemble processed frames and audio into final output video.

    Args:
        frames_dir: Directory containing processed frame images.
        frame_format: Frame image format (png/jpg).
        audio_path: Path to audio file, or None for no audio.
        output_path: Path for the output video.
        fps: Frame rate of the output video.
        codec: Video codec for encoding.
        crf: Constant Rate Factor (0-51, lower = higher quality). 'auto' calculates from bitrate.
        preset: FFmpeg encoding preset.
        pixel_format: Pixel format for output.
        video_bitrate: Original video bitrate for CRF estimation.

    Returns:
        Path to the assembled output video.
    """
    # Auto-calculate CRF from original bitrate
    if crf == "auto":
        if video_bitrate and video_bitrate > 0:
            # Rough mapping: higher bitrate → lower CRF
            mbps = video_bitrate / 1_000_000
            if mbps >= 20:
                crf = 17
            elif mbps >= 10:
                crf = 19
            elif mbps >= 5:
                crf = 21
            else:
                crf = 23
        else:
            crf = 20  # Safe default

    logger.info(f"Assembling video with codec={codec}, CRF={crf}, preset={preset}")

    frame_pattern = os.path.join(frames_dir, f"frame_%06d.{frame_format}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Delete existing output file to prevent faststart double-moov corruption on in-place overwrites
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
    ]

    # Add audio if available
    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-i", audio_path, "-c:a", "copy"])

    cmd.extend([
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", pixel_format,
        "-movflags", "+faststart",
        output_path,
    ])

    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"Output video saved to: {output_path}")
    return output_path
