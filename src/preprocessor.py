"""Module 1: Video Preprocessor — Extract frames, audio, and metadata from video."""

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass

from .utils.video_utils import VideoMetadata, extract_audio, extract_frames, probe_video

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Result of video preprocessing."""

    frame_paths: list[str]
    audio_path: str | None
    metadata: VideoMetadata
    temp_dir: str  # Temporary working directory


class VideoPreprocessor:
    """Extracts frames, audio, and metadata from input video.

    Handles any video format/resolution/aspect ratio by leveraging FFmpeg.

    Attributes:
        frame_format: Image format for extracted frames ('png' or 'jpg').
        temp_dir: Optional fixed temporary directory. If None, auto-creates one.
    """

    def __init__(self, frame_format: str = "png", temp_dir: str | None = None):
        self.frame_format = frame_format
        self.temp_dir = temp_dir

    def extract(self, video_path: str) -> PreprocessResult:
        """Extract all components from the input video.

        Args:
            video_path: Path to the input video file.

        Returns:
            PreprocessResult containing frame paths, audio path, metadata,
            and the temporary directory used.

        Raises:
            FileNotFoundError: If video_path does not exist.
            RuntimeError: If FFmpeg/ffprobe fails.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # Create working directory
        work_dir = self.temp_dir or tempfile.mkdtemp(prefix="remotext_")
        frames_dir = os.path.join(work_dir, "frames")
        audio_dir = os.path.join(work_dir, "audio")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        logger.info(f"Preprocessing video: {video_path}")
        logger.info(f"Working directory: {work_dir}")

        # Step 1: Extract metadata
        metadata = probe_video(video_path)
        logger.info(
            f"Video info: {metadata.width}x{metadata.height} @ {metadata.fps:.2f}fps, "
            f"{metadata.duration:.1f}s, codec={metadata.video_codec}, "
            f"audio={metadata.audio_codec or 'none'}"
        )

        # Step 2: Extract frames
        frame_paths = extract_frames(video_path, frames_dir, self.frame_format)
        logger.info(f"Extracted {len(frame_paths)} frames")

        # Step 3: Extract audio
        audio_path = extract_audio(
            video_path,
            os.path.join(audio_dir, "audio"),
        )
        if audio_path:
            logger.info(f"Audio extracted: {audio_path}")
        else:
            logger.info("No audio stream in video")

        return PreprocessResult(
            frame_paths=frame_paths,
            audio_path=audio_path,
            metadata=metadata,
            temp_dir=work_dir,
        )

    @staticmethod
    def cleanup(temp_dir: str):
        """Remove temporary working directory.

        Args:
            temp_dir: Path to the temporary directory to remove.
        """
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
