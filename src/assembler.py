"""Module 6: Video Assembler — Reassemble processed frames with audio into final video."""

import logging
import os

from .utils.video_utils import VideoMetadata, assemble_video

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Assemble inpainted frames and original audio into the final output video.

    Ensures output matches input quality, resolution, and audio fidelity.

    Attributes:
        codec: Video codec for output ('libx264', 'libx265', etc.).
        crf: Constant Rate Factor. 'auto' matches input quality.
        preset: Encoding speed preset.
        pixel_format: Pixel format for output.
        audio_copy: If True, copies audio without re-encoding.
    """

    def __init__(
        self,
        codec: str = "libx264",
        crf: int | str = "auto",
        preset: str = "slow",
        pixel_format: str = "yuv420p",
        audio_copy: bool = True,
    ):
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.pixel_format = pixel_format
        self.audio_copy = audio_copy

    def assemble(
        self,
        frames_dir: str,
        frame_format: str,
        audio_path: str | None,
        metadata: VideoMetadata,
        output_path: str,
    ) -> str:
        """Assemble the final output video.

        Args:
            frames_dir: Directory containing inpainted frame images.
            frame_format: Frame image format ('png' or 'jpg').
            audio_path: Path to extracted audio file, or None.
            metadata: Original video metadata for quality matching.
            output_path: Path for the output video file.

        Returns:
            Path to the assembled output video.
        """
        logger.info(f"Assembling output video: {output_path}")
        logger.info(
            f"Settings: codec={self.codec}, crf={self.crf}, "
            f"preset={self.preset}, fps={metadata.fps:.2f}"
        )

        # Use audio_copy setting
        actual_audio = audio_path if self.audio_copy else None

        result_path = assemble_video(
            frames_dir=frames_dir,
            frame_format=frame_format,
            audio_path=actual_audio,
            output_path=output_path,
            fps=metadata.fps,
            codec=self.codec,
            crf=self.crf,
            preset=self.preset,
            pixel_format=self.pixel_format,
            video_bitrate=metadata.video_bitrate,
        )

        # Verify output
        if os.path.exists(result_path):
            output_size = os.path.getsize(result_path)
            input_size = metadata.file_size
            ratio = output_size / input_size if input_size > 0 else 0
            logger.info(
                f"Output video: {output_size / (1024 * 1024):.1f} MB "
                f"({ratio:.0%} of original {input_size / (1024 * 1024):.1f} MB)"
            )
        else:
            raise RuntimeError(f"Failed to create output video: {result_path}")

        return result_path
