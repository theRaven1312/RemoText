"""Module 5: Video Inpainter — ProPainter-based video inpainting with LaMa fallback."""

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .mask_generator import MaskResult
from .utils.device_utils import get_device, estimate_max_chunk_size

logger = logging.getLogger(__name__)

# Path to ProPainter repo
PROPAINTER_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "propainter")


@dataclass
class InpaintResult:
    """Result of the inpainting process."""

    output_dir: str            # Directory with inpainted frames
    frame_paths: list[str]     # Paths to inpainted frame images
    frames_inpainted: int      # Number of frames that were actually inpainted
    frames_copied: int         # Number of frames copied as-is (no mask)
    model_used: str            # Which model was used


class ProPainterInpainter:
    """Video inpainting using ProPainter for temporal consistency.

    ProPainter uses flow-guided propagation to borrow real pixels from
    neighboring frames, producing artifact-free results without flickering.

    Attributes:
        device: Compute device ('cuda', 'mps', 'cpu').
        chunk_size: Number of frames to process at once.
        chunk_overlap: Overlap between chunks for smooth transitions.
        fp16: Use half precision (CUDA only).
    """

    def __init__(
        self,
        device: str = "auto",
        chunk_size: int = 80,
        chunk_overlap: int = 10,
        fp16: bool = False,
    ):
        self.device = get_device(device)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fp16 = fp16 and self.device == "cuda"  # fp16 only on CUDA

        self._model = None
        self._model_ready = False

    def _ensure_model(self):
        """Download and set up ProPainter if not already available."""
        if self._model_ready:
            return

        propainter_path = Path(PROPAINTER_DIR)

        if not propainter_path.exists():
            logger.info("Cloning ProPainter repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/sczhou/ProPainter.git", str(propainter_path)],
                check=True,
                capture_output=True,
            )
            logger.info("ProPainter cloned successfully")

        # Check for model weights
        weights_dir = propainter_path / "weights"
        if not weights_dir.exists() or not list(weights_dir.glob("*.pth")):
            logger.info("Downloading ProPainter model weights...")
            self._download_weights(propainter_path)

        # Add ProPainter directories to Python path
        sys.path.insert(0, str(propainter_path))
        hf_inpainter_path = propainter_path / "web-demos" / "hugging_face" / "inpainter"
        if hf_inpainter_path.exists():
            sys.path.insert(0, str(hf_inpainter_path))

        import torch
        from base_inpainter import ProInpainter

        device = torch.device(self.device)
        logger.info(f"Loading ProPainter models on {self.device}...")
        self._model = ProInpainter(
            propainter_checkpoint=str(weights_dir / "ProPainter.pth"),
            raft_checkpoint=str(weights_dir / "raft-things.pth"),
            flow_completion_checkpoint=str(weights_dir / "recurrent_flow_completion.pth"),
            device=device,
            use_half=self.fp16,
        )
        self._model_ready = True
        logger.info(f"ProPainter model ready on {self.device}")

    def _download_weights(self, repo_path: Path):
        """Download ProPainter pretrained weights."""
        weights_dir = repo_path / "weights"
        weights_dir.mkdir(exist_ok=True)

        weight_urls = {
            "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
            "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
            "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
            "i3d_rgb_imagenet.pt": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt",
        }

        for filename, url in weight_urls.items():
            filepath = weights_dir / filename
            if not filepath.exists():
                logger.info(f"Downloading {filename}...")
                subprocess.run(
                    ["curl", "-L", "-o", str(filepath), url],
                    check=True,
                    capture_output=True,
                )

        logger.info("All ProPainter weights downloaded")

    def inpaint(
        self,
        frame_paths: list[str],
        mask_result: MaskResult,
        output_dir: str,
        resolution: tuple[int, int] | None = None,
        verbose: bool = True,
    ) -> InpaintResult:
        """Run ProPainter video inpainting.

        For frames without masks, copies them directly.
        For frames with masks, processes them in segments with context frames.

        Args:
            frame_paths: List of input frame paths.
            mask_result: MaskResult with mask paths and metadata.
            output_dir: Directory to save inpainted frames.
            resolution: (width, height) of the video for chunk size estimation.
            verbose: Show progress.

        Returns:
            InpaintResult with output paths and statistics.
        """
        self._ensure_model()
        os.makedirs(output_dir, exist_ok=True)

        # Determine optimal chunk size based on hardware
        if resolution:
            optimal_chunk = estimate_max_chunk_size(self.device, resolution)
            self.chunk_size = min(self.chunk_size, optimal_chunk)
            logger.info(f"Chunk size adjusted to {self.chunk_size} for {resolution}")

        # Check if any frames need inpainting
        if mask_result.total_masked_frames == 0:
            logger.info("No frames need inpainting, copying all frames")
            return self._copy_all_frames(frame_paths, output_dir)

        # Find contiguous segments that need inpainting
        segments = self._find_inpaint_segments(mask_result)
        logger.info(f"Found {len(segments)} segments needing inpainting")

        inpainted_frames = set()
        output_paths = [""] * len(frame_paths)

        # Process each segment
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            if verbose:
                logger.info(
                    f"Processing segment {seg_idx + 1}/{len(segments)} "
                    f"(frames {seg_start + 1}-{seg_end})..."
                )

            # Expand segment for temporal context
            ctx_start = max(0, seg_start - self.chunk_overlap)
            ctx_end = min(len(frame_paths), seg_end + self.chunk_overlap)

            seg_frames = frame_paths[ctx_start:ctx_end]
            seg_masks = mask_result.mask_paths[ctx_start:ctx_end]

            # Run ProPainter on this segment
            comp_frames = self._run_propainter_segment(seg_frames, seg_masks, verbose)

            # Save ONLY results for the target segment (exclude context padding)
            for i in range(seg_start, seg_end):
                rel_idx = i - ctx_start
                if rel_idx < len(comp_frames):
                    out_name = f"frame_{i + 1:06d}.png"
                    out_path = os.path.join(output_dir, out_name)
                    bgr_frame = cv2.cvtColor(comp_frames[rel_idx], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_path, bgr_frame)
                    output_paths[i] = out_path
                    inpainted_frames.add(i)

        # Copy frames that don't need inpainting (lossless & fast)
        copied = 0
        for idx, frame_path in enumerate(frame_paths):
            if idx not in inpainted_frames:
                out_name = f"frame_{idx + 1:06d}.png"
                out_path = os.path.join(output_dir, out_name)
                if not os.path.exists(out_path):
                    shutil.copy2(frame_path, out_path)
                output_paths[idx] = out_path
                copied += 1

        return InpaintResult(
            output_dir=output_dir,
            frame_paths=output_paths,
            frames_inpainted=len(inpainted_frames),
            frames_copied=copied,
            model_used="propainter",
        )

    def _find_inpaint_segments(
        self, mask_result: MaskResult
    ) -> list[tuple[int, int]]:
        """Find contiguous segments of frames that need inpainting."""
        segments: list[tuple[int, int]] = []
        in_segment = False
        seg_start = 0
        gap = 0
        max_gap = 5  # Merge segments within 5 frames of each other

        for idx, has_mask in enumerate(mask_result.has_mask):
            if has_mask:
                if not in_segment:
                    seg_start = idx
                    in_segment = True
                gap = 0
            else:
                if in_segment:
                    gap += 1
                    if gap > max_gap:
                        segments.append((seg_start, idx - gap))
                        in_segment = False

        if in_segment:
            segments.append((seg_start, len(mask_result.has_mask)))

        return segments

    def _run_propainter_segment(
        self,
        frame_paths: list[str],
        mask_paths: list[str],
        verbose: bool,
    ) -> list[np.ndarray]:
        """Run ProPainter inference on a segment of frames.

        Returns:
            List of RGB numpy frames.
        """
        # On MPS (Apple Silicon), full 720p/1080p RAFT causes out-of-memory.
        # Ratio 0.5 runs reliably within unified memory. On CUDA (e.g. T4 GPU), ratio 1.0 runs at full resolution.
        ratio = 0.5 if self.device == "mps" else 1.0

        # Load RGB frames and masks
        rgb_frames = []
        masks_np = []
        for fp, mp in zip(frame_paths, mask_paths):
            f = cv2.imread(fp)
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if f is not None:
                rgb_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if m is not None:
                masks_np.append(m)
            elif f is not None:
                masks_np.append(np.zeros((f.shape[0], f.shape[1]), dtype=np.uint8))

        if not rgb_frames:
            return []

        try:
            comp_frames = self._model.inpaint(
                rgb_frames,
                masks_np,
                ratio=ratio,
                dilate_radius=0,
                subvideo_length=self.chunk_size,
                neighbor_length=min(10, self.chunk_size // 2),
                ref_stride=5,
            )
            return comp_frames
        except Exception as e:
            logger.warning(f"ProPainter segment error: {e}, falling back to LaMa for segment")
            lama = LaMaInpainter(device=self.device)
            fallback_frames = []
            for f_rgb, m_np in zip(rgb_frames, masks_np):
                if m_np.any():
                    fallback_frames.append(lama.inpaint_single(f_rgb, m_np))
                else:
                    fallback_frames.append(f_rgb)
            return fallback_frames

    def _copy_all_frames(
        self, frame_paths: list[str], output_dir: str
    ) -> InpaintResult:
        """Copy all frames as-is when no inpainting is needed."""
        output_paths = []
        for idx, fp in enumerate(frame_paths):
            out_name = f"frame_{idx + 1:06d}.png"
            out_path = os.path.join(output_dir, out_name)
            shutil.copy2(fp, out_path)
            output_paths.append(out_path)

        return InpaintResult(
            output_dir=output_dir,
            frame_paths=output_paths,
            frames_inpainted=0,
            frames_copied=len(frame_paths),
            model_used="none",
        )


class LaMaInpainter:
    """Image-based inpainter using LaMa (Large Mask Inpainting) neural network.

    Processes frames individually using Fourier Convolutions — fast, high quality,
    crisp background textures without blurring.

    Attributes:
        device: Compute device ('cuda', 'mps', 'cpu').
        refine_iterations: Number of refinement passes.
    """

    def __init__(self, device: str = "auto", refine_iterations: int = 1):
        self.device = get_device(device)
        self.refine_iterations = refine_iterations
        self._model = None

    def _ensure_model(self):
        """Set up LaMa model."""
        if self._model is not None:
            return

        try:
            from simple_lama_inpainting import SimpleLama
            logger.info(f"Loading LaMa neural model on {self.device}...")
            self._model = SimpleLama(device=self.device)
            logger.info("LaMa model ready")
        except Exception as e:
            logger.warning(f"LaMa neural inpainter unavailable ({e}), will use fallback")
            self._model = "fallback"

    def inpaint_single(self, frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint a single RGB frame with a binary mask."""
        self._ensure_model()
        if self._model is not None and self._model != "fallback":
            from PIL import Image

            img_pil = Image.fromarray(frame_rgb)
            mask_pil = Image.fromarray(mask)
            res_pil = self._model(img_pil, mask_pil)
            return np.array(res_pil)
        else:
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            res_bgr = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
            return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

    def inpaint(
        self,
        frame_paths: list[str],
        mask_result: MaskResult,
        output_dir: str,
        verbose: bool = True,
    ) -> InpaintResult:
        """Run LaMa inpainting frame-by-frame.

        Args:
            frame_paths: List of input frame paths.
            mask_result: MaskResult with mask paths.
            output_dir: Output directory.
            verbose: Show progress.

        Returns:
            InpaintResult with output paths and statistics.
        """
        self._ensure_model()
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        inpainted_count = 0
        copied_count = 0

        iterator = enumerate(zip(frame_paths, mask_result.mask_paths, mask_result.has_mask))
        if verbose:
            iterator = tqdm(list(iterator), desc="Inpainting (LaMa)", unit="frame")

        for idx, (frame_path, mask_path, has_mask) in iterator:
            out_name = f"frame_{idx + 1:06d}.png"
            out_path = os.path.join(output_dir, out_name)

            if has_mask:
                frame_bgr = cv2.imread(frame_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if frame_bgr is not None and mask is not None and mask.any():
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    res_rgb = self.inpaint_single(frame_rgb, mask)
                    res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_path, res_bgr)
                    inpainted_count += 1
                else:
                    shutil.copy2(frame_path, out_path)
                    copied_count += 1
            else:
                shutil.copy2(frame_path, out_path)
                copied_count += 1

            output_paths.append(out_path)

        return InpaintResult(
            output_dir=output_dir,
            frame_paths=output_paths,
            frames_inpainted=inpainted_count,
            frames_copied=copied_count,
            model_used="lama",
        )


class VideoInpainter:
    """High-level inpainter with automatic model selection and fallback.

    Tries primary model (ProPainter or LaMa); falls back to the other if primary fails.

    Attributes:
        primary_model: Primary inpainting model ('propainter' or 'lama').
        fallback_model: Fallback model if primary fails.
    """

    def __init__(
        self,
        primary_model: str = "propainter",
        fallback_model: str = "lama",
        device: str = "auto",
        chunk_size: int = 80,
        chunk_overlap: int = 10,
        fp16: bool = False,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fp16 = fp16

        self._propainter = ProPainterInpainter(
            device=device,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            fp16=fp16,
        ) if primary_model == "propainter" else None

        self._lama = LaMaInpainter(device=device)

    def inpaint(
        self,
        frame_paths: list[str],
        mask_result: MaskResult,
        output_dir: str,
        resolution: tuple[int, int] | None = None,
        verbose: bool = True,
    ) -> InpaintResult:
        """Run inpainting with automatic fallback.

        Args:
            frame_paths: List of input frame paths.
            mask_result: MaskResult with mask paths.
            output_dir: Output directory.
            resolution: Video resolution for chunk size estimation.
            verbose: Show progress.

        Returns:
            InpaintResult with output paths and statistics.
        """
        if mask_result.total_masked_frames == 0:
            logger.info("No text detected — skipping inpainting")
            return self._copy_all_frames(frame_paths, output_dir)

        # Try primary model
        if self._propainter and self.primary_model == "propainter":
            try:
                logger.info("Running ProPainter video inpainting...")
                return self._propainter.inpaint(
                    frame_paths, mask_result, output_dir, resolution, verbose
                )
            except Exception as e:
                logger.warning(f"ProPainter failed: {e}, falling back to {self.fallback_model}")

        # Fallback to LaMa
        logger.info(f"Running {self.fallback_model} inpainting...")
        return self._lama.inpaint(frame_paths, mask_result, output_dir, verbose)

    def _copy_all_frames(
        self, frame_paths: list[str], output_dir: str
    ) -> InpaintResult:
        """Copy all frames as-is when no inpainting is needed."""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        for idx, fp in enumerate(frame_paths):
            out_name = f"frame_{idx + 1:06d}.png"
            out_path = os.path.join(output_dir, out_name)
            shutil.copy2(fp, out_path)
            output_paths.append(out_path)

        return InpaintResult(
            output_dir=output_dir,
            frame_paths=output_paths,
            frames_inpainted=0,
            frames_copied=len(frame_paths),
            model_used="none",
        )
