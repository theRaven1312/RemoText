"""Module 4: Mask Generator — Create binary inpainting masks from text detections."""

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

from .detector import FrameDetections

logger = logging.getLogger(__name__)


@dataclass
class MaskResult:
    """Result of mask generation for the entire video."""

    mask_paths: list[str]      # Paths to saved mask images
    has_mask: list[bool]       # Whether each frame has any mask content
    total_masked_frames: int   # Number of frames with non-empty masks


class MaskGenerator:
    """Generate binary inpainting masks from text detection polygons.

    Creates precise pixel-level masks with configurable dilation and feathering
    to ensure complete coverage of text while minimizing damage to surrounding areas.

    Pipeline: Polygon → Fill → Dilate → Feather → Threshold → Binary Mask

    Attributes:
        dilation_kernel: Size of the dilation kernel (pixels).
        dilation_iterations: Number of dilation passes.
        feather_radius: Gaussian blur kernel size for edge feathering.
        padding: Extra padding around detected text polygons (pixels).
    """

    def __init__(
        self,
        dilation_kernel: int = 3,
        dilation_iterations: int = 1,
        feather_radius: int = 2,
        padding: int = 2,
    ):
        self.dilation_kernel = dilation_kernel
        self.dilation_iterations = dilation_iterations
        self.feather_radius = feather_radius
        self.padding = padding

    def generate_mask(
        self,
        frame_shape: tuple[int, int, int],
        detections: list,
    ) -> np.ndarray:
        """Generate a binary mask for a single frame from text detections.

        Args:
            frame_shape: Shape of the frame (height, width, channels).
            detections: List of TextDetection objects for this frame.

        Returns:
            Binary mask as numpy array (H, W), values 0 or 255.
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not detections:
            return mask

        for det in detections:
            # Convert polygon to integer points
            pts = np.array(det.polygon, dtype=np.int32)

            # Add padding by expanding the polygon
            if self.padding > 0:
                # Compute centroid
                centroid = pts.mean(axis=0)
                # Expand each point outward from centroid
                direction = pts - centroid
                norms = np.linalg.norm(direction, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-6)  # Avoid division by zero
                unit_dir = direction / norms
                pts = (pts + unit_dir * self.padding).astype(np.int32)

            # Fill the polygon
            cv2.fillPoly(mask, [pts], 255)

        # Dilation — expand mask to cover anti-aliased text edges
        if self.dilation_kernel > 0 and self.dilation_iterations > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_kernel, self.dilation_kernel),
            )
            mask = cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

        # Feathering — smooth mask edges for natural blending
        if self.feather_radius > 0:
            ksize = self.feather_radius * 2 + 1  # Must be odd
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Threshold back to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask

    def generate_batch(
        self,
        frame_paths: list[str],
        all_detections: dict[int, FrameDetections],
        output_dir: str,
        verbose: bool = True,
    ) -> MaskResult:
        """Generate masks for all frames and save to disk.

        Args:
            frame_paths: List of frame image paths.
            all_detections: Dict mapping frame index to filtered detections.
            output_dir: Directory to save mask images.
            verbose: Show progress bar.

        Returns:
            MaskResult with paths and statistics.
        """
        os.makedirs(output_dir, exist_ok=True)

        mask_paths: list[str] = []
        has_mask: list[bool] = []
        total_masked = 0

        iterator = enumerate(frame_paths)
        if verbose:
            iterator = tqdm(list(iterator), desc="Generating masks", unit="frame")

        for idx, frame_path in iterator:
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Cannot read frame: {frame_path}")
                # Create empty mask
                mask = np.zeros((720, 1280), dtype=np.uint8)
            else:
                frame_det = all_detections.get(idx)
                dets = frame_det.detections if frame_det else []
                mask = self.generate_mask(frame.shape, dets)

            # Save mask
            mask_filename = f"mask_{idx:06d}.png"
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, mask)
            mask_paths.append(mask_path)

            frame_has_mask = mask.any()
            has_mask.append(frame_has_mask)
        # Temporal smoothing: fill isolated 1-frame gaps to prevent subtitle flicker
        for idx in range(1, len(mask_paths) - 1):
            if not has_mask[idx] and has_mask[idx - 1] and has_mask[idx + 1]:
                m_prev = cv2.imread(mask_paths[idx - 1], cv2.IMREAD_GRAYSCALE)
                m_next = cv2.imread(mask_paths[idx + 1], cv2.IMREAD_GRAYSCALE)
                if m_prev is not None and m_next is not None:
                    bridged = cv2.bitwise_or(m_prev, m_next)
                    if bridged.any():
                        cv2.imwrite(mask_paths[idx], bridged)
                        has_mask[idx] = True
                        total_masked += 1

        logger.info(
            f"Masks generated: {total_masked}/{len(frame_paths)} frames have text to remove"
        )

        return MaskResult(
            mask_paths=mask_paths,
            has_mask=has_mask,
            total_masked_frames=total_masked,
        )
