"""Scene change detection using SSIM for keyframe optimization."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_ssim_gray(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute structural similarity between two frames (grayscale).

    Uses a lightweight OpenCV-based SSIM that avoids importing skimage
    for speed in the hot path. Falls back to a simpler metric if needed.

    Args:
        frame_a: First frame (BGR or grayscale).
        frame_b: Second frame (BGR or grayscale).

    Returns:
        SSIM score between 0.0 (completely different) and 1.0 (identical).
    """
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

    # Convert to grayscale if needed
    if len(frame_a.shape) == 3:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    else:
        gray_a = frame_a

    if len(frame_b.shape) == 3:
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        gray_b = frame_b

    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    gray_a = gray_a.astype(np.float64)
    gray_b = gray_b.astype(np.float64)

    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_histogram_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Fast histogram-based scene change score.

    Much faster than SSIM but less accurate. Good for pre-filtering.

    Args:
        frame_a: First frame (BGR).
        frame_b: Second frame (BGR).

    Returns:
        Similarity score between 0.0 (different) and 1.0 (identical).
    """
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

    # Compute histograms for each channel
    similarity = 0.0
    for i in range(3):
        hist_a = cv2.calcHist([frame_a], [i], None, [64], [0, 256])
        hist_b = cv2.calcHist([frame_b], [i], None, [64], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        similarity += cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    return similarity / 3.0


class SceneChangeDetector:
    """Detects scene changes and determines keyframes for OCR processing.

    Uses a two-tier approach:
    1. Fast histogram check as pre-filter
    2. SSIM check for borderline cases

    Attributes:
        ssim_threshold: SSIM threshold below which a scene change is detected.
        hist_threshold: Histogram correlation threshold for fast pre-filter.
        force_interval: Force a keyframe every N frames regardless of similarity.
    """

    def __init__(
        self,
        ssim_threshold: float = 0.90,
        hist_threshold: float = 0.85,
        force_interval: int = 30,
    ):
        self.ssim_threshold = ssim_threshold
        self.hist_threshold = hist_threshold
        self.force_interval = force_interval

        self._prev_frame: np.ndarray | None = None
        self._frames_since_keyframe: int = 0

    def reset(self):
        """Reset detector state for a new video."""
        self._prev_frame = None
        self._frames_since_keyframe = 0

    def is_keyframe(self, frame: np.ndarray) -> bool:
        """Determine if the given frame should be treated as a keyframe.

        A frame is a keyframe if:
        - It's the first frame
        - Force interval has elapsed
        - Scene has changed significantly

        Args:
            frame: Current video frame (BGR).

        Returns:
            True if this frame should have full OCR processing.
        """
        self._frames_since_keyframe += 1

        # First frame is always a keyframe
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            self._frames_since_keyframe = 0
            return True

        # Force keyframe at regular intervals
        if self._frames_since_keyframe >= self.force_interval:
            self._prev_frame = frame.copy()
            self._frames_since_keyframe = 0
            return True

        # Fast histogram pre-check
        hist_sim = compute_histogram_diff(self._prev_frame, frame)
        if hist_sim < self.hist_threshold:
            # Definite scene change
            self._prev_frame = frame.copy()
            self._frames_since_keyframe = 0
            return True

        # Borderline — use more accurate SSIM
        if hist_sim < self.ssim_threshold:
            # Downsample for faster SSIM computation
            scale = 0.5
            small_prev = cv2.resize(self._prev_frame, None, fx=scale, fy=scale)
            small_curr = cv2.resize(frame, None, fx=scale, fy=scale)
            ssim = compute_ssim_gray(small_prev, small_curr)

            if ssim < self.ssim_threshold:
                self._prev_frame = frame.copy()
                self._frames_since_keyframe = 0
                return True

        # No scene change — reuse previous detections
        return False
