import logging
import os
from dataclasses import dataclass, field

# Prevent PaddlePaddle 3.x PIR-to-oneDNN NotImplementedError and speed up initialization
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
from tqdm import tqdm

from .utils.scene_detect import SceneChangeDetector

logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """A single detected text region in a frame."""

    polygon: list[list[float]]  # 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    confidence: float

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Get axis-aligned bounding box (x_min, y_min, x_max, y_max)."""
        pts = np.array(self.polygon)
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    @property
    def area(self) -> float:
        """Calculate area of the polygon."""
        pts = np.array(self.polygon)
        return float(cv2.contourArea(pts.astype(np.float32)))

    @property
    def text_height(self) -> float:
        """Estimated text height in pixels."""
        pts = np.array(self.polygon)
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        return float(y_max - y_min)


@dataclass
class FrameDetections:
    """All text detections for a single frame."""

    frame_idx: int
    detections: list[TextDetection] = field(default_factory=list)
    is_keyframe: bool = False  # Whether this frame had full OCR processing


def _box_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _is_text_present(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    min_laplacian_var: float = 4.0,
    min_contrast_std: float = 5.0,
) -> bool:
    """Check if a bounding box region in frame contains text edges and contrast."""
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    x2 = min(frame.shape[1], box[2])
    y2 = min(frame.shape[0], box[3])
    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    std = float(gray.std())
    return lap_var >= min_laplacian_var and std >= min_contrast_std


class TextDetector:
    """Detect text in video frames using PaddleOCR with keyframe optimization.

    Uses scene change detection to avoid running full OCR on every frame.
    Only keyframes get full OCR; similar frames reuse previous detections.

    Attributes:
        lang: OCR language ('ch' for Simplified Chinese, 'chinese_cht' for Traditional).
        use_angle_cls: Enable angle classification for rotated text.
        det_db_thresh: Detection confidence threshold.
        rec_score_thresh: Recognition score threshold.
        keyframe_interval: Maximum frames between forced OCR runs.
        scene_change_thresh: SSIM threshold for scene change detection.
    """

    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.25,
        rec_score_thresh: float = 0.35,
        keyframe_interval: int = 10,
        scene_change_thresh: float = 0.90,
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.rec_score_thresh = rec_score_thresh
        self.keyframe_interval = keyframe_interval
        self.scene_change_thresh = scene_change_thresh

        self._ocr = None  # Lazy init

    def _init_ocr(self):
        """Initialize PaddleOCR model lazily, auto-detecting GPU support."""
        if self._ocr is None:
            # Optimize environment flags
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

            import paddle
            use_gpu = False
            try:
                if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                    use_gpu = True
            except Exception:
                use_gpu = False

            from paddleocr import PaddleOCR

            ocr_kwargs = {
                "lang": self.lang,
                "use_textline_orientation": self.use_angle_cls,
                "text_det_box_thresh": self.det_db_thresh,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            }

            if use_gpu:
                ocr_kwargs["device"] = "gpu:0"
                ocr_kwargs["use_gpu"] = True
                device_name = "GPU (CUDA)"
            else:
                ocr_kwargs["device"] = "cpu"
                ocr_kwargs["use_gpu"] = False
                ocr_kwargs["enable_mkldnn"] = False
                device_name = "CPU"

            try:
                self._ocr = PaddleOCR(**ocr_kwargs)
            except TypeError:
                ocr_kwargs.pop("device", None)
                ocr_kwargs.pop("enable_mkldnn", None)
                self._ocr = PaddleOCR(**ocr_kwargs)

            logger.info(f"PaddleOCR initialized on {device_name} (lang={self.lang})")

    def detect_single(self, frame: np.ndarray) -> list[TextDetection]:
        """Run OCR detection + recognition on a single frame.

        Args:
            frame: Video frame as BGR numpy array.

        Returns:
            List of TextDetection objects found in the frame.
        """
        self._init_ocr()

        results = self._ocr.predict(frame)

        detections = []
        if results:
            for res in results:
                polys = res.get("dt_polys", [])
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])

                for polygon, text, confidence in zip(polys, texts, scores):
                    if confidence >= self.rec_score_thresh:
                        # Convert numpy array (4,2) to list of [x, y] pairs
                        poly_list = polygon.tolist() if hasattr(polygon, "tolist") else polygon
                        detections.append(TextDetection(
                            polygon=poly_list,
                            text=text,
                            confidence=confidence,
                        ))

        return detections

    def detect_batch(
        self,
        frame_paths: list[str],
        verbose: bool = True,
    ) -> dict[int, FrameDetections]:
        """Run text detection across all video frames with bidirectional keyframe propagation.

        Runs full OCR on keyframes (scene changes or forced intervals).
        Non-keyframes between keyframes inherit detections bidirectionally:
        - Detections present in both keyframes are continuously maintained.
        - Detections present in only one keyframe are verified with edge/contrast checks.

        Args:
            frame_paths: Sorted list of frame image paths.
            verbose: Show progress bar.

        Returns:
            Dict mapping frame index to FrameDetections.
        """
        self._init_ocr()

        scene_detector = SceneChangeDetector(
            ssim_threshold=self.scene_change_thresh,
            force_interval=self.keyframe_interval,
        )

        all_detections: dict[int, FrameDetections] = {}
        keyframe_indices: list[int] = []
        keyframe_dets: dict[int, list[TextDetection]] = {}

        # Pass 1: Identify keyframes and run OCR
        iterator = enumerate(frame_paths)
        if verbose:
            iterator = tqdm(list(iterator), desc="Detecting keyframes", unit="frame")

        for idx, frame_path in iterator:
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Failed to read frame: {frame_path}")
                all_detections[idx] = FrameDetections(frame_idx=idx)
                continue

            if scene_detector.is_keyframe(frame) or idx == 0:
                dets = self.detect_single(frame)
                keyframe_indices.append(idx)
                keyframe_dets[idx] = dets
                all_detections[idx] = FrameDetections(
                    frame_idx=idx,
                    detections=dets,
                    is_keyframe=True,
                )

        # Pass 2: Bidirectional propagation for non-keyframes
        for i in range(len(keyframe_indices)):
            k_curr = keyframe_indices[i]
            k_next = keyframe_indices[i + 1] if i + 1 < len(keyframe_indices) else len(frame_paths)

            dets_curr = keyframe_dets[k_curr]
            dets_next = keyframe_dets.get(k_next, [])

            # Match detections between k_curr and k_next
            persistent_dets: list[TextDetection] = []
            ending_dets: list[TextDetection] = []

            for d_c in dets_curr:
                matched = any(
                    _box_iou(d_c.bounding_box, d_n.bounding_box) > 0.3
                    or (d_c.text == d_n.text and len(d_c.text) > 0)
                    for d_n in dets_next
                )
                if matched:
                    persistent_dets.append(d_c)
                else:
                    ending_dets.append(d_c)

            starting_dets = [
                d_n
                for d_n in dets_next
                if not any(
                    _box_iou(d_n.bounding_box, d_c.bounding_box) > 0.3
                    or (d_n.text == d_c.text and len(d_n.text) > 0)
                    for d_c in dets_curr
                )
            ]

            # Interpolate frames between k_curr and k_next
            for mid_idx in range(k_curr + 1, k_next):
                if mid_idx >= len(frame_paths):
                    break
                frame_mid = cv2.imread(frame_paths[mid_idx])
                if frame_mid is None:
                    all_detections[mid_idx] = FrameDetections(frame_idx=mid_idx)
                    continue

                mid_dets = list(persistent_dets)

                # Check ending detections (forward check from k_curr)
                for d_end in ending_dets:
                    if _is_text_present(frame_mid, d_end.bounding_box):
                        mid_dets.append(d_end)

                # Check starting detections (backward check from k_next)
                for d_start in starting_dets:
                    if _is_text_present(frame_mid, d_start.bounding_box):
                        mid_dets.append(d_start)

                all_detections[mid_idx] = FrameDetections(
                    frame_idx=mid_idx,
                    detections=mid_dets,
                    is_keyframe=False,
                )

        total = len(frame_paths)
        keyframe_count = len(keyframe_indices)
        saved_pct = (1.0 - keyframe_count / total) * 100 if total > 0 else 0
        logger.info(
            f"Detection complete: {keyframe_count} keyframes / {total} total frames "
            f"({saved_pct:.1f}% OCR runs saved)"
        )

        return all_detections
