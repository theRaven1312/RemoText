"""Module 3: Chinese Text Filter — Filter detections to only Chinese text."""

import logging
import re
from dataclasses import dataclass

from .detector import FrameDetections, TextDetection

logger = logging.getLogger(__name__)

# Unicode ranges covering Chinese/CJK characters
_CHINESE_REGEX = re.compile(
    r"["
    r"\u4e00-\u9fff"          # CJK Unified Ideographs (main block, ~20k chars)
    r"\u3400-\u4dbf"          # CJK Unified Ideographs Extension A
    r"\U00020000-\U0002a6df"  # CJK Unified Ideographs Extension B
    r"\U0002a700-\U0002b73f"  # CJK Unified Ideographs Extension C
    r"\U0002b740-\U0002b81f"  # CJK Unified Ideographs Extension D
    r"\uf900-\ufaff"          # CJK Compatibility Ideographs
    r"\u3000-\u303f"          # CJK Symbols and Punctuation
    r"\uff00-\uffef"          # Fullwidth Forms (Chinese punctuation)
    r"\u2e80-\u2eff"          # CJK Radicals Supplement
    r"\u3100-\u312f"          # Bopomofo
    r"\u31a0-\u31bf"          # Bopomofo Extended
    r"]"
)

# Common non-Chinese scripts that might coexist
_LATIN_ONLY_REGEX = re.compile(r"^[a-zA-Z0-9\s\.\,\!\?\-\_\@\#\$\%\&\*\(\)\[\]\{\}\/\\:;\"\'<>+=~`]+$")


def contains_chinese(text: str) -> bool:
    """Check if text contains at least one Chinese character.

    Args:
        text: The text string to check.

    Returns:
        True if the text contains Chinese characters.
    """
    return bool(_CHINESE_REGEX.search(text))


def chinese_char_ratio(text: str) -> float:
    """Calculate the ratio of Chinese characters in the text.

    Args:
        text: The text string to analyze.

    Returns:
        Float between 0.0 and 1.0 representing the ratio of Chinese chars.
    """
    if not text:
        return 0.0
    chinese_count = len(_CHINESE_REGEX.findall(text))
    return chinese_count / len(text)


class ChineseTextFilter:
    """Filter text detections to keep only those containing Chinese characters.

    Supports two modes:
    - remove_entire_cluster=True: If ANY Chinese char is found in a text block,
      remove the entire block (including numbers, symbols, etc.)
    - remove_entire_cluster=False: Only mark the Chinese character portions

    Attributes:
        remove_entire_cluster: Whether to remove entire text clusters containing Chinese.
        remove_logos_watermarks: Whether to remove logos/watermarks with Chinese text.
        min_confidence: Minimum OCR confidence to consider a detection valid.
        min_text_size: Minimum text height in pixels to consider.
        max_text_size: Maximum text height in pixels (None = no limit).
    """

    def __init__(
        self,
        remove_entire_cluster: bool = True,
        remove_logos_watermarks: bool = True,
        min_confidence: float = 0.5,
        min_text_size: float = 10,
        max_text_size: float | None = None,
    ):
        self.remove_entire_cluster = remove_entire_cluster
        self.remove_logos_watermarks = remove_logos_watermarks
        self.min_confidence = min_confidence
        self.min_text_size = min_text_size
        self.max_text_size = max_text_size

    def should_remove(self, detection: TextDetection) -> bool:
        """Determine if a text detection should be removed.

        Args:
            detection: A TextDetection object to evaluate.

        Returns:
            True if this detection should be removed (inpainted).
        """
        # Check confidence threshold
        if detection.confidence < self.min_confidence:
            return False

        # Check text size constraints
        height = detection.text_height
        if height < self.min_text_size:
            return False
        if self.max_text_size is not None and height > self.max_text_size:
            return False

        # Check for Chinese content
        text = detection.text.strip()
        if not text:
            return False

        if self.remove_entire_cluster:
            # If the text block contains ANY Chinese → remove entire block
            return contains_chinese(text)
        else:
            # More conservative: only remove if predominantly Chinese
            return chinese_char_ratio(text) > 0.3

    def filter_detections(
        self,
        all_detections: dict[int, FrameDetections],
    ) -> dict[int, FrameDetections]:
        """Filter all frame detections to keep only Chinese text for removal.

        Args:
            all_detections: Dict mapping frame index to FrameDetections.

        Returns:
            Filtered dict with only Chinese text detections marked for removal.
        """
        filtered: dict[int, FrameDetections] = {}
        total_detected = 0
        total_chinese = 0

        for idx, frame_det in all_detections.items():
            chinese_dets = []
            for det in frame_det.detections:
                total_detected += 1
                if self.should_remove(det):
                    chinese_dets.append(det)
                    total_chinese += 1

            filtered[idx] = FrameDetections(
                frame_idx=frame_det.frame_idx,
                detections=chinese_dets,
                is_keyframe=frame_det.is_keyframe,
            )

        # Deduplicate counting (many frames share same detections via keyframe reuse)
        frames_with_text = sum(1 for fd in filtered.values() if fd.detections)
        logger.info(
            f"Chinese filter: {total_chinese}/{total_detected} detections are Chinese, "
            f"{frames_with_text}/{len(filtered)} frames have Chinese text"
        )

        return filtered
