"""Tests for the Chinese text filter module."""

import pytest
from src.filter import contains_chinese, chinese_char_ratio, ChineseTextFilter
from src.detector import TextDetection


class TestContainsChinese:
    """Test the contains_chinese utility function."""

    def test_pure_chinese(self):
        assert contains_chinese("你好世界") is True

    def test_pure_english(self):
        assert contains_chinese("Hello World") is False

    def test_mixed_chinese_english(self):
        assert contains_chinese("Hello 你好") is True

    def test_chinese_with_numbers(self):
        assert contains_chinese("第3集") is True
        assert contains_chinese("2024年") is True

    def test_pure_numbers(self):
        assert contains_chinese("12345") is False

    def test_japanese_kanji(self):
        # Japanese Kanji shares the CJK range
        assert contains_chinese("漢字") is True

    def test_korean(self):
        # Korean Hangul is NOT in the CJK Unified Ideographs range
        assert contains_chinese("한국어") is False

    def test_chinese_punctuation(self):
        assert contains_chinese("。") is True
        assert contains_chinese("，") is True

    def test_fullwidth_chars(self):
        assert contains_chinese("Ａ") is True  # Fullwidth A is in the range

    def test_empty_string(self):
        assert contains_chinese("") is False

    def test_special_symbols(self):
        assert contains_chinese("@#$%") is False

    def test_traditional_chinese(self):
        assert contains_chinese("繁體中文") is True


class TestChineseCharRatio:
    """Test the chinese_char_ratio utility function."""

    def test_all_chinese(self):
        ratio = chinese_char_ratio("你好世界")
        assert ratio == 1.0

    def test_no_chinese(self):
        ratio = chinese_char_ratio("Hello")
        assert ratio == 0.0

    def test_mixed(self):
        ratio = chinese_char_ratio("Hi你好")
        assert 0.4 < ratio < 0.6  # 2 Chinese out of 4 chars

    def test_empty(self):
        ratio = chinese_char_ratio("")
        assert ratio == 0.0


class TestChineseTextFilter:
    """Test the ChineseTextFilter class."""

    @staticmethod
    def _make_detection(text: str, confidence: float = 0.9, height: float = 30.0) -> TextDetection:
        """Helper to create a TextDetection with dummy polygon."""
        # Create a polygon that gives the specified height
        return TextDetection(
            polygon=[[0, 0], [100, 0], [100, height], [0, height]],
            text=text,
            confidence=confidence,
        )

    def test_remove_chinese_cluster(self):
        """Entire cluster containing Chinese should be removed."""
        filt = ChineseTextFilter(remove_entire_cluster=True)
        det = self._make_detection("第3集 Hello")
        assert filt.should_remove(det) is True

    def test_keep_english_only(self):
        """Pure English text should not be removed."""
        filt = ChineseTextFilter(remove_entire_cluster=True)
        det = self._make_detection("Episode 3")
        assert filt.should_remove(det) is False

    def test_low_confidence_skip(self):
        """Low confidence detections should be skipped."""
        filt = ChineseTextFilter(min_confidence=0.7)
        det = self._make_detection("你好", confidence=0.5)
        assert filt.should_remove(det) is False

    def test_min_text_size(self):
        """Text smaller than min size should be skipped."""
        filt = ChineseTextFilter(min_text_size=20)
        det = self._make_detection("你好", height=15)
        assert filt.should_remove(det) is False

    def test_max_text_size(self):
        """Text larger than max size should be skipped."""
        filt = ChineseTextFilter(max_text_size=50)
        det = self._make_detection("你好", height=100)
        assert filt.should_remove(det) is False

    def test_conservative_mode(self):
        """Conservative mode only removes predominantly Chinese text."""
        filt = ChineseTextFilter(remove_entire_cluster=False)
        # Mostly English with one Chinese char
        det = self._make_detection("Hello World 你")
        # Ratio = 1/13 ≈ 0.077 < 0.3 threshold
        assert filt.should_remove(det) is False

        # Mostly Chinese
        det2 = self._make_detection("你好世界Hi")
        # Ratio = 4/6 ≈ 0.67 > 0.3 threshold
        assert filt.should_remove(det2) is True

    def test_empty_text(self):
        """Empty text should not be removed."""
        filt = ChineseTextFilter()
        det = self._make_detection("", confidence=0.9)
        assert filt.should_remove(det) is False
