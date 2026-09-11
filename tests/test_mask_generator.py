"""Tests for the mask generator module."""

import numpy as np
import pytest
from src.detector import TextDetection
from src.mask_generator import MaskGenerator


class TestMaskGenerator:
    """Test the MaskGenerator class."""

    @staticmethod
    def _make_detection(x: int, y: int, w: int, h: int) -> TextDetection:
        """Create a TextDetection with a rectangular polygon."""
        return TextDetection(
            polygon=[
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ],
            text="测试",
            confidence=0.95,
        )

    def test_empty_detections(self):
        """Empty detections should produce an all-zero mask."""
        gen = MaskGenerator()
        mask = gen.generate_mask((720, 1280, 3), [])
        assert mask.shape == (720, 1280)
        assert mask.sum() == 0

    def test_single_detection_creates_mask(self):
        """A single detection should create a non-zero mask region."""
        gen = MaskGenerator(dilation_kernel=0, feather_radius=0, padding=0)
        det = self._make_detection(100, 100, 200, 50)
        mask = gen.generate_mask((720, 1280, 3), [det])

        assert mask.shape == (720, 1280)
        assert mask.sum() > 0

        # The center of the detection should be white
        assert mask[125, 200] == 255

        # Far corners should be black
        assert mask[0, 0] == 0
        assert mask[719, 1279] == 0

    def test_dilation_expands_mask(self):
        """Dilation should make the mask larger than the original polygon."""
        gen_no_dilation = MaskGenerator(dilation_kernel=0, feather_radius=0, padding=0)
        gen_with_dilation = MaskGenerator(dilation_kernel=5, dilation_iterations=2, feather_radius=0, padding=0)

        det = self._make_detection(100, 100, 200, 50)

        mask_small = gen_no_dilation.generate_mask((720, 1280, 3), [det])
        mask_large = gen_with_dilation.generate_mask((720, 1280, 3), [det])

        assert mask_large.sum() > mask_small.sum()

    def test_padding_expands_polygon(self):
        """Padding should expand the mask beyond the original polygon bounds."""
        gen_no_pad = MaskGenerator(dilation_kernel=0, feather_radius=0, padding=0)
        gen_with_pad = MaskGenerator(dilation_kernel=0, feather_radius=0, padding=10)

        det = self._make_detection(100, 100, 200, 50)

        mask_no_pad = gen_no_pad.generate_mask((720, 1280, 3), [det])
        mask_with_pad = gen_with_pad.generate_mask((720, 1280, 3), [det])

        assert mask_with_pad.sum() > mask_no_pad.sum()

    def test_multiple_detections(self):
        """Multiple detections should all be present in the mask."""
        gen = MaskGenerator(dilation_kernel=0, feather_radius=0, padding=0)
        det1 = self._make_detection(100, 100, 200, 50)
        det2 = self._make_detection(500, 400, 300, 60)

        mask = gen.generate_mask((720, 1280, 3), [det1, det2])

        # Both detection centers should be white
        assert mask[125, 200] == 255
        assert mask[430, 650] == 255

    def test_mask_is_binary(self):
        """Output mask should only contain 0 and 255 values."""
        gen = MaskGenerator()
        det = self._make_detection(100, 100, 200, 50)
        mask = gen.generate_mask((720, 1280, 3), [det])

        unique_values = set(np.unique(mask))
        assert unique_values.issubset({0, 255})

    def test_mask_matches_frame_size(self):
        """Mask dimensions should match the frame dimensions."""
        gen = MaskGenerator()
        for h, w in [(480, 640), (720, 1280), (1080, 1920)]:
            mask = gen.generate_mask((h, w, 3), [])
            assert mask.shape == (h, w)
