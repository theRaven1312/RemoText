"""Module 7: Pipeline Orchestrator — Ties all modules together into a complete workflow."""

import logging
import os
import time
from dataclasses import dataclass, field

import yaml

from .assembler import VideoAssembler
from .detector import TextDetector
from .filter import ChineseTextFilter
from .inpainter import VideoInpainter
from .mask_generator import MaskGenerator
from .preprocessor import VideoPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the entire Remotext pipeline."""

    # Preprocessor
    frame_format: str = "png"
    temp_dir: str | None = None

    # Detector
    ocr_lang: str = "ch"
    use_angle_cls: bool = False
    det_db_thresh: float = 0.25
    rec_score_thresh: float = 0.35
    keyframe_interval: int = 10
    scene_change_thresh: float = 0.90

    # Filter
    remove_entire_cluster: bool = True
    remove_logos_watermarks: bool = True
    min_confidence: float = 0.35
    min_text_size: float = 10
    max_text_size: float | None = None

    # Mask
    dilation_kernel: int = 3
    dilation_iterations: int = 1
    feather_radius: int = 2
    mask_padding: int = 2

    # Inpainter
    primary_model: str = "propainter"
    fallback_model: str = "lama"
    device: str = "auto"
    chunk_size: int = 80
    chunk_overlap: int = 10
    fp16: bool = False

    # Assembler
    codec: str = "libx264"
    crf: int | str = "auto"
    preset: str = "slow"
    pixel_format: str = "yuv420p"
    audio_copy: bool = True

    # Edge cases
    skip_face_regions: bool = False

    # Pipeline
    verbose: bool = True
    keep_temp: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML config file.

        Returns:
            PipelineConfig instance.
        """
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        config = cls()

        # Map YAML sections to flat config attributes
        mappings = {
            "preprocessor": {"frame_format": "frame_format", "temp_dir": "temp_dir"},
            "detector": {
                "lang": "ocr_lang",
                "use_angle_cls": "use_angle_cls",
                "det_db_thresh": "det_db_thresh",
                "rec_score_thresh": "rec_score_thresh",
                "keyframe_interval": "keyframe_interval",
                "scene_change_thresh": "scene_change_thresh",
            },
            "filter": {
                "remove_entire_cluster": "remove_entire_cluster",
                "remove_logos_watermarks": "remove_logos_watermarks",
                "min_confidence": "min_confidence",
            },
            "mask_generator": {
                "dilation_kernel": "dilation_kernel",
                "dilation_iterations": "dilation_iterations",
                "feather_radius": "feather_radius",
                "padding": "mask_padding",
            },
            "inpainter": {
                "primary_model": "primary_model",
                "fallback_model": "fallback_model",
            },
            "assembler": {
                "codec": "codec",
                "crf": "crf",
                "preset": "preset",
                "pixel_format": "pixel_format",
                "audio_copy": "audio_copy",
            },
            "edge_cases": {
                "skip_face_regions": "skip_face_regions",
                "min_text_size": "min_text_size",
                "max_text_size": "max_text_size",
            },
            "pipeline": {
                "device": "device",
                "verbose": "verbose",
                "keep_temp": "keep_temp",
            },
        }

        for section, fields in mappings.items():
            section_data = raw.get(section, {})
            if section_data:
                for yaml_key, attr_name in fields.items():
                    if yaml_key in section_data and section_data[yaml_key] is not None:
                        setattr(config, attr_name, section_data[yaml_key])

        # Handle nested inpainter settings
        inpainter_data = raw.get("inpainter", {})
        pp_data = inpainter_data.get("propainter", {})
        if pp_data:
            config.chunk_size = pp_data.get("chunk_size", config.chunk_size)
            config.chunk_overlap = pp_data.get("chunk_overlap", config.chunk_overlap)
            config.fp16 = pp_data.get("fp16", config.fp16)

        return config


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""

    output_path: str
    total_time: float        # Total processing time in seconds
    frames_processed: int
    frames_with_chinese: int
    frames_inpainted: int
    model_used: str
    stages: dict[str, float] = field(default_factory=dict)  # Stage name → time


class RemotextPipeline:
    """Main pipeline orchestrating all processing stages.

    Usage:
        ```python
        pipeline = RemotextPipeline()
        result = pipeline.run("input.mp4", "output.mp4")
        print(f"Done in {result.total_time:.1f}s")
        ```

    Attributes:
        config: PipelineConfig with all settings.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        # Initialize modules
        self.preprocessor = VideoPreprocessor(
            frame_format=self.config.frame_format,
            temp_dir=self.config.temp_dir,
        )
        self.detector = TextDetector(
            lang=self.config.ocr_lang,
            use_angle_cls=self.config.use_angle_cls,
            det_db_thresh=self.config.det_db_thresh,
            rec_score_thresh=self.config.rec_score_thresh,
            keyframe_interval=self.config.keyframe_interval,
            scene_change_thresh=self.config.scene_change_thresh,
        )
        self.filter = ChineseTextFilter(
            remove_entire_cluster=self.config.remove_entire_cluster,
            remove_logos_watermarks=self.config.remove_logos_watermarks,
            min_confidence=self.config.min_confidence,
            min_text_size=self.config.min_text_size,
            max_text_size=self.config.max_text_size,
        )
        self.mask_gen = MaskGenerator(
            dilation_kernel=self.config.dilation_kernel,
            dilation_iterations=self.config.dilation_iterations,
            feather_radius=self.config.feather_radius,
            padding=self.config.mask_padding,
        )
        self.inpainter = VideoInpainter(
            primary_model=self.config.primary_model,
            fallback_model=self.config.fallback_model,
            device=self.config.device,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            fp16=self.config.fp16,
        )
        self.assembler = VideoAssembler(
            codec=self.config.codec,
            crf=self.config.crf,
            preset=self.config.preset,
            pixel_format=self.config.pixel_format,
            audio_copy=self.config.audio_copy,
        )

    def run(self, input_path: str, output_path: str) -> PipelineResult:
        """Run the complete text removal pipeline.

        Args:
            input_path: Path to the input video.
            output_path: Path for the output video.

        Returns:
            PipelineResult with timing and statistics.

        Raises:
            FileNotFoundError: If input video doesn't exist.
            RuntimeError: If any pipeline stage fails critically.
        """
        total_start = time.time()
        stages: dict[str, float] = {}
        verbose = self.config.verbose

        logger.info("=" * 60)
        logger.info(f"Remotext Pipeline v{__import__('src').__version__}")
        logger.info(f"Input:  {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)

        # ── Stage 1: Preprocess ─────────────────────────────────────
        t0 = time.time()
        logger.info("\n📹 Stage 1/6: Preprocessing video...")
        preprocess = self.preprocessor.extract(input_path)
        stages["preprocess"] = time.time() - t0
        logger.info(f"  ✓ {len(preprocess.frame_paths)} frames extracted ({stages['preprocess']:.1f}s)")

        try:
            # ── Stage 2: Detect Text ────────────────────────────────
            t0 = time.time()
            logger.info("\n🔍 Stage 2/6: Detecting text...")
            all_detections = self.detector.detect_batch(
                preprocess.frame_paths, verbose=verbose
            )
            stages["detect"] = time.time() - t0
            logger.info(f"  ✓ Detection complete ({stages['detect']:.1f}s)")

            # ── Stage 3: Filter Chinese ─────────────────────────────
            t0 = time.time()
            logger.info("\n🈲 Stage 3/6: Filtering Chinese text...")
            chinese_detections = self.filter.filter_detections(all_detections)
            stages["filter"] = time.time() - t0

            frames_with_chinese = sum(
                1 for fd in chinese_detections.values() if fd.detections
            )
            logger.info(
                f"  ✓ {frames_with_chinese} frames contain Chinese text ({stages['filter']:.1f}s)"
            )

            # ── Stage 4: Generate Masks ─────────────────────────────
            t0 = time.time()
            logger.info("\n🎭 Stage 4/6: Generating masks...")
            masks_dir = os.path.join(preprocess.temp_dir, "masks")
            mask_result = self.mask_gen.generate_batch(
                preprocess.frame_paths, chinese_detections, masks_dir, verbose=verbose
            )
            stages["mask"] = time.time() - t0
            logger.info(
                f"  ✓ {mask_result.total_masked_frames} masks generated ({stages['mask']:.1f}s)"
            )

            # ── Stage 5: Inpaint ────────────────────────────────────
            t0 = time.time()
            logger.info("\n🎨 Stage 5/6: Inpainting...")
            inpainted_dir = os.path.join(preprocess.temp_dir, "inpainted")
            inpaint_result = self.inpainter.inpaint(
                preprocess.frame_paths,
                mask_result,
                inpainted_dir,
                resolution=preprocess.metadata.resolution,
                verbose=verbose,
            )
            stages["inpaint"] = time.time() - t0
            logger.info(
                f"  ✓ {inpaint_result.frames_inpainted} frames inpainted "
                f"using {inpaint_result.model_used} ({stages['inpaint']:.1f}s)"
            )

            # ── Stage 6: Assemble Video ─────────────────────────────
            t0 = time.time()
            logger.info("\n📦 Stage 6/6: Assembling output video...")
            self.assembler.assemble(
                frames_dir=inpainted_dir,
                frame_format=self.config.frame_format,
                audio_path=preprocess.audio_path,
                metadata=preprocess.metadata,
                output_path=output_path,
            )
            stages["assemble"] = time.time() - t0
            logger.info(f"  ✓ Video assembled ({stages['assemble']:.1f}s)")

        finally:
            # Cleanup temp files
            if not self.config.keep_temp:
                self.preprocessor.cleanup(preprocess.temp_dir)

        total_time = time.time() - total_start

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ Pipeline complete!")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Output: {output_path}")
        for stage, t in stages.items():
            logger.info(f"  {stage:>12}: {t:.1f}s ({t / total_time * 100:.0f}%)")
        logger.info("=" * 60)

        return PipelineResult(
            output_path=output_path,
            total_time=total_time,
            frames_processed=len(preprocess.frame_paths),
            frames_with_chinese=frames_with_chinese,
            frames_inpainted=inpaint_result.frames_inpainted,
            model_used=inpaint_result.model_used,
            stages=stages,
        )
