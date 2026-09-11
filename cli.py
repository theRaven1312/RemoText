#!/usr/bin/env python3
"""Remotext CLI — Remove Chinese text from videos.

Usage:
    python cli.py --input video.mp4 --output clean.mp4
    python cli.py --input video.mp4 --output clean.mp4 --config config/default.yaml
    python cli.py --input video.mp4 --output clean.mp4 --device mps --verbose
"""

import argparse
import logging
import os
import sys

# Prevent PaddlePaddle 3.x PIR-to-oneDNN NotImplementedError
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from src.pipeline import PipelineConfig, RemotextPipeline


def setup_logging(verbose: bool = True):
    """Configure logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remotext — Remove Chinese text from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py -i video.mp4 -o clean.mp4

  # Use custom config
  python cli.py -i video.mp4 -o clean.mp4 --config config/default.yaml

  # Force CPU processing
  python cli.py -i video.mp4 -o clean.mp4 --device cpu

  # Use LaMa instead of ProPainter
  python cli.py -i video.mp4 -o clean.mp4 --model lama

  # Keep temporary files for debugging
  python cli.py -i video.mp4 -o clean.mp4 --keep-temp
        """,
    )

    # Required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path for the output video file",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: built-in defaults)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--model",
        choices=["propainter", "lama"],
        default="propainter",
        help="Primary inpainting model (default: propainter)",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="OCR language: 'ch' (Simplified) or 'chinese_cht' (Traditional)",
    )
    parser.add_argument(
        "--crf",
        default="auto",
        help="Video quality CRF (0-51, lower=better, 'auto'=match input)",
    )
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        default="slow",
        help="Encoding speed preset (default: slow)",
    )

    # Tuning arguments
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=None,
        help="Force OCR every N frames (default: 10, lower = more sensitive to fast text)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum OCR confidence threshold (default: 0.35, lower = catch faint text)",
    )

    # Flags
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after processing",
    )
    parser.add_argument(
        "--skip-faces",
        action="store_true",
        help="Skip Chinese text overlapping face regions",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress (default: True)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    setup_logging(verbose=not args.quiet)

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Build config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # Override config with CLI arguments
    config.device = args.device
    config.primary_model = args.model
    config.ocr_lang = args.lang
    config.crf = int(args.crf) if args.crf.isdigit() else args.crf
    config.preset = args.preset
    config.keep_temp = args.keep_temp
    config.skip_face_regions = args.skip_faces
    config.verbose = not args.quiet

    if args.keyframe_interval is not None:
        config.keyframe_interval = args.keyframe_interval
    if args.min_confidence is not None:
        config.min_confidence = args.min_confidence
        config.rec_score_thresh = args.min_confidence

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run pipeline
    pipeline = RemotextPipeline(config)

    try:
        result = pipeline.run(args.input, args.output)

        print(f"\n{'=' * 50}")
        print(f"✅ Success!")
        print(f"   Output: {result.output_path}")
        print(f"   Time:   {result.total_time:.1f}s")
        print(f"   Frames: {result.frames_processed} total")
        print(f"   Chinese text found in: {result.frames_with_chinese} frames")
        print(f"   Inpainted: {result.frames_inpainted} frames")
        print(f"   Model:  {result.model_used}")
        print(f"{'=' * 50}")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
