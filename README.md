# 🈲 Remotext — Remove Chinese Text from Videos

**Remotext** is an AI-powered pipeline that automatically detects and removes hardcoded Chinese text from videos while preserving original quality, resolution, and audio.

## ✨ Features

- **Automatic Chinese text detection** using PaddleOCR (state-of-the-art Chinese OCR)
- **Temporal-aware video inpainting** using ProPainter (flow-guided, no flickering)
- **Smart keyframe optimization** — only runs OCR on scene changes, 70-90% faster
- **Quality preservation** — output matches input resolution, FPS, and audio exactly
- **Multi-device support** — CUDA (NVIDIA), MPS (Apple Silicon M1/M2/M3), or CPU
- **Configurable pipeline** — YAML config with CLI overrides
- **Graceful fallback** — ProPainter → LaMa → OpenCV Telea

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Remotext.git
cd Remotext

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Basic usage
python cli.py -i input_video.mp4 -o output_clean.mp4

# Use Apple Silicon GPU
python cli.py -i input.mp4 -o output.mp4 --device mps

# Use CUDA GPU
python cli.py -i input.mp4 -o output.mp4 --device cuda

# Use LaMa (lighter, works on weak hardware)
python cli.py -i input.mp4 -o output.mp4 --model lama

# Traditional Chinese detection
python cli.py -i input.mp4 -o output.mp4 --lang chinese_cht

# Skip Chinese text over faces
python cli.py -i input.mp4 -o output.mp4 --skip-faces
```

### Python API

```python
from src.pipeline import RemotextPipeline, PipelineConfig

config = PipelineConfig(
    device="mps",              # Apple Silicon
    primary_model="propainter",
    ocr_lang="ch",
    remove_entire_cluster=True,
)

pipeline = RemotextPipeline(config)
result = pipeline.run("input.mp4", "output.mp4")

print(f"Done in {result.total_time:.1f}s")
print(f"Removed Chinese text from {result.frames_with_chinese} frames")
```

## 🏗️ Architecture

```
Input Video → Preprocessor (FFmpeg) → Text Detector (PaddleOCR)
    → Chinese Filter (Unicode) → Mask Generator (OpenCV)
    → Video Inpainter (ProPainter) → Assembler (FFmpeg) → Output Video
```

### Pipeline Stages

| # | Stage | Tool | Purpose |
|---|-------|------|---------|
| 1 | Preprocess | FFmpeg | Extract frames + audio + metadata |
| 2 | Detect | PaddleOCR | Find all text regions (keyframe-optimized) |
| 3 | Filter | Unicode regex | Keep only Chinese text detections |
| 4 | Mask | OpenCV | Create dilated binary masks |
| 5 | Inpaint | ProPainter | Remove text with temporal consistency |
| 6 | Assemble | FFmpeg | Reconstruct video with original audio |

## ⚙️ Configuration

Edit `config/default.yaml` or pass CLI flags. Key settings:

```yaml
detector:
  keyframe_interval: 30        # OCR every N frames
  scene_change_thresh: 0.90    # SSIM threshold

mask_generator:
  dilation_kernel: 5           # Expand mask coverage
  feather_radius: 3            # Smooth mask edges

inpainter:
  primary_model: propainter    # or 'lama'
  chunk_size: 80               # Frames per batch
```

## 💻 Hardware Requirements

| Device | VRAM | Speed (60s 1080p) | Notes |
|--------|------|-------------------|-------|
| NVIDIA RTX 3060 | 12 GB | ~3 min | Best performance |
| NVIDIA T4 (Colab) | 16 GB | ~4 min | Free option |
| Apple M1 (MPS) | Unified | ~8 min | Works well, use float32 |
| CPU only | — | ~30 min | Slow, use LaMa fallback |

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 License

MIT License
