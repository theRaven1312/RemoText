"""Device detection utilities — auto-detect CUDA, MPS, or CPU."""

import logging

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> str:
    """Detect the best available compute device.

    Args:
        preference: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        Device string compatible with torch: 'cuda', 'mps', or 'cpu'.
    """
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, "total_memory", 0) / (1024**3)
            logger.info(f"Using CUDA device: {gpu_name} ({vram:.1f} GB VRAM)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = "cpu"
            logger.info("No GPU detected, falling back to CPU")
    else:
        device = preference
        logger.info(f"Using user-specified device: {device}")

    return device


def get_optimal_dtype(device: str, allow_fp16: bool = False):
    """Get optimal dtype for the device.

    Args:
        device: The compute device string.
        allow_fp16: Whether to allow half precision.

    Returns:
        torch.dtype — float16 or float32.
    """
    import torch

    # MPS has limited fp16 support, safer to use fp32
    if device == "mps" or not allow_fp16:
        return torch.float32

    if device == "cuda":
        # Check compute capability for fp16 support
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:  # Volta+
            return torch.float16

    return torch.float32


def estimate_max_chunk_size(device: str, resolution: tuple[int, int]) -> int:
    """Estimate optimal chunk size based on available memory.

    Args:
        device: Compute device string.
        resolution: (width, height) of the video.

    Returns:
        Recommended chunk size (number of frames).
    """
    import torch

    w, h = resolution
    pixels = w * h

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = getattr(props, "total_memory", 0) / (1024**3)
        if vram_gb >= 12:
            base_chunk = 120
        elif vram_gb >= 8:
            base_chunk = 80
        elif vram_gb >= 4:
            base_chunk = 40
        else:
            base_chunk = 20
    elif device == "mps":
        # M1 has unified memory; be conservative
        base_chunk = 40
    else:
        base_chunk = 20

    # Scale down for higher resolutions
    if pixels > 1920 * 1080:
        base_chunk = max(10, base_chunk // 2)
    elif pixels > 1280 * 720:
        base_chunk = max(15, int(base_chunk * 0.75))

    return base_chunk
