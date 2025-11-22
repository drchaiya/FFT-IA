# fft_ia/__init__.py
from .core import FFTInspiredAttention
from .butterfly import get_all_butterfly_indices

__version__ = "0.1.0"
__all__ = ["FFTInspiredAttention", "get_all_butterfly_indices"]

# Auto-detect best backend
try:
    from .fused_kernel import FFTInspiredAttentionFused
    FFTInspiredAttention = FFTInspiredAttentionFused  # override with fused version
    print("FFT-IA: Triton fused kernel loaded — peak performance")
except Exception as e:
    print("FFT-IA: Using pure PyTorch implementation (install Triton for max speed)")
