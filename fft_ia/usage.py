# option 1
from fft_ia import FFTInspiredAttention
from fft_ia.utils import pad_to_power_of_2, unpad

layer = FFTInspiredAttention(dim=512, heads=8)

x = torch.randn(1, 3715, 512)           # any length
x_pad, n_pad = pad_to_power_of_2(x)     # → 4096
out_pad = layer(x_pad)
out = unpad(out_pad, 3715)              # back to original

# option 2
from fft_ia import FFTInspiredAttention

# Replace any nn.Transformer or Llama attention layer
layer = FFTInspiredAttention(dim=4096, heads=32)

x = torch.randn(1, 12345, 4096)   # any length
out = layer(x)                    # auto-pads → 16384 → butterfly → auto-unpads
print(out.shape)                  # → [1, 12345, 4096]
