# FFT-IA: FFT-Inspired Attention — True O(N log N) with Softmax Fidelity

**Paper**: "FFT-Inspired Attention (FFT-IA): O(N log N) Complexity via Hierarchical Structural Pruning and Softmax Fidelity" (IEEE-style, Nov 2025)

### Features
- Exact radix-2 butterfly factorization (Cooley-Tukey style)
- Dynamic Q/K re-projection per stage → content-aware
- Full local Softmax → **Softmax Fidelity preserved**
- No approximation, no kernels, no hashing
- Asymptotic **O(N log N)** in sequence length
- Works with any power-of-2 sequence length

### Install
```bash
pip install -e .
