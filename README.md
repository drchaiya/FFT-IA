# FFT-Inspired Attention (\text{FFT}-\text{IA}): The O(N \log N) Transformer
Achieving Sub-Quadratic Complexity with Full Softmax Fidelity

🚀 FFT-Inspired Attention (\text{FFT}-\text{IA}): The O(N \log N) Transformer
Achieving Sub-Quadratic Complexity with Full Softmax Fidelity
This repository introduces the Fast Fourier Transform-Inspired Attention (\text{FFT}-\text{IA}) framework, a novel theoretical solution that fundamentally solves the O(N^2) complexity bottleneck of the Multi-Head Self-Attention (MHSA) mechanism. By adopting a fixed, hierarchical structural factorization inspired by the Cooley-Tukey FFT algorithm, we mathematically enforce an \mathbf{O(N \log N)} asymptotic complexity in sequence length N.
We are seeking collaborators with expertise in low-level kernel optimization (CUDA/Triton) to build the Proof-of-Concept (POC) and realize the full wall-clock speedup potential.
💡 The Innovation: Structural Factorization
Standard MHSA requires an O(N^2) dense interaction matrix (QK^\top). \text{FFT}-\text{IA} replaces this single, global operation with a cascade of \mathbf{L=\log_2 N} sequential, sparse operations.
Core Mechanism: The Butterfly-Attention Block
The dense attention computation is factored into \log_2 N stages. Each stage uses a fixed, radix-2 "butterfly" connectivity pattern—the Butterfly-Attention Block—which performs a localized, O(1) interaction per token.
How the Cascade Works:
 * The sequence V_{0} is iteratively transformed: V_{i} = P_{i} \cdot V_{i-1}.
 * Each factor P_i is a fixed, sparse matrix enforcing a specific stride (2^{i-1}), guaranteeing an \mathbf{O(N)} operation per stage.
 * The composition of \log_2 N stages ensures all tokens interact hierarchically, achieving a global receptive field at an \mathbf{O(N \log N)} total complexity.
Key Feature 1: Asymptotic \mathbf{O(N \log N)} Complexity
The total theoretical FLOPs cost is reduced by over \mathbf{60\%} for sequence lengths N > 2048, shifting the Transformer's efficiency frontier.
Key Feature 2: Softmax Fidelity
Unlike many approximation methods that sacrifice the Softmax non-linearity, \text{FFT}-\text{IA} retains it:
 * We compute the exact attention score \left(\frac{Q K^\top}{\sqrt{d_k}}\right) for every connected pair of tokens.
 * The Softmax is applied locally over the two connected tokens (\mathcal{C}_j), acting as a normalized, adaptive pooling step.
 * Dynamism is retained by re-projecting Q and K from the intermediate state V_{i-1} at every stage.
🛠️ The Collaboration Ask: Kernel Fusion
While the theoretical complexity is \mathbf{O(N \log N)}, the total computational cost is \mathbf{O(N d^2 \log N)} due to the overhead of \log_2 N sequential Q/K re-projections (Equation 3 in the paper).
This overhead is the single barrier to practical wall-clock speedup.
We urgently seek collaborators to address the Paramount Technical Challenge:
Goal: Implement a Fused Hierarchical Attention Kernel
The project needs a dedicated, custom kernel (e.g., CUDA or Triton) that can perform the entire \log_2 N sequential Butterfly-Attention stages in a single fused operation. This will eliminate the kernel launch overhead and allow us to realize the massive wall-clock speedup promised by the \mathbf{O(N \log N)} complexity.
Next Steps for POC
| Task | Description | Status | Priority |
|---|---|---|---|
| Model Skeleton | Implement the \text{FFT}-\text{IA} Layer in a standard framework (PyTorch/TF) for functional verification. | Pending | High |
| Custom Kernel | Develop the high-performance, fused CUDA/Triton kernel for the \log_2 N stages. | Critical Need | URGENT |
| Complexity Benchmarks | Scripts to validate O(N \log N) wall-clock scaling empirically. | Pending | High |
| Empirical Validation | Train the model on standard sequence tasks (e.g., Long Range Arena). | Pending | Medium |
📄 Read the Full Paper
For complete mathematical details, theoretical projections, and the full defense of Softmax Fidelity, please refer to the pre-print:
[Link to Paper PDF (The IEEE Document)]
paper/FFT-IA_Paper.pdf
Citation
If you reference this work, please use the following BibTeX entry:
@article{fftia2025,
  title={{FFT-Inspired Attention (FFT-IA): $\mathbf{O(N \log N)}$ Complexity via Hierarchical Structural Pruning and Softmax Fidelity}},
  author={Tantisukarom},
  journal={IEEE Transactions on Parallel and Distributed Systems, Submitted November 2025},
  year={2025},
}

🙏 Contribution
We welcome contributions from researchers and engineers specializing in model architecture, low-level GPU programming, and efficient deep learning systems. Please check the POC_SPEC.md (to be created) for detailed requirements or open an Issue to discuss your potential contribution.
