# 🚀 FFT-Inspired Attention (FFT-IA): $\mathcal{O}(N \log N)$ Scalable Transformers

A novel theoretical and architectural framework to overcome the $\mathcal{O}(N^2)$ complexity bottleneck of the Multi-Head Self-Attention (MHSA) mechanism in Transformers, achieving an $\mathbf{\mathcal{O}(N \log N)}$ asymptotic complexity via **fixed structural factorization** inspired by the Cooley-Tukey algorithm.

---

## 🌟 Core Contributions

1.  **$\mathbf{\mathcal{O}(N \log N)}$ Asymptotic Complexity:** Achieved through a **fixed structural factorization** of the attention matrix, systematically decomposing the dense $\mathcal{O}(N^2)$ space into a cascade of $\mathbf{\log_2 N}$ local, $\mathcal{O}(N)$ operations.
2.  **Structural Pruning, Not Approximation:** Efficiency is gained via **fixed structural pruning** based on the radix-2 butterfly connection pattern, not by approximating the attention function or substituting it entirely.
3.  **Softmax Fidelity:** The mechanism computes **exact attention scores** within its local, sparse scope and retains the essential Softmax non-linearity. The local Softmax acts as a **normalized adaptive pooling step** over the connected tokens.
4.  **Content Dynamism:** Contextual awareness is maintained by **dynamically re-projecting $Q$ and $K$ from the intermediate state** at every sequential stage, enabling content-dependent scoring despite the fixed connectivity graph.

---

## 🛠️ Architecture: The Hierarchical Butterfly-Attention Block

The dense $QK^\top$ computation is replaced by $L = \log_2 N$ sequential sparse projection factors $P_i$.

### 1. Sequential Transformation Flow

The input value vector $V=V_0$ is sequentially transformed across $\log_2 N$ stages:

$$
V_i = P_i \cdot V_{i-1}
$$

Where $P_i$ is the sparse attention factor for stage $i$.

### 2. Dynamic Q/K Re-Projection

To ensure dynamism, Query $Q$ and Key $K$ are re-projected from the intermediate state $V_{i-1}$ at each stage:

$$
Q_i = W_{Q, i} V_{i-1}, \quad K_i = W_{K, i} V_{i-1}
$$

### 3. Butterfly-Attention Block (BAB) 

Each stage uses the **Butterfly-Attention Block (BAB)**, a fixed, radix-2 cyclic connection pattern with stride $2^{i-1}$.

* **Fixed Constraint:** The attention is restricted to a local set of tokens $\mathcal{C}_j$ for token $j$:
    $$
    \mathcal{C}_j = \{k \mid k = j \quad \text{or} \quad k = j \pm 2^{i-1}\}
    $$
* **Local Softmax:** The attention score is computed *exactly* and normalized only over the connected set $\mathcal{C}_j$:
    $$
    P_i[j, k] = \text{softmax}_{k' \in \mathcal{C}_j} \left(\frac{Q_{i, j} K_{i, k'}^\top}{\sqrt{d_k}}\right) \quad \text{if } k \in \mathcal{C}_j
    $$

---

## 📈 Complexity Analysis

The total asymptotic complexity in sequence length $N$ is guaranteed by the fixed structure:

$$\mathbf{\mathcal{O}(N \log N)}$$

The total FLOPs cost includes the overhead of repeated Q/K re-projection:

$$
\text{FLOPs} = \mathbf{\mathcal{O}(N \cdot (\log N) \cdot (d^2 + d_k))}
$$

### 🔑 The Paramount Challenge: Kernel Fusion

The practical realization of wall-clock speedup for $\text{FFT}$-$\text{IA}$ is contingent upon dedicated **Kernel Fusion** for the $\log_2 N$ sequential, irregular operations. This is necessary to eliminate the overhead of repeated kernel launches and efficiently handle the $\mathbf{O(N d^2 \log N)}$ complexity dominated by re-projection.

---

## 📚 Distinctions from Prior Work

| Method Category | Example | $\text{FFT}$-$\text{IA}$ Distinction |
| :--- | :--- | :--- |
| **Approximation** | Reformer (Hashing/Kernels) | Computes **exact** scores locally, maintains Softmax Fidelity. |
| **FFT-Substitution** | FNet | Retains **dynamic, content-dependent** $QK^\top$ scoring via re-projection. |
| **Dynamic Sparse** | Longformer, Sparse Transformer | Fixed, **theoretically guaranteed** $\mathcal{O}(N \log N)$ complexity via structural constraint, not learned or heuristic patterns. |

---

## 💡 Structural Inductive Bias Hypothesis

The fixed, hierarchical butterfly structure acts as a powerful **structural regularization** mechanism, hypothesized to:

* **Mitigate Spurious Correlations:** Forcing the model to establish long-range dependencies through a compositional cascade, rather than direct, potentially overfit, global links.
* **Promote Compositional Processing:** Structurally restricting the model's capacity for simple associative memory retrieval, encouraging aggregation.

---

## 📍 Getting Started (Future Work)

The next steps for this project involve:

1.  **CUDA/Triton Kernel Development:** Implementing a highly optimized, fused custom kernel for the $\log_2 N$ sequential stages to achieve practical wall-clock speedup.
2.  **Empirical Validation:** Training $\text{FFT}$-$\text{IA}$ models on standard benchmarks (e.g., NLP, vision) to validate speed and accuracy.
3.  **Ablation Studies:** Quantifying the benefits of the structural inductive bias against baseline Transformers.
