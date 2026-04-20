# NTK-SURGERY: Federated Unlearning via Neural Tangent Kernel Surgery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![ACCV 2024](https://img.shields.io/badge/ACCV-2024-red.svg)](https://accv2024.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
- [Datasets](#datasets)
- [Baselines](#baselines)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## 📖 Overview

**NTK-SURGERY** is a novel federated unlearning framework that shifts unlearning from **weight-space** to **function-space** using Neural Tangent Kernel (NTK) theory. Our method enables **exact client removal** with **O(1) communication rounds**, bypassing the exponential sensitivity growth that constrains existing methods like SIFU in non-convex deep learning regimes.

### Key Advantages

| Feature | NTK-SURGERY | SIFU | FedEraser | Scratch |
|---------|-------------|------|-----------|---------|
| Communication Rounds | **1** | 50-2044 | 100+ | 50+ |
| Theoretical Guarantees | ✅ Exact | ⚠️ (ε,δ) | ❌ None | ✅ Exact |
| Non-Convex Support | ✅ Yes | ❌ Degrades | ⚠️ Limited | ✅ Yes |
| Server Time (CIFAR-100) | **21.4s** | 1089.4s | 1614.8s | 2134.6s |
| Speedup vs Scratch | **100×** | 2× | 1.3× | 1× |
| Exactness Score | **0.981** | 0.821 | 0.712 | 1.000 |

---

## ⚠️ Problem Statement

### Current Federated Unlearning Limitations

Existing Federated Unlearning (FU) methods face critical deficiencies in computing viability and theoretical guarantees:

#### 1. **Sensitivity Bound Explosion (SIFU)**

SIFU relies on sensitivity bounds defined by contraction factors `B(f_I, η)`:

```
ζ(n, c) = Σ_{s=0}^{n-1} B(f_I, η)^{(n-s-1)K} · ||ω(I, θ_s)||₂
```

**Critical Issue:** In non-convex regimes typical of deep networks, `B(f_I, η) > 1` (Eq. 26, SIFU), causing **exponential growth** of ζ(n, c) with communication rounds `n`. This forces rollback to early checkpoints (`T ≈ 0`), effectively reducing SIFU to **retraining from scratch**.

#### 2. **Computational Inefficiency**

| Method | Complexity | Issue |
|--------|------------|-------|
| Scratch | O(N · K · P) | Prohibitive for large networks |
| SIFU | O(T · K · P), T→0 | Degrades to Scratch in non-convex |
| FedEraser | O(epochs · M · K · P) | Exceeds Scratch cost |
| Hessian-based | O(P³) | Intractable for deep models |

#### 3. **Theoretical Guarantee Gaps**

- **Fine-Tuning:** Parameters remain dependent on initial trajectory influenced by forgotten data (SIFU Appendix A)
- **FedEraser:** No theoretical unlearning proofs
- **Differential Privacy:** Noise magnitudes impede model convergence on complex datasets
- **Existing Methods:** Assume convex losses or independent server data, violating FL constraints

---

## 🎯 Key Contributions

### 1. **Function-Space Unlearning Framework**

We shift unlearning from weight-space temporal backtracking to **function-space structural editing** using NTK theory. This bypasses sensitivity accumulation inherent in weight-space methods.

### 2. **Four-Component Architecture**

| Component | Section | Equation | Contribution |
|-----------|---------|----------|--------------|
| Federated NTK | 4.1 | Eq. (1), (2) | Linear client influence representation |
| Influence Matrix | 4.2 | Eq. (3), (4) | Maps labels to predictions via ℐ = I_N - λG_λ |
| Surgery Operator | 4.3 | Eq. (5), (6), (7) | Closed-form client removal via Woodbury |
| Finite-Width Projection | 4.4 | Eq. (8), (9), (10) | Projects function-space to weights with O(P⁻¹/²) error |

### 3. **Theoretical Guarantees**

**Theorem 1 (Exact Unlearning):** In the NTK regime, removing client `c` via `K_global^{(-c)} = K_global - (n_c/N)S_c K_c S_c^⊤` yields predictions **identical to retraining** on `D \ D_c`.

**Complexity:** O(M²) vs SIFU's O(N · K · P), **independent of training duration**.

### 4. **Empirical Validation**

- **6 datasets:** MNIST, FashionMNIST, CIFAR-10, CIFAR-100, CelebA, TinyImageNet
- **10 baselines:** Scratch, Fine-Tuning, FedEraser, SIFU, FedSGD, BFU, Forget-SVGD, Knowledge Distillation, FU, F²L²
- **Exactness Score > 0.96** across all datasets
- **52.6× speedup** over SIFU on CIFAR-100

---

## 🔬 Methodology

### Section 4.1: Federated NTK Representation

```
Θ(x, x'; θ₀) = ⟨∇_θ f(x; θ₀), ∇_θ f(x'; θ₀)⟩ = Σ_{p=1}^{P} ∂f(x;θ₀)/∂θ_p · ∂f(x';θ₀)/∂θ_p
```

**Global Kernel Aggregation:**
```
K_global = Σ_{c=1}^{M} (n_c/N) S_c K_c S_c^⊤
```

### Section 4.2: Influence Matrix

```
ℐ = I_N - λG_λ, where G_λ = (K_global + λI_N)^{-1}
```

**Eigenvalue Properties:**
```
μ_k(ℐ) = σ_k(K_global) / (σ_k(K_global) + λ) ∈ [0, 1)
```

### Section 4.3: Surgery Operator

**Woodbury Update (Eq. 6):**
```
G_λ^{(-c)} = G_λ - G_λ S_c (-N/n_c K_c^{-1} + S_c^⊤ G_λ S_c)^{-1} S_c^⊤ G_λ
```

**Surgery Operator:**
```
ℐ^{(-c)} = ℐ + λ[G_λ S_c (S_c^⊤ G_λ S_c - N/n_c K_c^{-1})^{-1} S_c^⊤ G_λ]
```

### Section 4.4: Finite-Width Projection

**Weight Update (Eq. 9):**
```
θ_new = θ_t + J_t^⊤ G_λ^{(-c)} (ℐ^{(-c)}Y - f(X, θ_t))
```

**Error Bound (Eq. 10):**
```
||R₂|| ≤ (L_J/2) · ||J_t^⊤ G_λ^{(-c)} ΔY||² = O(P^{-1/2})
```

---

## 📊 Datasets

| Dataset | Classes | Samples | Partition | Reference |
|---------|---------|---------|-----------|-----------|
| **MNIST** | 10 | 70,000 | Label Shard | [LeCun et al., 1998](#references) |
| **FashionMNIST** | 10 | 70,000 | Label Shard | [Xiao et al., 2017](#references) |
| **CIFAR-10** | 10 | 60,000 | Dirichlet (α=0.1) | [Krizhevsky & Hinton, 2009](#references) |
| **CIFAR-100** | 100 | 60,000 | Dirichlet (α=0.1) | [Krizhevsky & Hinton, 2009](#references) |
| **CelebA** | 2 | 202,599 | Dirichlet (α=0.1) | [Liu et al., 2015](#references) |
| **TinyImageNet** | 200 | 100,000 | Dirichlet (α=0.5) | [Le & Yang, 2015](#references) |

### Federated Setup
- **Clients:** M = 100 (50 for TinyImageNet)
- **Samples per Client:** 100
- **Non-IID:** Dirichlet distribution for heterogeneous settings
- **Models:** CNN (width multiplier 4) for image datasets, MLP for tabular

---

## 🏆 Baselines

We compare against **10 state-of-the-art methods**:

| Method | Type | Theoretical Guarantees | Key Limitation |
|--------|------|----------------------|----------------|
| **Scratch (Gold)** | Retraining | ✅ Exact | O(N·K·P) complexity |
| **Fine-Tuning** | Weight Update | ❌ None | Parameter dependence on θ₀ |
| **FedEraser** | Gradient Erasure | ❌ None | Exceeds Scratch cost |
| **SIFU** | Certified Unlearning | ⚠️ (ε,δ) | Exponential sensitivity growth |
| **FedSGD** | Federated SGD | ❌ None | High communication overhead |
| **BFU** | Bayesian | ⚠️ Approximate | Scales poorly with P |
| **Forget-SVGD** | Variational | ⚠️ Approximate | Particle methods scale poorly |
| **Knowledge Distillation** | Knowledge Transfer | ❌ None | Incomplete teacher removal |
| **FU** | Generic FU | ⚠️ Convex only | Violates FL constraints |
| **F²L²** | Linear Learning | ✅ Certified | Convex losses only |

---

## 📈 Evaluation Metrics

### Unlearning Efficacy (Section 5.2)

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|----------------|
| **Forget Accuracy (FA) ↓** | (1/N_f) Σ 1[f(x_i) = y_i] | Random Chance | Lower = better unlearning |
| **Retain Accuracy (RA) ↑** | (1/N_r) Σ 1[f(x_i) = y_i] | Scratch Performance | Higher = better utility |
| **Exactness Score (ES) ↑** | 1 - \|\|f_surgery - f_scratch\|\|₂ / \|\|f_scratch\|\|₂ | 1.0 | Higher = closer to retraining |

### Efficiency Metrics

| Metric | Measurement | Target |
|--------|-------------|--------|
| **Communication Rounds ↓** | Server-client exchanges | 1 (NTK-SURGERY) |
| **Server Time (s) ↓** | Wall-clock unlearning time | Minimize |
| **FLOPs ↓** | Computational operations | O(M²) for NTK-SURGERY |
| **Speedup Factor ↑** | Time_scratch / Time_method | Maximize |

### Theoretical Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **NTK Alignment ↑** | \|\|K_cross\|\|_F / √(\|\|K_source\|\|_F · \|\|K_target\|\|_F) | > 0.95 |
| **Sensitivity Ratio ↑** | ζ_SIFU / ζ_NTK | > 100× |
| **Condition Number ↓** | κ(G_λ) | < 10⁴ |

---

## 📊 Experimental Results

### Table 1: Unlearning Efficacy Across Datasets

| Dataset | Method | FA (%) ↓ | RA (%) ↑ | ES ↑ |
|---------|--------|----------|----------|------|
| **MNIST** | Scratch | 10.2 (0.3) | 92.8 (0.4) | 1.000 |
| | Fine-Tuning | 68.4 (2.1) | 91.2 (0.5) | 0.312 |
| | FedEraser | 12.8 (0.8) | 89.7 (0.9) | 0.847 |
| | SIFU | 11.1 (0.5) | 92.1 (0.4) | 0.923 |
| | **NTK-SURGERY** | **10.4 (0.3)** | **92.7 (0.3)** | **0.996** |
| **CIFAR-10** | Scratch | 10.1 (0.5) | 78.3 (0.8) | 1.000 |
| | Fine-Tuning | 54.7 (3.2) | 75.1 (1.1) | 0.412 |
| | FedEraser | 14.8 (1.3) | 73.4 (1.4) | 0.768 |
| | SIFU | 12.3 (0.8) | 77.1 (0.9) | 0.854 |
| | **NTK-SURGERY** | **10.6 (0.5)** | **78.0 (0.7)** | **0.987** |
| **CIFAR-100** | Scratch | 1.2 (0.1) | 42.1 (1.2) | 1.000 |
| | Fine-Tuning | 18.4 (2.1) | 38.7 (1.5) | 0.364 |
| | FedEraser | 2.8 (0.4) | 37.2 (1.8) | 0.712 |
| | SIFU | 1.9 (0.3) | 40.8 (1.3) | 0.821 |
| | **NTK-SURGERY** | **1.4 (0.2)** | **41.9 (1.1)** | **0.981** |
| **CelebA** | Scratch | 50.3 (1.2) | 84.2 (0.9) | 1.000 |
| | Fine-Tuning | 78.1 (2.4) | 81.3 (1.2) | 0.298 |
| | FedEraser | 54.7 (1.8) | 79.8 (1.5) | 0.687 |
| | SIFU | 52.1 (1.4) | 83.1 (1.0) | 0.793 |
| | **NTK-SURGERY** | **50.8 (1.3)** | **84.0 (0.8)** | **0.974** |
| **TinyImageNet** | Scratch | 0.4 (0.0) | 31.2 (1.4) | 1.000 |
| | Fine-Tuning | 12.7 (1.8) | 27.4 (1.9) | 0.321 |
| | FedEraser | 1.9 (0.5) | 26.1 (2.1) | 0.634 |
| | SIFU | 1.1 (0.3) | 29.3 (1.6) | 0.742 |
| | **NTK-SURGERY** | **0.6 (0.1)** | **30.9 (1.3)** | **0.968** |

### Table 2: Efficiency Metrics (CIFAR-100)

| Method | Comm. Rounds ↓ | Server Time (s) ↓ | Speedup vs Scratch |
|--------|---------------|-------------------|-------------------|
| Scratch | 4512 (231) | 2134.6 (109.2) | 1.0× |
| Fine-Tuning | 2912 (149) | 1378.2 (70.5) | 1.5× |
| FedEraser | 3421 (176) | 1614.8 (82.6) | 1.3× |
| SIFU | 2314 (119) | 1089.4 (55.7) | 2.0× |
| **NTK-SURGERY** | **1.0 (0.0)** | **21.4 (2.2)** | **99.7×** |

### Table 3: Theoretical Metrics

| Dataset | Method | NTK Alignment ↑ | Sensitivity Ratio ↑ | Condition Number |
|---------|--------|-----------------|---------------------|------------------|
| **MNIST** | SIFU | 0.821 (0.05) | 1.00 (baseline) | 1.2×10³ |
| | **NTK-SURGERY** | **0.987 (0.02)** | **47.3 (6.2)** | 8.5×10² |
| **CIFAR-10** | SIFU | 0.687 (0.07) | 1.00 (baseline) | 3.4×10³ |
| | **NTK-SURGERY** | **0.964 (0.03)** | **128.4 (17.3)** | 2.1×10³ |
| **CIFAR-100** | SIFU | 0.643 (0.08) | 1.00 (baseline) | 5.7×10³ |
| | **NTK-SURGERY** | **0.957 (0.04)** | **142.7 (19.8)** | 3.8×10³ |

---

## 💪 Strengths of NTK-SURGERY

### 1. **Exact Unlearning Guarantees**
- **ES > 0.96** across all datasets (vs SIFU's 0.82 average)
- Functionally equivalent to retraining from scratch
- No sensitivity accumulation over training rounds

### 2. **Computational Efficiency**
- **O(1) communication rounds** (vs 50-2044 for SIFU)
- **O(M²) complexity** (vs O(N·K·P) for baselines)
- **52.6× - 99.7× speedup** over SIFU

### 3. **Non-Convex Robustness**
- Bypasses contraction factor `B(f_I, η) > 1` constraint
- Stable unlearning regardless of loss landscape curvature
- No rollback to early checkpoints required

### 4. **Theoretical Foundation**
- Proven exactness under NTK assumptions (Theorem 1)
- Error bound: O(P⁻¹/²) for finite-width projection
- Spectral stability guarantees via Woodbury identity

### 5. **Practical Deployment**
- Server-side only computation (no client communication)
- Compatible with standard FedAvg routines
- No additional data access requirements

---

## 🚀 Installation

### Requirements

```bash
Python >= 3.8
PyTorch >= 1.9.0
NumPy >= 1.19.0
SciPy >= 1.7.0
Matplotlib >= 3.4.0
Seaborn >= 0.11.0
Pandas >= 1.3.0
scikit-learn >= 0.24.0
```

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/ntk-surgery.git
cd ntk-surgery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download datasets
bash scripts/download_datasets.sh --output_dir /home/phd/datasets
```

---

## 📖 Usage

### Quick Start

```bash
# Run main experiments
python experiments/run_main.py --datasets CIFAR-10,CIFAR-100 --device cuda

# Run ablation study
python experiments/run_ablation.py --dataset CIFAR-10

# Run domain generalization
python experiments/run_domain_generalization.py

# Run hyperparameter search
python experiments/run_hyperparameter_search.py --dataset CIFAR-10

# Generate plots
bash scripts/generate_plots.sh --plot_type all --format pdf
```

### Python API

```python
from unlearning.unlearn_client import UnlearnClient, UnlearningConfig
from models.cnn import CNN, CNNConfig

# Initialize model
config = CNNConfig(input_channels=3, num_classes=10, width_multiplier=4)
model = CNN(config)

# Initialize unlearner
unlearning_config = UnlearningConfig(
    lambda_reg=0.05,
    width_multiplier=4,
    device='cuda'
)
unlearner = UnlearnClient(model, unlearning_config)
unlearner.set_client_partitions(client_indices)

# Unlearn client
result = unlearner.unlearn(client_id=0, X=X_global, y=y_global)

print(f"Exactness Score: {result.exactness_score:.4f}")
print(f"Unlearning Time: {result.unlearning_time:.2f}s")
print(f"Communication Rounds: {result.communication_rounds}")
```

### Run Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test module
python tests/test_ntk_surgery.py
python tests/test_baselines.py
python tests/test_metrics.py
python tests/test_data_loader.py
```

---

## 📁 Repository Structure

```
ntk-surgery/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── config/                      # Configuration files
│   ├── default_config.yaml
│   ├── dataset_config.yaml
│   └── model_config.yaml
├── data/                        # Data loading utilities
│   ├── __init__.py
│   ├── data_loader.py
│   ├── dataset_utils.py
│   ├── federated_partition.py
│   └── preprocessor.py
├── models/                      # Neural network architectures
│   ├── __init__.py
│   ├── ntk_model.py
│   ├── cnn.py
│   ├── mlp.py
│   └── resnet.py
├── ntk_surgery/                 # Core NTK-SURGERY implementation
│   ├── __init__.py
│   ├── federated_ntk.py         # Section 4.1
│   ├── influence_matrix.py      # Section 4.2
│   ├── surgery_operator.py      # Section 4.3
│   └── finite_width_projection.py  # Section 4.4
├── baselines/                   # Baseline method implementations
│   ├── __init__.py
│   ├── sifu.py
│   ├── federaser.py
│   ├── fine_tuning.py
│   ├── fedsgd.py
│   ├── bfu.py
│   ├── forget_svgd.py
│   ├── knowledge_distillation.py
│   ├── fu.py
│   └── f2l2.py
├── metrics/                     # Evaluation metrics
│   ├── __init__.py
│   ├── unlearning_metrics.py
│   ├── efficiency_metrics.py
│   └── theoretical_metrics.py
├── training/                    # Federated training
│   ├── __init__.py
│   ├── fedavg.py
│   ├── local_training.py
│   └── server_aggregation.py
├── unlearning/                  # Unlearning orchestration
│   ├── __init__.py
│   ├── unlearn_client.py
│   └── unlearn_evaluator.py
├── experiments/                 # Experiment runners
│   ├── __init__.py
│   ├── run_main.py
│   ├── run_ablation.py
│   ├── run_domain_generalization.py
│   └── run_hyperparameter_search.py
├── visualization/               # Plot generation
│   ├── __init__.py
│   ├── plot_efficacy.py
│   ├── plot_efficiency.py
│   ├── plot_theoretical.py
│   └── plot_ablation.py
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── logger.py
│   ├── checkpoint.py
│   ├── random_seed.py
│   └── memory_tracker.py
├── scripts/                     # Shell scripts
│   ├── download_datasets.sh
│   ├── run_experiments.sh
│   └── generate_plots.sh
└── tests/                       # Test suite
    ├── __init__.py
    ├── test_ntk_surgery.py
    ├── test_baselines.py
    ├── test_metrics.py
    └── test_data_loader.py
```

---

## 📚 References

### Datasets

```bibtex
@inproceedings{lecun1998mnist,
  title={The MNIST database of handwritten digits},
  author={LeCun, Yann and Cortes, Corinna and Burges, Christopher J. C.},
  year={1998},
  publisher={AT\&T Labs-Research}
}

@article{xiao2017fashionmnist,
  title={Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms},
  author={Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  journal={arXiv preprint arXiv:1708.07747},
  year={2017}
}

@techreport{krizhevsky2009cifar,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  institution={University of Toronto},
  year={2009}
}

@inproceedings{liu2015celeba,
  title={Deep learning face attributes in the wild},
  author={Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={ICCV},
  year={2015}
}

@misc{le2015tinyimagenet,
  title={Tiny ImageNet visual recognition challenge},
  author={Le, Ya and Yang, Xuan},
  year={2015},
  publisher={Stanford University}
}
```

### Baselines

```bibtex
@inproceedings{fraboni2024sifu,
  title={SIFU: Sequential Informed Federated Unlearning},
  author={Fraboni, Yuri and others},
  booktitle={AISTATS},
  year={2024}
}

@inproceedings{liu2021federaser,
  title={Federaser: Enabling Efficient Client-Level Data Removal from Federated Learning Models},
  author={Liu, Gaoyang and others},
  booktitle={IWQoS},
  year={2021}
}

@inproceedings{jin2023forgettable,
  title={Forgettable Federated Linear Learning with Certified Data Removal},
  author={Jin, Rong and others},
  journal={arXiv preprint arXiv:2306.02216},
  year={2023}
}

@inproceedings{liu2022right,
  title={The Right to Be Forgotten in Federated Learning},
  author={Liu, Yixin and others},
  booktitle={INFOCOM},
  year={2022}
}
```

### NTK Theory

```bibtex
@inproceedings{jacot2018neural,
  title={Neural Tangent Kernel: Convergence and Generalization in Neural Networks},
  author={Jacot, Arthur and Gabriel, Franck and Hongler, Clément},
  booktitle={NeurIPS},
  year={2018}
}

@inproceedings{wang2020tackling,
  title={Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization},
  author={Wang, Jianyu and others},
  booktitle={NeurIPS},
  year={2020}
}
```

---

## 📄 Citation

If you use NTK-SURGERY in your research, please cite:

```bibtex
@inproceedings{ntk-surgery2024,
  title={NTK-SURGERY: Federated Unlearning via Neural Tangent Kernel Surgery},
  author={Saeed Iqbal, Xiaopin Zhong, Muhammad Attique Khan, Zongze Wu, Weixiang Liu, Mohammad Alhefdi, and Amir Hussain},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2024}
}
```

---

## ⚖️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the authors directly.

---

**Last Updated:** April 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✅
