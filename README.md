# Research on Improving NeRF Training Quality Using Depth Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


## 👥 **Participants**

**Ayan Ali**  
ITMO University, Faculty of Information Technology and Programming  
Master's Program: Robotics and Artificial Intelligence (2024-2026)  
Email: ayansaleem827@gmail.com  
GitHub: [@ayanali827](https://github.com/ayanali827)

**Urwa**  
ITMO University, Faculty of Information Technology and Programming  
Master's Program: Robotics and Artificial Intelligence (2024-2026)  
Email: urwa.mughal7@gmail.com
GitHub: [@UrwaMughal7](https://github.com/UrwaMughal7)

**Umer Ahmed Baig Mughal**  
ITMO University, Faculty of Information Technology and Programming  
Master's Program: Robotics and Artificial Intelligence (2024-2026)  
Email: umerahmedbaig98@gmail.com
GitHub: [umerahmedbaig7](https://github.com/umerahmedbaig7)

---

## 🎯 **Research Objective**

**Compare strategies for using depth data to improve Neural Radiance Fields (NeRF) training quality.**

Specifically, this research investigates:

1. **Hard Constraint Strategy** - Using depth as a hard constraint for ray sampling
2. **Soft Constraint Strategy** - Using depth as a hint in the loss function  
3. **Hybrid Approach** - Combining both strategies

**Goal:** Evaluate the effectiveness of each approach on a common baseline, both individually and in hybrid schemes, assessing improvements in **geometry accuracy** and **rendering realism**.

---

## 📋 **Research Description**

### Problem Statement

Neural Radiance Fields (NeRF) excel at novel view synthesis but suffer from:
- **Geometric ambiguity** - Multiple 3D configurations can explain the same 2D image
- **Depth uncertainty** - Density can be distributed across empty space ("floaters")
- **Surface imprecision** - Volume rendering spreads density instead of sharp surfaces

### Proposed Solution

**Leverage depth supervision** from depth sensors (LiDAR, stereo cameras, depth cameras) or pre-trained depth estimators to constrain the 3D geometry during NeRF training.

### Research Questions

1. **How effective is depth-guided ray sampling (hard constraint)?**
   - Concentrating samples near known surfaces
   - Penalizing density in free space

2. **How effective is depth loss supervision (soft constraint)?**
   - Direct MSE loss on predicted depth
   - Regularization via depth prior

3. **Does combining both strategies (hybrid) outperform individual approaches?**
   - Balancing exploration (uniform) and exploitation (guided)
   - Joint optimization of RGB and depth

4. **What are the trade-offs?**
   - Rendering quality (PSNR, SSIM)
   - Geometric accuracy (depth L1/L2 error)
   - Training time and computational cost

---

## 🔬 **Methodology**

### Experimental Design

We implement and compare **four training strategies** on the NeRF Synthetic dataset:

#### 1. **Baseline NeRF** (No Depth)

**Training:**
- Standard volume rendering
- RGB photometric loss only: `L = ||C_pred - C_gt||²`
- Stratified sampling: uniform distribution along rays

**Purpose:** Establish baseline performance without depth supervision

---

#### 2. **Soft Depth Supervision** (Depth as Hint)

**Training:**
- Add depth loss to objective: `L = L_RGB + λ_soft · ||D_pred - D_gt||²`
- Same stratified sampling as baseline
- Backprop through volume rendering to depth prediction

**Theory:**
```
D_pred = Σ w_i · t_i  (expected depth)
where w_i = T_i · α_i (accumulated weights)
```

**Hyperparameter:** λ_soft = 0.01

**Purpose:** Test if depth loss alone improves geometry

---

#### 3. **Hard Depth-Guided Sampling** (Depth as Constraint)

**Training:**
- **Guided sampling:** 75% samples in [D_gt - ε, D_gt + ε] window
- **Free-space loss:** Penalize density before surface: `L_free = Σ σ(t) for t < D_gt`
- **Surface concentration loss:** Encourage sharp weights: `L_surf = |D_pred - D_gt| + λ·Var(w)`

**Total Loss:**
```
L = L_RGB + λ_hard · (L_free + L_surf)
```

**Hyperparameter:** λ_hard = 0.005, ε = 0.3m

**Purpose:** Test if sampling guidance improves geometry

---

#### 4. **Hybrid Strategy** (Combined Approach)

**Training:**
- **Sampling:** 50% uniform + 50% depth-guided
- **Loss:** Combines soft + hard components

```
L = L_RGB + λ_soft · L_depth_MSE + λ_hard · (L_free + L_surf)
```

**Purpose:** Test if combining strategies yields best results

---

### Evaluation Metrics

**Rendering Quality:**
- **PSNR** (Peak Signal-to-Noise Ratio) - Higher is better
- **SSIM** (Structural Similarity Index) - Higher is better  
- **LPIPS** (Learned Perceptual Image Patch Similarity) - Lower is better

**Geometric Accuracy:**
- **Depth L1 Error** - Mean absolute error: `|D_pred - D_gt|`
- **Depth L2 Error** - Root mean squared error: `√((D_pred - D_gt)²)`
- **Surface Sharpness** - Variance of weight distribution

**Efficiency:**
- Training time (hours)
- GPU memory usage (GB)
- Inference speed (frames/sec)

---

### Theoretical Background

**Neural Radiance Fields** represent scenes as continuous functions:

```
F_θ: (x, d) → (RGB, σ)
```

Where:
- **x** = 3D position
- **d** = viewing direction
- **RGB** = emitted color
- **σ** = volume density

**Volume Rendering Equation:**

```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt

where T(t) = exp(-∫ σ(r(s)) ds)  (transmittance)
```

**Our Depth Supervision Extensions:**

1. **Soft Constraint:** Direct supervision on expected depth
   ```
   L_soft = ||Σ w_i · t_i - D_gt||²
   ```

2. **Hard Constraint:** Guided sampling + geometric losses
   ```
   - Sample 75% near D_gt ± ε
   - Penalize σ where t < D_gt (free space)
   - Minimize Var(w) (sharp surface)
   ```

3. **Hybrid:** Combined optimization
   ```
   L = L_RGB + λ_soft·L_soft + λ_hard·(L_free + L_surf)
   ```

**For detailed derivations, see [THEORY.md](THEORY.md)**

---

## 🚀 **Getting Started**

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Linux/macOS/Windows with WSL2

### Installation Options

#### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t depth-nerf .

# Run container with GPU
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace depth-nerf
```

#### Option 2: Conda Environment

```bash
# Create environment
conda create -n depth-nerf python=3.9
conda activate depth-nerf

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

#### Option 3: Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

---

### Quick Start

#### Step 1: Clone Repository

```bash
git clone https://github.com/ayanali827/depth-supervised-nerf-notebooks.git
cd depth-supervised-nerf-notebooks
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

#### Step 4: Download Data

```bash
# Download NeRF Synthetic dataset
bash download_example_data.sh

# Data will be in: data/nerf_synthetic/lego/
```

#### Step 5: Launch Jupyter

```bash
jupyter notebook
# Open http://localhost:8888
```

---

## 🚀 **Running Experiments**

### Notebook Execution Pipeline

```
00_Setup_and_Dependencies.ipynb        (Setup environment)
  ↓
01_Data_Loading.ipynb                  (Load NeRF Synthetic dataset)
  ↓
02_Core_Components.ipynb               (Define NeRF architecture)
  ↓
[Run ALL four experiments in parallel or sequentially]
  ↓
03_Baseline_NeRF.ipynb                 (Experiment 1: No depth)
04_Soft_Depth_Supervision.ipynb        (Experiment 2: Soft constraint)
05_Hard_Depth_Sampling.ipynb           (Experiment 3: Hard constraint)
06_Hybrid_Strategy.ipynb               (Experiment 4: Combined)
  ↓
07_Comprehensive_Evaluation.ipynb      (Compare all strategies)
  ↓
08_Rendering_and_Visualization.ipynb   (Generate videos)
```

---

### Running Individual Experiments

#### Experiment 1: Baseline NeRF

```bash
jupyter notebook 03_Baseline_NeRF.ipynb
```

**Expected output:**
```
results/baseline/
├── model_baseline.pth      (trained weights)
├── psnr_history.npy       (metrics)
└── loss_history.npy
```

**Training time:** ~12-15 hours (50K iterations, V100)

---

#### Experiment 2: Soft Depth Supervision

```bash
jupyter notebook 04_Soft_Depth_Supervision.ipynb
```

**Configuration:**
- λ_soft = 0.01
- Same sampling as baseline
- Additional depth MSE loss

**Output:** `results/soft/model_soft.pth`

---

#### Experiment 3: Hard Depth Sampling

```bash
jupyter notebook 05_Hard_Depth_Sampling.ipynb
```

**Configuration:**
- 75% guided sampling
- Free-space loss
- Surface concentration loss
- λ_hard = 0.005

**Output:** `results/hard/model_hard.pth`

---

#### Experiment 4: Hybrid Strategy

```bash
jupyter notebook 06_Hybrid_Strategy.ipynb
```

**Configuration:**
- 50% uniform + 50% guided sampling
- Soft + hard losses combined
- λ_soft = 0.01, λ_hard = 0.005

**Output:** `results/hybrid/model_hybrid.pth`

---

### Comparative Evaluation

```bash
# Run after all 4 experiments complete
jupyter notebook 07_Comprehensive_Evaluation.ipynb
```

**Generates:**
- PSNR/SSIM comparison tables
- Depth error statistics
- Visual comparison plots
- Statistical significance tests

**Output:**
```
results/
├── evaluation_results.npy
├── evaluation_comparison.png
└── statistical_analysis.csv
```

---

### Novel View Rendering

```bash
jupyter notebook 08_Rendering_and_Visualization.ipynb
```

**Generates:**
- Individual videos: `results/{baseline,soft,hard,hybrid}/renders/*.mp4`
- Side-by-side comparison: `results/comparison.mp4`
- Depth map visualizations

---

## 📊 **Research Results**

### Quantitative Comparison (Lego Scene, 20K iterations)

| Strategy   | PSNR ↑         | SSIM ↑         | MSE ↓          | MAE ↓          | Robustness    |
|-----------|----------------|----------------|----------------|----------------|---------------|
| **Baseline** | 19.91 ± 1.60 dB | 0.658 ± 0.064 | 0.0101 ± 0.003 | 0.0477 ± 0.012 | High variance |
| **Soft**     | 20.72 ± 1.23 dB | 0.523 ± 0.068 | 0.0088 ± 0.002 | 0.0482 ± 0.008 | Poor structure |
| **Hard**     | 21.27 ± 1.06 dB | 0.686 ± 0.027 | 0.0079 ± 0.002 | 0.0393 ± 0.006 | Stable        |
| **Hybrid**   | **22.32 ± 1.04 dB** | **0.778 ± 0.029** | **0.0060 ± 0.001** | **0.0318 ± 0.005** | **Most robust** |

**Key Findings:**
- 🏆 Hybrid achieves **+2.15 dB PSNR** and **+0.12 SSIM** over baseline
- ⚠️ Soft paradox: Better PSNR than baseline but **worst SSIM (0.523)** — depth loss hurts structural quality
- 🎯 Hybrid reduces error by **41% (MSE)** and **33% (MAE)** compared to baseline
- 📊 Lowest variance: Hybrid (±1.04 dB) vs Baseline (±1.60 dB) — **35% more consistent**

---

---

## 🏆 **Best Result Achieved (Hybrid)**

![hybrid](https://github.com/user-attachments/assets/b67dbd45-ed02-4a8f-99b8-5ad6e18472eb)


<br>

![Hybrid](Analysis/validation_curves_combined.png)
<br>
<em><strong>Best result achieved:</strong> Hybrid depth supervision strategy combining soft and hard constraints for optimal rendering quality and geometric accuracy.</em>

---

### Research Conclusions

1. **Depth supervision improves geometric consistency** at the cost of slight RGB quality degradation

2. **Hard constraints (guided sampling) are most effective** for geometric accuracy but hurt photometric quality

3. **Soft constraints (loss-based) preserve RGB quality** better but provide weaker geometric improvements

4. **Hybrid approach is recommended** for applications requiring both realistic rendering and accurate geometry (robotics, AR/VR)

5. **Computational cost:** Depth supervision adds 10-35% training time overhead

---

## 📁 **Research Artifacts**

```
Depth-NeRF-Research/
├── trained_models/           ← All .pth weight files (200MB)
│   ├── baseline.pth
│   ├── soft.pth
│   ├── hard.pth
│   └── hybrid.pth
│
├── rendered_videos/          ← Novel view synthesis (1.2GB)
│   ├── baseline.mp4
│   ├── soft.mp4
│   ├── hard.mp4
│   ├── hybrid.mp4
│   └── comparison.mp4
│
├── evaluation_results/       ← Metrics and plots (50MB)
│   ├── psnr_comparison.png
│   ├── ssim_comparison.png
│   ├── depth_error_maps/
│   └── statistical_tests.csv
│
├── training_logs/            ← TensorBoard logs (50MB)
│   └── [.tfevents files]
│
└── raw_data/                 ← NeRF Synthetic dataset (2.5GB)
    └── nerf_synthetic/
```

---

## 📂 **Repository Structure**

```
depth-supervised-nerf-notebooks/
├── README.md                               ← This file
├── THEORY.md                               ← Detailed math/theory
├── LICENSE                                 ← MIT License
├── requirements.txt                        ← Python dependencies
├── .gitignore                              ← Git exclusions
├── Dockerfile                              ← Docker image
├── download_example_data.sh                ← Data script
├── hybrid.gif                              ← Existing best result animation
├── hybrid_best.gif                         ← New best result GIF
│
├── 00_Setup_and_Dependencies.ipynb         ← Environment setup
├── 01_Data_Loading.ipynb                   ← Dataset loading
├── 02_Core_Components.ipynb                ← NeRF architecture
├── 03_Baseline_NeRF.ipynb                  ← Experiment 1
├── 04_Soft_Depth_Supervision.ipynb         ← Experiment 2
├── 05_Hard_Depth_Sampling.ipynb            ← Experiment 3
├── 06_Hybrid_Strategy.ipynb                ← Experiment 4
├── 07_Comprehensive_Evaluation.ipynb       ← Results analysis
├── 08_Rendering_and_Visualization.ipynb    ← Video generation
│
├── Analysis/                               ← Extended analysis
│   ├── ablation_studies.ipynb
│   └── hyperparameter_sweep.ipynb
│
├── Complete_Pipeline/                      ← Complete workflows
│
├── data/                                   ← Datasets (git-ignored)
│   └── nerf_synthetic/lego/
│
└── results/                                ← Outputs (git-ignored)
    ├── baseline/
    ├── soft/
    ├── hard/
    └── hybrid/
```

---

## 🐛 **Troubleshooting**

**Issue: CUDA Out of Memory**

```python
# Solution: Reduce batch size
batch_rays = 512  # instead of 1024

# Use half-resolution
load_synthetic_split("train", half_res=True)

# Reduce samples per ray
N_samples = 32  # instead of 64
```

**Issue: Training Too Slow**

```bash
# Verify GPU usage
nvidia-smi  # Should show 90-100% utilization

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Issue: Poor Convergence**

- Increase iterations: `iters = 50000`
- Try hybrid strategy (most robust)
- Check depth ground truth quality

---

## 📚 **Citation**

```bibtex
@misc{ali2025depthnerf,
  title={Research on Improving NeRF Training Quality Using Depth Data},
  author={Ali, Ayan and Mughal, Umer Ahmed Baig and Urwa},
  year={2025},
  institution={ITMO University},
  note={Comparative study of depth supervision strategies for Neural Radiance Fields},
  howpublished={\url{https://github.com/ayanali827/depth-supervised-nerf-notebooks}}
}
```

**Original NeRF:**

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and others},
  booktitle={ECCV},
  year={2020}
}
```



## 🙏 **Acknowledgments**

- **ITMO University** - Robotics and AI Master's Program
- **Course:** Machine Learning in Robotics
- **Original NeRF** - Mildenhall et al., ECCV 2020
- **NeRF Synthetic Dataset** - Authors of NeRF paper

---

**Last Updated:** December 14, 2025  
**Authors:** Ayan Ali ([@ayanali827](https://github.com/ayanali827)), Umer Ahmed Baig Mughal ([@umerahmedbaig7](https://github.com/umerahmedbaig7)), Urwa ([@UrwaMughal7](https://github.com/UrwaMughal7))  
**Course:** Machine Learning in Robotics, ITMO University  
**Research Focus:** Depth-supervised Neural Radiance Fields
