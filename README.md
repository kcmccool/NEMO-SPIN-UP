
# Learning based emulation of ocean-model spin-up dynamics

## Grid-Invariant Ocean Forecasting with Tokenized Features (DINO_GridInvariant_N2)

This repository contains the code for  grid-invariant ocean forecasting pipeline built around tokenized ocean fields, featuring three physics-informed model variants for state reconstruction and stratification prediction.

---

## Overview

The project aims to learn ocean state dynamics by converting physical fields into a grid-invariant token representation. The pipeline combines:

* Physics-aware feature construction
* Continuous coordinate embeddings
* Variable-aware token modeling
* Multi-task prediction of ocean state and stratification
* Reconstruction back to 3D ocean fields for evaluation and visualization

---

## Project Structure

```text
.
├── base_config.py
├── config.yaml
├── ctx.py
├── dataset.py
├── features.py
├── augment_input.py
├── perceiver_mlp.py
├── perceiver_diffusion.py
├── ocean_nca_lppn.py
├── loss_mlp.py
├── loss.py
├── loss_nca.py
├── train_mlp.py
├── train.py
├── train_nca.py
├── evaluate.py
├── export_metrics_mlp.py
├── export_metrics_diffusion.py
├── diagnose_diffusion.py
├── plot_mlp.py
├── tests/
├── requirements.txt
├── diffusion-env.yml
└── README.md

```

### Key Components

* **`ctx.py`**: Loads grid metadata, depth levels, and mesh scale metrics.
* **`features.py`**: Builds the physics-informed 36-channel token feature tensor.
* **`dataset.py`**: Loads `.npy` snapshots, standardizes tokens, and reconstructs grids.
* **`perceiver_mlp.py`**: Supervised baseline architecture.
* **`perceiver_diffusion.py`**: Latent cross-attention diffusion backbone.
* **`ocean_nca_lppn.py`**: Grid-invariant NCA + LPPN pipeline.
* **`train_*.py`**: Training entrypoints for model variants.
* **`evaluate.py`**: Field reconstruction and metric computation.
* **`tests/`**: Integration and forward-pass sanity checks.

---

## Model Variants

| Model | Script | Core Architecture | Key Mechanism | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Diffusion** | `perceiver_diffusion.py` | Latent Cross-Attention Diffusion | Concatenates feature values, variable IDs, 4D Fourier coordinates, and time embeddings. Preserves spatial detail via local state query tokens; uses Xavier uniform initialization. | `pred_noise`<br>`pred_n2` |
| **MLP Baseline** | `perceiver_mlp.py` | Direct Supervised Deep MLP | Linear feature projections with harmonic 4D coordinate embeddings through dense bottleneck layers (`256` $\rightarrow$ `128` $\rightarrow$ `64`) with `LayerNorm` and `ReLU`. | `pred_state`<br>`pred_n2` |
| **NCA + LPPN** | `ocean_nca_lppn.py` | 3D NCA + Local Pattern Producing Network | Computes metric-scaled finite-difference gradients/Laplacians ($e_{1t}, e_{2t}, e_{3t}$). Evolves coarse latents horizontally and decodes fields continuously via 3D trilinear sampling (`F.grid_sample`). | `pred_fields`<br>`pred_n2`<br>`overflow_loss` |

## Loss Functions
| Script | Loss Class | Formulation / Key Strategy | Description |
| :--- | :--- | :--- | :--- |
| `loss.py` | `ScaledMaskedLoss` | $L_{\text{total}} = w_1 \frac{L_{\text{diff}}}{L_{0,\text{diff}}} + w_2 \frac{L_{\text{N2}}}{L_{0,\text{N2}}}$ | **Scale Normalization**: Normalizes task losses against step-0 initial baseline values ($L_0$) to balance gradient updates across multi-task heads. |
| `loss_mlp.py` | `DynamicMLPLoss` | $\sigma^2 = 1 + \text{softplus}(\theta)$<br>$L = \frac{1}{2} \left( \frac{L_{\text{task}}}{\sigma^2} + \log(\sigma^2) \right)$ | **Uncertainty Weighting**: Bounds homoscedastic variance $\ge 1.0$ via shifted Softplus to prevent negative losses or gradient stalling without parameter clipping. |
| `loss_nca.py` | `DynamicMLPLoss` | $L_{\text{NCA}} = L_{\text{base}} + \lambda_{\text{AC}} L_{\text{AC}} + \lambda_{\text{overflow}} L_{\text{overflow}}$ | **Physics Multi-Term Loss**: Combines uncertainty-weighted MSE, 3D FFT spatial autocorrelation matching (`rfftn`), and state boundary overflow penalties.(https://arxiv.org/pdf/2506.22899) |

## Data Flow

1. **Grid Context**: Load mesh mask and grid scale factors via `ctx.py`.
2. **Feature Extraction**: Construct 36-channel physics-informed features from NEMO arrays (temperature, salinity, density, spatial stencils, geometry).
3. **Tokenization**: Map ocean grids to invariant continuous token representations.
4. **Model Forward Pass**: Process tokens through chosen model variant (MLP, Diffusion, or NCA).
5. **Multi-Task Prediction**: Predict target ocean state variables (`soce`, `toce`, `ssh`) and auxiliary stratification ($N^2$).
6. **Reconstruction**: Reconstruct continuous 3D physical fields for field evaluation and visualization.

---

## Configuration & Hyperparameters

Default settings are defined via dataclasses and overridden at runtime by `config.yaml`:

| Category | Key Parameters | Default Setting |
| --- | --- | --- |
| **Model Dimensions** | `embedding_dim` / `latent_dim` / `num_latents` | `128` / `256` / `64` |
| **Attention** | `num_heads` / `depth` | `4` / `4` |
| **Variables** | `num_variable_types` | `3` (`soce: 0`, `toce: 1`, `ssh: 2`) |
| **Training** | `mixed_precision` / Batch Size / Acc. Steps | `fp16` / `6` per GPU / `4` accum (`24` effective) |
| **Optimizer** | `learning_rate` / Schedule / Seed | `0.0002` / Cosine decay (`100` warmup) / `42` |
| **Diffusion** | `num_train_timesteps` / `num_inference_steps` | `1000` / `1000` |

---

## Quickstart & Execution

```bash
# Train Model Variants
python train.py      # Diffusion pipeline
python train_mlp.py  # Supervised MLP baseline
python train_nca.py  # NCA + LPPN pipeline

```

---

## Data and Results

Large datasets (NEMO ocean snapshots), pre-computed feature arrays, training data, generated npy files and model checkpoints are excluded from this repository and are hosted separately.

---

## Requirements & Environment

The stack relies on standard scientific Python libraries:

`pytorch`, `numpy`, `xarray`, `matplotlib`, `scipy`, `pyyaml`, `diffusers`

Setup environments using either `requirements.txt` or `diffusion-env.yml`.

---

## Validation

Run automated sanity checks and test coverage:

```bash
pytest tests/

```

Includes unit tests for grid metadata loading, token dataset generation, model forward passes, and loss stabilization checks.

---

### NEMO Spin-UpBenchmark Experimentation & Comparative Baselines

In addition to the primary tokenized grid-invariant architectures, this project incorporates comparative experiments with non-linear spatial dimensionality reduction and sequence forecasting methods adapted from the [nemo_spinup_benchmark](https://github.com/m2lines/nemo-spinup-forecast) repository. These baselines allow for benchmarking field-reconstruction and temporal rollout capabilities.

**Dimensionality Reduction**

| Technique | Class Name | Description & Mechanism | Integration / Handling |
| --- | --- | --- | --- |
| **Convolutional VAE** | `DimensionalityReductionCVAE` | Projects 2D/3D ocean fields into low-dimensional latent space using convolutional encoder-decoder networks optimized via ELBO loss. | Implements dynamic spatial interpolation for NEMO grids|

**Sequential Neural Forecasters**

| Forecaster Module | Class Name | Underlying Mechanics | Application to Latent Dynamics |
| --- | --- | --- | --- |
| **Deep Koopman Operator** | `DeepKoopmanForecaster` | Non-linear encoder/decoder linked by a constrained linear transition operator $K$. | Enforces a spectral radius penalty ($\rho(K) \le 1.0$) to constrain eigenvalues and prevent divergence during long rollouts. |
| **Neural ODE** | `NeuralODEForecaster` | Parameterizes continuous-time hidden state vector fields integrated via `torchdiffeq` solvers (e.g., `dopri5`). | Models non-uniformly sampled or continuous physical time-series trajectories past $t = 1.0$. |
| **Recurrent Neural Network** | `RNNForecaster` | Standard autoregressive sequence modeling utilizing LSTM. | Serves as a classical deep learning benchmark for non-linear multi-step time series forecasting. |
