# deepspat_examples

Minimal, reproducible examples for **nonstationary spatial & spatio-temporal modeling** with **`deepspat`** (Gaussian + extremes).

## Tested environment
- **Python**: 3.11  
  **TensorFlow**: 2.19.0 • **Keras**: 2.15.0 • **TensorFlow Probability**: 0.15.1
- **R**: ≥ 4.2 with packages: `deepspat`, `reticulate`, `tensorflow`, `tfprobability`, `keras`, `dplyr`, `ggplot2`, `patchwork`, `fields`, `gstat`, `GpGp`, `sp`, `viridis`, `gridExtra`, `ggpubr`, `this.path`

## How to run (direct execution)
From the repository root:
```
cd Examples

# Spatio-temporal Gaussian demo → Pic_nepal_GP_ST/
Rscript app_model_GP_ST.R && Rscript app_results_GP_ST.R

# Max-stable (Brown–Resnick) demo → Pic_nepal_MSP/
Rscript app_model_MSP.R && Rscript app_results_MSP.R

# Optional simulations
Rscript sims_1.R && Rscript results_sims_1.R
Rscript sims_2.R && Rscript results_sims_2.R
```
Or instead, you can also implment run_all.R directly.

## Background
Deep compositional spatial models couple standard spatial covariance and extreme-value constructions with an injective warping of the spatial (and, when needed, temporal) domain. The warping is built as a composition of elemental injective mappings within a deep-learning framework. We consider deformations known up to weights to be estimated. Estimation and inference are performed in TensorFlow via automatic differentiation. The examples here illustrate this paradigm by fitting stationary models in the warped space--Gaussian processes and Brown–Resnick max-stable processes--to reproduce the Nepal case study and simulation workflows.
