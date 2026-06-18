# deepspat_examples

Minimal, reproducible examples for **nonstationary spatial and spatio-temporal modeling** with **`deepspat`** (Gaussian + extremes).

## 1. Overview

This repository provides minimal and reproducible examples for fitting deep compositional spatial models using `deepspat`.

The examples illustrate nonstationary spatial and spatio-temporal modeling by fitting stationary models in a warped domain. The included workflows cover both Gaussian processes and extreme-value models.

The folder `Examples/` contains:

- `app_model_GP_ST.R`: spatio-temporal Gaussian process model fitting
- `app_results_GP_ST.R`: result processing and plotting for the Gaussian example
- `app_model_MSP.R`: Brown–Resnick max-stable process model fitting
- `app_results_MSP.R`: result processing and plotting for the max-stable example
- `sims_1.R`, `results_sims_1.R`: optional simulation workflow 1
- `sims_2.R`, `results_sims_2.R`: optional simulation workflow 2
- `run_all.R`: optional script for running all examples directly

The output folders are:

- `Pic_nepal_GP_ST/`: figures and results for the spatio-temporal Gaussian demo
- `Pic_nepal_MSP/`: figures and results for the Brown–Resnick max-stable demo

## 2. Installation

To help users prepare a reproducible runtime environment, we provide the setup workflow below.

Because users may have different local machine settings, such as different R versions or missing system tools, the commands below may need to be adapted to the local system configuration. In particular, some components may require manual setup, such as:

- Git
- Rtools

The examples have been tested using:

- **Python 3.11**
- **TensorFlow 2.19.0**
- **Keras 2.15.0**
- **TensorFlow Probability 0.15.1**
- **R ≥ 4.2**

Required R packages include:

- `deepspat`
- `reticulate`
- `tensorflow`
- `tfprobability`
- `keras`
- `dplyr`
- `ggplot2`
- `patchwork`
- `fields`
- `gstat`
- `GpGp`
- `sp`
- `viridis`
- `gridExtra`
- `ggpubr`
- `this.path`

### Step 1. Set up the Python environment

```r
install.packages("reticulate")
library(reticulate)

py_version <- "3.11:latest"
path_to_python <- reticulate::install_python(version = py_version)

reticulate::virtualenv_create(
  envname = "deepspat_examples",
  python = path_to_python,
  version = py_version
)
```

### Step 2. Restart the R session and install TensorFlow-related Python packages

After creating the virtual environment, **restart the R session**. Then run:

```r
library(reticulate)
reticulate::use_virtualenv("deepspat_examples", required = TRUE)

tensorflow::install_tensorflow(
  method = "virtualenv",
  envname = "deepspat_examples",
  version = "2.19.0"
)

keras::install_keras(
  method = "virtualenv",
  envname = "deepspat_examples",
  version = "2.15.0"
)

reticulate::virtualenv_install(
  envname = "deepspat_examples",
  packages = "tensorflow-probability",
  version = "0.15.1"
)
```

### Step 3. Install the required R packages

```r
install.packages(c(
  "reticulate",
  "tensorflow",
  "keras",
  "tfprobability",
  "dplyr",
  "ggplot2",
  "patchwork",
  "fields",
  "gstat",
  "GpGp",
  "sp",
  "viridis",
  "gridExtra",
  "ggpubr",
  "this.path"
))
```

If `deepspat` is not already installed, install it from its source repository or from the location specified by the project maintainers.

### Step 4. Check that the installation works

After the environment is set up, run:

```r
library(reticulate)
library(tensorflow)

reticulate::use_virtualenv("deepspat_examples", required = TRUE)

py_config()
tf$constant("TensorFlow is available")
```

If these commands run without error, the environment is ready.

## 3. How to run

From the repository root:

```bash
cd Examples
```

### Spatio-temporal Gaussian demo

This demo writes results to `Pic_nepal_GP_ST/`.

```bash
Rscript app_model_GP_ST.R
Rscript app_results_GP_ST.R
```

### Max-stable Brown–Resnick demo

This demo writes results to `Pic_nepal_MSP/`.

```bash
Rscript app_model_MSP.R
Rscript app_results_MSP.R
```

### Optional simulations

```bash
Rscript sims_1.R
Rscript results_sims_1.R

Rscript sims_2.R
Rscript results_sims_2.R
```

Alternatively, run all examples directly with:

```bash
Rscript run_all.R
```

## 4. Background

Deep compositional spatial models couple standard spatial covariance and extreme-value constructions with an injective warping of the spatial, and when needed temporal, domain.

The warping is built as a composition of elemental injective mappings within a deep-learning framework. We consider deformations known up to weights to be estimated. Estimation and inference are performed in TensorFlow via automatic differentiation.

The examples in this repository illustrate this paradigm by fitting stationary models in the warped space, including Gaussian processes and Brown–Resnick max-stable processes, to reproduce the Nepal case study and simulation workflows.
