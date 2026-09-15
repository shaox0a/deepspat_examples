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
- `run_all.R` (repository root): optional script for running all examples directly

The output folders are:

- `Examples/Pic_nepal_GP_ST/`: figures and results for the spatio-temporal Gaussian demo
- `Examples/Pic_nepal_MSP/`: figures and results for the Brown–Resnick max-stable demo

## 2. Installation

To help users prepare a reproducible runtime environment, we provide the script:

- `reproduce_prepare.R`

This script creates and configures the required R and Python environment in one run. Run it from the repository root:

```bash
Rscript reproduce_prepare.R
```

If prompted to install R packages from source, select `no` to use binary packages. The equivalent setup steps are shown below for reference.

Because users may have different local machine settings, such as different R versions or missing system tools, the commands below may need to be adapted to the local system configuration. In particular, some components may require manual setup, such as:

- C/C++ build tools, such as Rtools on Windows

The examples have been tested using:

- **Python 3.12**
- **TensorFlow 2.18.0**
- **tf-keras 2.18.0**
- **TensorFlow Probability 0.25.0**
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

py_version <- "3.12:latest"
envname <- file.path(getwd(), "deepspat_venv")
path_to_python <- reticulate::install_python(version = py_version)

reticulate::virtualenv_create(
  envname = envname,
  python = path_to_python,
  version = py_version
)
```

### Step 2. Install TensorFlow-related Python packages

After creating the virtual environment, run:

```r
library(reticulate)
reticulate::use_virtualenv(envname, required = TRUE)

reticulate::virtualenv_install(
  envname = envname,
  packages = c(
    "tensorflow==2.18.0",
    "tensorflow-probability==0.25.0",
    "tf-keras==2.18.0",
    "scipy"
  )
)
```

### Step 3. Install the required R packages

```r
install.packages(c(
  "deepspat",
  "reticulate",
  "tensorflow",
  "keras",
  "tfprobability",
  "dplyr",
  "ggplot2",
  "ggnewscale",
  "elevatr",
  "RColorBrewer",
  "patchwork",
  "fields",
  "gstat",
  "GpGp",
  "cocons",
  "ggmap",
  "verification",
  "FNN",
  "devtools",
  "scales",
  "sp",
  "viridis",
  "gridExtra",
  "ggpubr",
  "this.path"
))

install.packages(
  "contoureR",
  repos = c("https://cran.r-universe.dev", "https://cloud.r-project.org")
)
```

### Step 4. Check that the installation works

After the environment is set up, run:

```r
library(reticulate)
library(tensorflow)

reticulate::use_virtualenv(file.path(getwd(), "deepspat_venv"), required = TRUE)

py_config()
tf$constant("TensorFlow is available")
```

If these commands run without error, the environment is ready. The Python path shown by `py_config()` should point to `deepspat_venv`.

## 3. How to run

From the repository root:

### Spatio-temporal Gaussian demo

This demo writes results to `Examples/Pic_nepal_GP_ST/`.

```bash
Rscript Examples/app_model_GP_ST.R
Rscript Examples/app_results_GP_ST.R
```

### Max-stable Brown–Resnick demo

This demo writes results to `Examples/Pic_nepal_MSP/`.

```bash
Rscript Examples/app_model_MSP.R
Rscript Examples/app_results_MSP.R
```

### Optional simulations

```bash
Rscript Examples/sims_1.R
Rscript Examples/results_sims_1.R

Rscript Examples/sims_2.R
Rscript Examples/results_sims_2.R
```

Alternatively, run all examples directly with:

```bash
Rscript run_all.R
```

## 4. Background

Deep compositional spatial models couple standard spatial covariance and extreme-value constructions with an injective warping of the spatial, and when needed temporal, domain.

The warping is built as a composition of elemental injective mappings within a deep-learning framework. We consider deformations known up to weights to be estimated. Estimation and inference are performed in TensorFlow via automatic differentiation.

The examples in this repository illustrate this paradigm by fitting stationary models in the warped space, including Gaussian processes and Brown–Resnick max-stable processes, to reproduce the Nepal case study and simulation workflows.
