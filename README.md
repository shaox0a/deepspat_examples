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
- `check_packages.R`: checks that the R packages required by the examples are installed
- `contour_extract.R`: optional reference script used to generate the supplied
  elevation and contour data; it is not run by the normal reproduction workflow
- `run_all.R` (repository root): optional script for running all examples directly

The output folders are:

- `Examples/Pic_nepal_GP_ST/`: figures and results for the spatio-temporal Gaussian demo
- `Examples/Pic_nepal_MSP/`: figures and results for the Brown–Resnick max-stable demo

The folder `Examples/Data/` contains the data supplied with the repository:

- `NepalExtended.RData` contains `S`, a 1,419 × 2 matrix of spatial
  coordinates, and `Z.max`, a 1,419 × 192 matrix of block-maxima observations
  used by the Brown–Resnick max-stable application. The rows of `Z.max`
  correspond to the locations in `S`.
- `NepalExtended_mean.RData` contains `dataset`, with 22,704 observations at
  1,419 locations for 2004–2019. Its columns are `year`, spatial coordinates
  `s1` and `s2`, elevation `elev`, and the mean-temperature response `Y_mean`.
  It is used by the spatio-temporal Gaussian application.
- `NepalMap.RData` contains `nepal`, the terrain-map raster used as the
  background map in the Nepal application figures.
- `NepalContour.RData` contains the MSP elevation data `df_elev_MSP` and the
  precomputed contour vertices `df_contour_GP_ST` and `df_contour_MSP`. The
  elevation data contain coordinates `s1` and `s2` and elevation `elev`; the
  contour objects contain identifiers, coordinates `x` and `y`, and elevation
  level `z`. These data are used only for application figures and do not need
  to be regenerated for the normal reproduction workflow. See
  `Examples/contour_extract.R` for generation details.

## 2. Installation

To help users prepare a reproducible runtime environment, we provide two scripts:

- `reproduce_prepare_packages.R`: installs missing R packages and updates `deepspat` when needed
- `reproduce_prepare_pyenv.R`: creates and configures the project-local Python environment

`run_all.R` sources the R-package preparation script, which installs only missing packages, and sources the Python preparation script only when `deepspat_venv` does not exist. To prepare the environment and run all examples, run the following command from the repository root:

```bash
Rscript run_all.R
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

### Step 1. Install the required R packages

From the repository root, run:

```r
source("reproduce_prepare_packages.R")
```

This installs the latest CRAN source version of `deepspat` if it is missing or outdated and installs only the other required R packages that are missing.

### Step 2. Set up the Python environment

```r
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

py_version <- "3.12:latest"
envname <- file.path(getwd(), "deepspat_venv")
path_to_python <- reticulate::virtualenv_starter("3.12")
if (is.null(path_to_python)) {
  path_to_python <- reticulate::install_python(version = py_version)
}

reticulate::virtualenv_create(
  envname = envname,
  python = path_to_python,
  version = py_version
)
```

The setup reuses an existing Python 3.12 installation when available and only installs Python when none is found. Avoid forcing `install_python()`, as this may overwrite an existing reticulate-managed Python installation.

**Remark:** On macOS, building Python may fail because the required Homebrew OpenSSL formula is unavailable or outdated. If Homebrew is installed, update Homebrew and OpenSSL, then rerun the setup:

```bash
brew update
brew upgrade openssl@3
```

If `openssl@3` is not installed, run `brew install openssl@3` instead. This is a local system configuration issue, not part of the package implementation, so it is not handled by the reproduction scripts.

### Step 3. Install TensorFlow-related Python packages

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

```bash
Rscript run_all.R
```

Individual examples can also be run as follows.

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

## 4. Background

Deep compositional spatial models couple standard spatial covariance and extreme-value constructions with an injective warping of the spatial, and when needed temporal, domain.

The warping is built as a composition of elemental injective mappings within a deep-learning framework. We consider deformations known up to weights to be estimated. Estimation and inference are performed in TensorFlow via automatic differentiation.

The examples in this repository illustrate this paradigm by fitting stationary models in the warped space, including Gaussian processes and Brown–Resnick max-stable processes, to reproduce the Nepal case study and simulation workflows.
