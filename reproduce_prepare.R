################################################################################
# Create and configure the project-local Python environment.
# `deepspat_venv` is created alongside `reproduce_prepare.R`.
# If prompted to install packages from source, select "no" to use binary packages.
install.packages(c("reticulate", "this.path"))
library(reticulate)

project_path <- this.path::this.dir()
py_version <- "3.12:latest"
envname <- file.path(project_path, "deepspat_venv")
path_to_python <- reticulate::install_python(version = py_version)

reticulate::virtualenv_create(
  envname = envname,
  python = path_to_python,
  version = py_version
)

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

# The Python path shown below should point to `deepspat_venv`.
reticulate::py_config()

################################################################################
# Install `deepspat` and the R packages required by it.
install.packages(c(
  "deepspat",
  "reticulate",
  "tensorflow",
  "keras",
  "tfprobability",
  "dplyr",
  "fields"
))

# Install additional R packages required by the example scripts.
install.packages(c(
  "ggplot2",
  "ggpubr",
  "ggnewscale",
  "elevatr",
  "RColorBrewer",
  "gridExtra",
  "viridis",
  "cocons",
  "ggmap",
  "GpGp",
  "gstat",
  "verification",
  "FNN",
  "devtools",
  "patchwork",
  "scales",
  "sp"
))

################################################################################
# Install additional packages required by the application examples.

# `contoureR` is archived on CRAN and may need to be built from source.
# Check that the required build tools are available.
Sys.which("make")
Sys.which("g++")

# Install `contoureR` from R-universe.
install.packages(
  "contoureR",
  repos = c("https://cran.r-universe.dev", "https://cloud.r-project.org")
)
