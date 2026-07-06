################################################################################
# Python environment setup
install.packages("reticulate")
library(reticulate)

py_version <- "3.11:latest"
path_to_python <- reticulate::install_python(version = py_version)

# Git installation:
# https://git-scm.com/install/windows
# During installation, choose:
# "Git from the command line and also from 3rd-party software"
# so that Git is automatically added to your system PATH.

reticulate::virtualenv_create(
  envname = "dcsmext",
  python = path_to_python,
  version = py_version
)

# Restart the R session before continuing.
reticulate::use_virtualenv("dcsmext", required = TRUE)

tensorflow::install_tensorflow(
  method = "virtualenv",
  envname = "dcsmext",
  version = "2.19.0"
)

keras::install_keras(
  method = "virtualenv",
  envname = "dcsmext",
  version = "2.15.0"
)

reticulate::virtualenv_install(
  envname = "dcsmext",
  packages = "tensorflow-probability",
  version = "0.15.1"
)

################################################################################
# Install required R packages
install.packages(c(
  "reticulate",
  "tensorflow",
  "keras",
  "tfprobability",
  "dplyr",
  "fields",
  "ggplot2",
  "ggpubr",
  "ggnewscale",
  "elevatr",
  "RColorBrewer",
  "this.path",
  "gridExtra",
  "viridis"
))

library(tensorflow)
library(keras)
library(tfprobability)
library(dplyr)
library(fields)
library(ggplot2)
library(ggpubr)
library(ggnewscale)
library(elevatr)
library(RColorBrewer)
library(this.path)
library(gridExtra)
library(viridis)

################################################################################
# The packages `maps` and `contoureR` are only needed for reproducing
# the plots in the UK precipitation data application.
# If you do not need to reproduce those figures, you can skip this section.

install.packages("maps")
install.packages("contoureR")

# Note:
# `contoureR` is mainly available for older R versions (for example, R 4.3.2).
# If you are using a newer version of R (for example, R 4.5.x),
# the package may need to be installed from source via CRAN.
# In that case, Rtools is usually required on Windows to compile the package.

# For newer R versions (here, the results were reproduced using R 4.5.3),
# you can install Rtools from:
# https://cran.r-project.org/bin/windows/Rtools/rtools45/rtools.html

# To verify that Rtools has been installed correctly, run:
Sys.which("make")
Sys.which("g++")

install.packages(
  "contoureR",
  repos = c("https://cran.r-universe.dev", "https://cloud.r-project.org")
)
library(contoureR)