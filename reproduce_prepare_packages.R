################################################################################
# Install `deepspat` and the R packages required by it.
# Install from source because CRAN binaries for older R versions may be outdated.
options(repos = c(CRAN = "https://cloud.r-project.org"))

deepspat_packages <- c(
  "reticulate",
  "tensorflow",
  "keras",
  "tfprobability",
  "dplyr",
  "fields"
)

# Additional R packages required by the example scripts.
example_packages <- c(
  "ggplot2",
  "ggpubr",
  "ggnewscale",
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
  "sp",
  "this.path"
)

# Install the latest CRAN source version of `deepspat` if it is missing or outdated.
if (!requireNamespace("deepspat", quietly = TRUE) ||
    "deepspat" %in% rownames(old.packages(type = "source"))) {
  install.packages("deepspat", type = "source")
}

# Update `ggplot2` if needed because `patchwork` requires `is_ggplot()`.
if (!requireNamespace("ggplot2", quietly = TRUE) ||
    packageVersion("ggplot2") < package_version("3.5.2")) {
  install.packages("ggplot2")
}

# Install only the required R packages that are currently missing.
required_packages <- c(deepspat_packages, example_packages)
missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

