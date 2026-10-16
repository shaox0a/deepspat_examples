required_packages <- c(
  "FNN",
  "RColorBrewer",
  "cocons",
  "deepspat",
  "devtools",
  "dplyr",
  "fields",
  "ggmap",
  "ggnewscale",
  "ggplot2",
  "ggpubr",
  "GpGp",
  "gridExtra",
  "gstat",
  "keras",
  "patchwork",
  "reticulate",
  "scales",
  "sp",
  "tensorflow",
  "tfprobability",
  "this.path",
  "verification",
  "viridis"
)

# Check which packages are installed
missing_packages <- character(0)
installed_packages <- character(0)

cat("Checking required packages...\n\n")

for (pkg in required_packages) {
  if (pkg %in% rownames(installed.packages())) {
    installed_packages <- c(installed_packages, pkg)
    cat(sprintf("  ✓ %s\n", pkg))
  } else {
    missing_packages <- c(missing_packages, pkg)
    cat(sprintf("  ✗ %s (MISSING)\n", pkg))
  }
}

cat("\n")
cat(sprintf("Summary: %d/%d packages installed\n", 
            length(installed_packages), 
            length(required_packages)))

if (length(missing_packages) > 0) {
  cat("\nMissing packages:\n")
  for (pkg in missing_packages) {
    cat(sprintf("  - %s\n", pkg))
  }
  stop(
    sprintf(
      "\nError: %d package(s) are missing. Please install the missing package(s) listed above.",
      length(missing_packages)
    )
  )
} else {
  cat("\nAll required packages are installed! ✓\n")
}

# Reject outdated CRAN binaries that are incompatible with these examples.
if (packageVersion("deepspat") < package_version("0.3.5")) {
  stop("deepspat version 0.3.5 or later is required.")
}
