################################################################################
# Extract and save the Nepal elevation data and contours used by the applications.
#
# This optional script generated the elevation and contour data supplied in
# `Examples/Data/Nepal_contour.RData`. Run it from the repository root with:
#
# Rscript Examples/contour_extract.R
#
# Running this script is not recommended for the normal reproduction workflow.
# For the MSP application, elevation is obtained from AWS using `elevatr`; for
# the GP-ST application, elevation is read from `NepalExtended_mean.RData`. The
# contours are then extracted from the irregular spatial locations using
# `contoureR::getContourLines()`.
#
# `contoureR` is archived on CRAN, requires compilation from source, and depends
# on platform-specific C++ build tools. Its installation may therefore fail in
# confined environments or because of compiler/SDK configuration problems.
# The contours have therefore been generated in advance and saved in
# `Examples/Data/Nepal_contour.RData`, so `run_all.R` does not need to install
# or load `contoureR`. These data are used only for presenting the application
# results and figures; they are not part of the core method reproduction or
# model fitting. Users requiring contours for new data can follow this script
# or use another suitable method for irregular spatial data.

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("this.path", quietly = TRUE)) install.packages("this.path")
if (!requireNamespace("elevatr", quietly = TRUE)) install.packages("elevatr")

project_path <- file.path(this.path::this.dir(), "..")
setwd(project_path)

if (!requireNamespace("contoureR", quietly = TRUE)) {
  devtools::install_version(
    "contoureR",
    version = "1.0.5",
    upgrade = "never"
  )
}

# Extract contours for the GP-ST application from the supplied elevation data.
load("Examples/Data/NepalExtended_mean.RData")
df_elev_GP_ST <- unique(data.frame(dataset[, c("s1", "s2", "elev")]))
df_contour_GP_ST <- contoureR::getContourLines(df_elev_GP_ST, nlevels = 4)

# Obtain elevation and extract contours for the MSP application locations.
load("Examples/Data/NepalExtended.RData")
elev_extract <- elevatr::get_elev_point(
  data.frame(x = S[, 1], y = S[, 2]),
  prj = 4326,
  src = "aws"
)
elev <- elev_extract$elevation
elev[is.na(elev)] <- 0
df_elev_MSP <- data.frame(s1 = S[, 1], s2 = S[, 2], elev = elev)
df_contour_MSP <- contoureR::getContourLines(df_elev_MSP, nlevels = 4)

save(
  df_elev_MSP,
  df_contour_GP_ST,
  df_contour_MSP,
  file = "Examples/Data/Nepal_contour.RData"
)
