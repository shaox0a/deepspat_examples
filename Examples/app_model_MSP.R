###############################################
# 01_nepal_MSP_model.R
# - Fit MSP-BR model for Nepal data
# - Precompute all quantities that depend on `d1`
# - Save them as a pure R list (fitresults) to be used by plotting script
###############################################

rm(list = ls())
examples_path <- NULL
# Specify the path to the deepspat examples directory.
deepspat_path <- NULL
# Set `deepspat_path` to a local package path, or NULL to use library(deepspat).
if (!is.null(examples_path)) {
  setwd(examples_path)
}

message("Currently running: Fitting the models for case study with Nepal maximum temperature")

###############################################
# Load core modelling libraries
###############################################
if (is.null(deepspat_path)) {
  library(deepspat)
} else {
  pkgload::load_all(
    deepspat_path,
    quiet = TRUE
  )
}
library(tensorflow)    # Backend for deepspat
library(keras)
library(tfprobability)
library(dplyr)
library(fields)        # rdist for pairwise distances

# For elevation contours that will later be warped
library(elevatr)       # Elevation data
library(contoureR)     # Contour extraction from scattered elevation data

# Custom helper functions: edm_est, extcoef, grad_extcoef, fmadogram, etc.
source("Examples/utils_ext.R")


###############################################
# Application & model labels
###############################################
model    <- "MSP-BR"
app_data <- "NepalExtended"


###############################################
# Load main data (S, Z.max, etc.)
###############################################
# NepalExtended.RData is assumed to contain at least:
#  - S: locations (lon, lat)
#  - Z.max: block maxima at each location (columns = replicates)
simnames <- load(file = "Examples/Data/NepalExtended.RData")
print(simnames)

# Block maxima data matrix
data <- Z.max

# Build combined data frame: locations + maxima
df <- cbind(S, data) %>% as.data.frame()
names(df) <- c("s1", "s2", paste0("z", 1:(ncol(df) - 2)))

# Split into locations and responses
df_loc  <- dplyr::select(df, s1, s2)
df_data <- df[, 3:ncol(df)]


###############################################
# Empirical extremal dependence measure (EDM)
# for all locations (full grid)
###############################################
edm_est_filename <- paste0("Examples/Data/", app_data, "_", model, "_empextdep.rds")

# Compute and cache EDM for all site pairs if not already done
if (!file.exists(edm_est_filename)) {
  # edm_est: custom function from utils_ext.R
  all_edm_est <- edm_est(df_data, as.matrix(df_loc), model)
  saveRDS(all_edm_est, edm_est_filename)
}


###############################################
# Subsample locations for model fitting
###############################################
seedn1 <- 1
set.seed(seedn1)     # For reproducible subsampling
D_obs <- 500         # Number of locations used in model fitting

sam1 <- sample(1:nrow(df), D_obs)

df.obs  <- df[sam1, ]                 # Subsampled data frame
obs_loc <- df.obs[, c("s1", "s2")]   # Subsampled locations
obs_data <- df.obs[, 3:ncol(df.obs)] # Subsampled maxima

# Quick visual sanity check of subsampled locations
plot(obs_loc)

# Combined locations + data for deepspat_MSP
obs_all <- cbind(obs_loc, obs_data) %>% as.data.frame()


###############################################
# Empirical EDM on subsample (used in MSP fitting)
###############################################
method <- "MRPL"           # Estimation method for deepspat_MSP
family <- "power_nonstat"  # Non-stationary power variogram family
dtype  <- "float64"        # TensorFlow dtype

# Empirical EDM at subsampled locations
obs_edm_est <- edm_est(obs_data, as.matrix(obs_loc), model)$edm

# First column often used as "empirical extremal coefficient" for all pairs
obs_edm_emp <- obs_edm_est[, 1]


###############################################
# Set up warping layers for deformation f(s)
###############################################
r1 <- 50L
layers <- c(
  # Axial warping unit in dimension 1 (longitude)
  AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
  # Axial warping unit in dimension 2 (latitude)
  AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
  # Radial basis function block (1D)
  RBF_block(1L, dtype = dtype),
  # Optional extra radial block for 2D (currently disabled)
  # RBF_block(2L, dtype = dtype),
  # Linear-fractional transformation layer
  LFT(dtype = dtype)
)


###############################################
# Fit MSP-BR model using deepspat_MSP
###############################################
d1 <- deepspat_MSP(
  f = as.formula(
    paste(
      paste(paste0("z", 1:(ncol(obs_all) - 2)), collapse = "+"),
      "~ s1 + s2 -1"
    )
  ),
  data       = obs_all,            # Subsampled data
  layers     = layers,             # Warping architecture
  method     = method,
  family     = family,
  dtype      = dtype,
  nsteps     = 50L,                # Main optimization iterations
  nsteps_pre = 50L,                # Pre-training iterations for stability
  par_init   = initvars(),         # Parameter initial values
  learn_rates = init_learn_rates(  # Learning rates
    eta_mean = 0.01,
    vario    = 0.01
  ),
  edm_emp = obs_edm_emp,           # Empirical extremal dependence input
  p       = 0.01                   # Tail probability level for extremes
)

print(d1)
d1_summary <- summary(d1)
print(d1_summary)

# NOTE:
# We do NOT attempt to save `d1` directly, because it contains TensorFlow
# sessions/graphs that are cumbersome to serialize robustly.
# Instead, we now extract and save *all numeric summaries* needed later.

###############################################
# Reference sites (used throughout the plots)
###############################################
ref_pts <- c(549L, 1317L)


###############################################
# Extract numeric summaries from fitted model
###############################################
# Prediction at all original locations (df_loc).
# This includes rescaled coordinates, warped coordinates, fitted dependence
# parameters, Sigma.psi, and one reference-site dependence map.
pred <- predict(d1, df_loc, type = "dependence",
                reference = ref_pts[1L], se = TRUE)

# Rescaled and warped coordinates
S_rescaled <- pred$srescaled    # Rescaled original coordinates
S_warped   <- pred$swarped      # Warped coordinates f(s)

# Fitted dependence parameters (e.g., Brown–Resnick range & shape)
range_fitted <- as.numeric(pred$fitted.phi)   # phi
dof_fitted   <- as.numeric(pred$fitted.kappa) # kappa

# Asymptotic covariance matrix for transformed parameters (for delta method)
Sigma_psi <- pred$Sigma.psi


###############################################
# Examples of the new S3 plot methods
# These are quick checks only and are not saved.
###############################################
plot(d1, type = "space", pred = pred)
plot(d1, type = "dependence", pred = pred)
plot(d1, type = "uncertainty", pred = pred)


###############################################
# Precompute elevation contours and their warping
###############################################
# This block reproduces the elevation-based contours that are later used
# in the plotting script, BUT also applies the warping f(s) to each
# contour vertex using `predict(d1, ..., type = "warp")`.

# Elevation extraction (EPSG:4326) at original locations S
elev_extract <- elevatr::get_elev_point(
  data.frame(x = S[, 1], y = S[, 2]),
  prj = 4326, src = "aws"
)
elev <- elev_extract$elevation
elev[is.na(elev)] <- 0  # Fallback for missing values

# Elevation field on the grid
df_elev <- data.frame(s1 = S[, 1], s2 = S[, 2], elev = elev)

# Extract contour lines (original space)
df_contour <- contoureR::getContourLines(df_elev, nlevels = 4)
# Columns are typically: x, y (coordinates), z (elevation), Group, level, etc.

# Warp contour vertices into warped space using f(s)
cont_warped <- predict(
  d1,
  data.frame(s1 = df_contour$x, s2 = df_contour$y),
  type = "warp"  # No uncertainty needed, only s_warped
)$swarped

# Attach warped coordinates (xw, yw) for later plotting
df_contour$xw <- cont_warped[, 1]
df_contour$yw <- cont_warped[, 2]


###############################################
# Precompute warped grid lines (for grid plots)
###############################################
# The plotting script later draws:
#  - grid lines in original space (constructed only from S)
#  - grid lines in warped space (constructed from warped S values)
# However, to obtain the warped grid lines, it originally used
# `predict(d1, verti[[i]], type = "warp")$swarped`. Since `d1` will not be saved,
# we precompute them here.

# Unique longitudes and latitudes
uni.lon <- unique(S[, 1])
uni.lat <- unique(S[, 2])

# Vertically oriented grid lines (original space)
verti <- lapply(seq_along(uni.lon), function(i) {
  data_tmp <- data.frame(S[which(S[, 1] == uni.lon[i]), ])
  data_tmp_ord <- data_tmp[order(data_tmp[, 2]), ]
  names(data_tmp_ord) <- c("s1", "s2")
  data_tmp_ord
})

# Horizontally oriented grid lines (original space)
horiz <- lapply(seq_along(uni.lat), function(i) {
  data_tmp <- data.frame(S[which(S[, 2] == uni.lat[i]), ])
  data_tmp_ord <- data_tmp[order(data_tmp[, 1]), ]
  names(data_tmp_ord) <- c("s1", "s2")
  data_tmp_ord
})

# Warp each grid line using the fitted deformation f(s)
df_verti_warped <- data.frame(
  do.call(
    "rbind",
    lapply(seq_along(verti), function(i) {
      warped_i <- predict(d1, verti[[i]], type = "warp")$swarped
      rbind(warped_i, c(NA, NA))  # NA separator between lines
    })
  )
)
df_horiz_warped <- data.frame(
  do.call(
    "rbind",
    lapply(seq_along(horiz), function(i) {
      warped_i <- predict(d1, horiz[[i]], type = "warp")$swarped
      rbind(warped_i, c(NA, NA))  # NA separator between lines
    })
  )
)

names(df_verti_warped) <- names(df_horiz_warped) <- c("s1", "s2")


###############################################
# Collect all results needed for plotting
# and save them to a single rds file
###############################################
fit_results <- list(
  app_data      = app_data,
  model         = model,
  S             = S,
  df_loc        = df_loc,
  S_rescaled    = S_rescaled,
  S_warped      = S_warped,
  range_fitted  = range_fitted,
  dof_fitted    = dof_fitted,
  Sigma_psi     = Sigma_psi,
  elev          = elev,
  df_elev       = df_elev,
  df_contour    = df_contour,
  df_verti_warped = df_verti_warped,
  df_horiz_warped = df_horiz_warped,
  ref_pts       = ref_pts,
  seed_subsample = seedn1,
  D_obs         = D_obs,
  subsample_idx = sam1
)

saveRDS(
  fit_results,
  file = paste0("Examples/Data/", app_data, "_", model, "_fitresults.rds")
)

cat("Model fitting finished. Numerical results saved to:\n",
    paste0("Examples/Data/", app_data, "_", model, "_fitresults.rds"), "\n")
