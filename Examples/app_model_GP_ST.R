############################################################
## File: nepal_model_ST_GP.R
## Role: Fit chosen deepspat ST-GP model(s) and save:
##       (i) prediction objects / scores
##       (ii) all numeric quantities needed for plotting
##            (warped grids, warped contours, corr vectors…)
############################################################

rm(list = ls())
# Set working directory to the repo root
# setwd(...)

# -------------------------------------------------------------------
# Packages
# -------------------------------------------------------------------
library(reticulate)
library(tensorflow)
library(tfprobability)
library(keras)

library(dplyr)
library(fields)
library(GpGp)
library(devtools)
library(deepspat)

library(verification)   # CRPS

# -------------------------------------------------------------------
# Utility functions: RMSPE, CRPS, covariance helper
# -------------------------------------------------------------------

RMSPE <- function(true, pred){
  # Root Mean Squared Prediction Error
  sqrt(mean((true - pred)^2))
}

CRPS <- function(true, pred, pred_var){
  # Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions
  # 'pred' is the predictive mean, 'pred_var' is the predictive variance
  crps(true, cbind(pred, sqrt(pred_var)))$CRPS
}

# Covariance helper using fitted deepspat_nn_ST_GP object
cov_fn_compute <- function(object, newdata1, newdata2, ...) {
  
  d <- object   # shorthand
  
  # Design matrices for fixed effects (f) and covariates (g)
  mmat1 <- model.matrix(update(d$f, NULL ~ .), newdata1)
  X1_new1 <- model.matrix(update(d$g, NULL ~ .), newdata1)
  X_new1 <- tf$constant(X1_new1, dtype="float32")
  
  mmat2 <- model.matrix(update(d$f, NULL ~ .), newdata2)
  X1_new2 <- model.matrix(update(d$g, NULL ~ .), newdata2)
  X_new2 <- tf$constant(X1_new2, dtype="float32")
  
  # Split into spatial (s) and temporal (t)
  t_tf1 <- tf$constant(as.matrix(mmat1[, ncol(mmat1)]), name = "t1", dtype = "float32")
  s_tf1 <- tf$constant(as.matrix(mmat1[, 1:(ncol(mmat1) - 1)]), name = "s1", dtype = "float32")
  t_tf2 <- tf$constant(as.matrix(mmat2[, ncol(mmat2)]), name = "t2", dtype = "float32")
  s_tf2 <- tf$constant(as.matrix(mmat2[, 1:(ncol(mmat2) - 1)]), name = "s2", dtype = "float32")
  
  ndata <- nrow(d$data)
  m <- d$m
  p <- ncol(d$X)
  npred <- nrow(newdata2)
  
  beta <- tf$constant(d$beta, dtype = "float32", shape = c(p, 1L))
  
  z_tf <- d$z_tf
  z_tf_0 <- z_tf - tf$matmul(d$X, beta) 
  
  # ------------------------------------------------------------------
  # Stationary separable exponential covariance
  # ------------------------------------------------------------------
  if (d$family %in% c("exp_stat_sep")){
    obs_swarped <- d$swarped_tf
    newdata_swarped1 <- s_tf1
    newdata_swarped2 <- s_tf2
    
    obs_twarped <- d$twarped_tf
    newdata_twarped1 <- t_tf1
    newdata_twarped2 <- t_tf2
  }
  
  # ------------------------------------------------------------------
  # Nonstationary separable exponential covariance
  # ------------------------------------------------------------------
  if (d$family %in% c("exp_nonstat_sep")){
    
    # Rescale spatial and temporal inputs
    s_in1 <- scale_0_5_tf(s_tf1, d$scalings[[1]]$min, d$scalings[[1]]$max)
    t_in1 <- scale_0_5_tf(t_tf1, d$scalings_t[[1]]$min, d$scalings_t[[1]]$max)
    
    s_in2 <- scale_0_5_tf(s_tf2, d$scalings[[1]]$min, d$scalings[[1]]$max)
    t_in2 <- scale_0_5_tf(t_tf2, d$scalings_t[[1]]$min, d$scalings_t[[1]]$max)
    
    # Spatial warping layers
    h_tf1 <- list(s_in1)
    h_tf2 <- list(s_in2)
    for(i in 1:d$nlayers_spat) {
      if (d$layers_spat[[i]]$name == "LFT") {
        a_inum_tf <- d$layers_spat[[i]]$trans(d$layers_spat[[i]]$pars)
        h_tf1[[i + 1]] <- d$layers_spat[[i]]$f(h_tf1[[i]], a_inum_tf)
        h_tf2[[i + 1]] <- d$layers_spat[[i]]$f(h_tf2[[i]], a_inum_tf)
      } else {
        h_tf1[[i + 1]] <- d$layers_spat[[i]]$f(h_tf1[[i]], d$eta_tf[[i]]) 
        h_tf2[[i + 1]] <- d$layers_spat[[i]]$f(h_tf2[[i]], d$eta_tf[[i]]) 
      }
      h_tf1[[i + 1]] <- h_tf1[[i + 1]] %>%
        scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                     smax_tf = d$scalings[[i + 1]]$max)
      h_tf2[[i + 1]] <- h_tf2[[i + 1]] %>%
        scale_0_5_tf(smin_tf = d$scalings[[i + 1]]$min,
                     smax_tf = d$scalings[[i + 1]]$max)
    }
    
    # Temporal warping layers
    h_t_tf1 <- list(t_in1)
    h_t_tf2 <- list(t_in2)
    for(i in 1:d$nlayers_temp) {
      h_t_tf1[[i + 1]] <- d$layers_temp[[i]]$f(h_t_tf1[[i]], d$eta_t_tf[[i]]) %>%
        scale_0_5_tf(smin_tf = d$scalings_t[[i + 1]]$min,
                     smax_tf = d$scalings_t[[i + 1]]$max)
      
      h_t_tf2[[i + 1]] <- d$layers_temp[[i]]$f(h_t_tf2[[i]], d$eta_t_tf[[i]]) %>%
        scale_0_5_tf(smin_tf = d$scalings_t[[i + 1]]$min,
                     smax_tf = d$scalings_t[[i + 1]]$max)
    }
    
    obs_swarped <- d$swarped_tf
    newdata_swarped1 <- h_tf1[[d$nlayers_spat + 1]]
    newdata_swarped2 <- h_tf2[[d$nlayers_spat + 1]]
    
    obs_twarped <- d$twarped_tf
    newdata_twarped1 <- h_t_tf1[[d$nlayers_temp + 1]]
    newdata_twarped2 <- h_t_tf2[[d$nlayers_temp + 1]]
  }
  
  # Exponential covariance in warped space
  K <- cov_exp_tf(x1 = newdata_swarped1, x2 = newdata_swarped2,
                  sigma2f = 1L, alpha = 1/d$l_tf)
  
  return(K)
}

# -------------------------------------------------------------------
# Load dataset & basic preprocessing
# -------------------------------------------------------------------

names <- load("Examples/NepalExtended_mean.rda")   # Loads 'dataset'

# (Optional quick check plot; can be commented out if you don't want a window)
# plot(dataset[1:1419, c("s1","s2")]); points(dataset[c(348,363), c("s1","s2")], col="red")

# Standardise response
meanY <- mean(dataset$Y_mean)
sdY   <- sd(dataset$Y_mean)
dataset$Y_mean <- (dataset$Y_mean - meanY) / sdY

# Train / test split (50% training, 50% testing)
set.seed(1)
sam2 <- sample(1:nrow(dataset), 0.5 * nrow(dataset))
train_data <- dataset[sam2,]
test_data  <- dplyr::setdiff(dataset, train_data)

obsdata <- train_data
newdata <- test_data
alldata <- rbind(test_data, train_data)

# -------------------------------------------------------------------
# Warping layers & nearest-neighbour structure
# -------------------------------------------------------------------

layers_spat <- c(
  AWU(r = 100L, dim = 1L, grad = 20),
  AWU(r = 100L, dim = 2L, grad = 20),
  RBF_block(res = 1L),
  RBF_block(res = 2L),
  LFT()
)

layers_temp <- c(
  AWU(r = 20L, dim = 1L, grad = 20)
)

# Spatial and spatio-temporal locations
locs   <- t(rbind(obsdata$s1, obsdata$s2))
locs_t <- t(rbind(obsdata$s1, obsdata$s2, obsdata$year))

# Vecchia ordering and NN indices
set.seed(10)
order_id <- sample(1:nrow(locs))
nn_id    <- find_ordered_nn(locs_t[order_id,], m = 50)
m <- ncol(nn_id) - 1
n <- nrow(nn_id)

# Padding for first m rows
for (i in 1:m){
  nn_id[i, (i+1):(m+1)] <- (n+1):(n+1+m-i)
}

# -------------------------------------------------------------------
# Choose which model(s) to fit
# -------------------------------------------------------------------
# options: "d1", "d2", "d3", "d4", "all"
# NOTE: The plotting script assumes that d3 has been fitted,
#       so keep model_choice = "d3" or "all" for now.

model_choice <- "d3"

fit_d1 <- fit_d2 <- fit_d3 <- fit_d4 <- FALSE

if (model_choice == "d1") {
  fit_d1 <- TRUE
} else if (model_choice == "d2") {
  fit_d2 <- TRUE
} else if (model_choice == "d3") {
  fit_d3 <- TRUE
} else if (model_choice == "d4") {
  fit_d4 <- TRUE
} else if (model_choice == "all") {
  fit_d1 <- fit_d2 <- fit_d3 <- fit_d4 <- TRUE
} else {
  stop("Unknown model_choice; use 'd1', 'd2', 'd3', 'd4', or 'all'.")
}

# Will collect prediction objects and scores here
pred_objects <- character(0)

# -------------------------------------------------------------------
# Fit Models: Stationary (d1, d2)
# -------------------------------------------------------------------

if (fit_d1) {
  d1 <- deepspat_nn_ST_GP(
    f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ elev,
    family = "exp_stat_sep",
    layers_spat = layers_spat, layers_temp = layers_temp,
    m = 50L,
    order_id = order_id, nn_id = nn_id,
    method = "REML", nsteps = 50L,
    par_init = initvars(l_top_layer = 0.5),
    learn_rates = init_learn_rates(eta_mean = 0.01)
  )
  
  locs_new   <- t(rbind(alldata$s1, alldata$s2, alldata$year))
  nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
  pred_d1    <- predict(d1, alldata, nn_id_pred)
  
  RMSPE_d1 <- RMSPE(
    test_data$Y_mean,
    pred_d1$df_pred$pred_mean[1:nrow(test_data)]
  )
  CRPS_d1  <- CRPS(
    test_data$Y_mean,
    pred_d1$df_pred$pred_mean[1:nrow(test_data)],
    pred_d1$df_pred$pred_var[1:nrow(test_data)] + 1/d1$precy_tf
  )
  
  pred_objects <- c(pred_objects, "pred_d1", "RMSPE_d1", "CRPS_d1")
}

if (fit_d2) {
  d2 <- deepspat_nn_ST_GP(
    f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ 1,
    family = "exp_stat_sep",
    layers_spat = layers_spat, layers_temp = layers_temp,
    m = 50L,
    order_id = order_id, nn_id = nn_id,
    method = "REML", nsteps = 50L,
    par_init = initvars(l_top_layer = 0.5),
    learn_rates = init_learn_rates(eta_mean = 0.01)
  )
  
  locs_new   <- t(rbind(alldata$s1, alldata$s2, alldata$year))
  nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
  pred_d2    <- predict(d2, alldata, nn_id_pred)
  
  RMSPE_d2 <- RMSPE(
    test_data$Y_mean,
    pred_d2$df_pred$pred_mean[1:nrow(test_data)]
  )
  CRPS_d2  <- CRPS(
    test_data$Y_mean,
    pred_d2$df_pred$pred_mean[1:nrow(test_data)],
    pred_d2$df_pred$pred_var[1:nrow(test_data)] + 1/d2$precy_tf
  )
  
  pred_objects <- c(pred_objects, "pred_d2", "RMSPE_d2", "CRPS_d2")
}

# -------------------------------------------------------------------
# Fit Models: Nonstationary (d3, d4)
# -------------------------------------------------------------------

if (fit_d3) {
  d3 <- deepspat_nn_ST_GP(
    f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ elev,
    family = "exp_nonstat_sep",
    layers_spat = layers_spat, layers_temp = layers_temp,
    m = 50L,
    order_id = order_id, nn_id = nn_id,
    method = "REML", nsteps = 50L,
    par_init = initvars(l_top_layer = 0.1),
    learn_rates = init_learn_rates(eta_mean = 0.003, LFTpars = 0.001)
  )
  
  locs_new   <- t(rbind(alldata$s1, alldata$s2, alldata$year))
  nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
  pred_d3    <- predict(d3, alldata, nn_id_pred)
  
  RMSPE_d3 <- RMSPE(
    test_data$Y_mean,
    pred_d3$df_pred$pred_mean[1:nrow(test_data)]
  )
  CRPS_d3  <- CRPS(
    test_data$Y_mean,
    pred_d3$df_pred$pred_mean[1:nrow(test_data)],
    pmax(
      pred_d3$df_pred$pred_var[1:nrow(test_data)] + as.numeric(1/d3$precy_tf),
      rep(1e-3, nrow(test_data))
    )
  )
  
  pred_objects <- c(pred_objects, "pred_d3", "RMSPE_d3", "CRPS_d3")
}

if (fit_d4) {
  d4 <- deepspat_nn_ST_GP(
    f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ 1,
    family = "exp_nonstat_sep",
    layers_spat = layers_spat, layers_temp = layers_temp,
    m = 50L,
    order_id = order_id, nn_id = nn_id,
    method = "REML", nsteps = 50L,
    par_init = initvars(l_top_layer = 0.1),
    learn_rates = init_learn_rates(eta_mean = 0.003, LFTpars = 0.001)
  )
  
  locs_new   <- t(rbind(alldata$s1, alldata$s2, alldata$year))
  nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
  pred_d4    <- predict(d4, alldata, nn_id_pred)
  
  RMSPE_d4 <- RMSPE(
    test_data$Y_mean,
    pred_d4$df_pred$pred_mean[1:nrow(test_data)]
  )
  CRPS_d4  <- CRPS(
    test_data$Y_mean,
    pred_d4$df_pred$pred_mean[1:nrow(test_data)],
    pmax(
      pred_d4$df_pred$pred_var[1:nrow(test_data)] + as.numeric(1/d4$precy_tf),
      rep(1e-3, nrow(test_data))
    )
  )
  
  pred_objects <- c(pred_objects, "pred_d4", "RMSPE_d4", "CRPS_d4")
}

# -------------------------------------------------------------------
# Save prediction objects / scores (for any fitted models)
# -------------------------------------------------------------------

if (length(pred_objects) > 0) {
  save(list = pred_objects,
       file = "Examples/Nepal_GP_pred_results.rda")
}

# -------------------------------------------------------------------
# For d3 only: precompute all quantities needed by plotting script
#              and save them in a separate .rda file
# -------------------------------------------------------------------

if (fit_d3) {
  
  # Year to visualise in the plots
  year_plot <- 2004L
  
  # Reference site indices (within the subset year == year_plot)
  ref.pts <- c(348, 363)
  
  # -----------------------------
  # 1) Warped elevation contours
  # -----------------------------
  df_elev <- data.frame(dataset) %>% distinct(s1, s2, elev)
  df_contour0 <- contoureR::getContourLines(df_elev, nlevels = 4)
  
  newdata_contour <- data.frame(
    s1   = df_contour0$x,
    s2   = df_contour0$y,
    elev = df_contour0$z,
    year = year_plot
  )
  locs_contour <- t(rbind(newdata_contour$s1,
                          newdata_contour$s2,
                          newdata_contour$year))
  nn_id_contour <- FNN::get.knnx(data = locs_t, query = locs_contour, k = 50)$nn.index
  pred_contour  <- predict(d3, newdata_contour, nn_id_contour)
  
  df_contour_plot <- df_contour0
  df_contour_plot$xw <- pred_contour$newdata_swarped[,1]
  df_contour_plot$yw <- pred_contour$newdata_swarped[,2]
  
  # -----------------------------
  # 2) Warped grid lines (vertical & horizontal) at year_plot
  # -----------------------------
  uni.lon <- unique(dataset$s1)
  uni.lat <- unique(dataset$s2)
  
  verti_list <- lapply(seq_along(uni.lon), function(i) {
    data_tmp    <- data.frame(dataset[dataset$s1 == uni.lon[i], ])
    data_tmp_un <- data_tmp %>% distinct(s1, s2, elev)
    data_tmp_ord <- data_tmp_un[order(data_tmp_un$s2), ]
    data_tmp_ord$year <- year_plot
    data_tmp_ord
  })
  
  horiz_list <- lapply(seq_along(uni.lat), function(i) {
    data_tmp    <- data.frame(dataset[dataset$s2 == uni.lat[i], ])
    data_tmp_un <- data_tmp %>% distinct(s1, s2, elev)
    data_tmp_ord <- data_tmp_un[order(data_tmp_un$s1), ]
    data_tmp_ord$year <- year_plot
    data_tmp_ord
  })
  
  df_verti_warped <- data.frame(
    do.call("rbind", lapply(verti_list, function(newdata_line) {
      locs_new_line <- t(rbind(newdata_line$s1,
                               newdata_line$s2,
                               newdata_line$year))
      nn_id_line <- FNN::get.knnx(data = locs_t,
                                  query = locs_new_line, k = 50)$nn.index
      pred_line <- predict(d3, newdata_line, nn_id_line)
      line_coords <- pred_line$newdata_swarped
      rbind(line_coords, c(NA, NA))
    }))
  )
  names(df_verti_warped) <- c("s1", "s2")
  
  df_horiz_warped <- data.frame(
    do.call("rbind", lapply(horiz_list, function(newdata_line) {
      locs_new_line <- t(rbind(newdata_line$s1,
                               newdata_line$s2,
                               newdata_line$year))
      nn_id_line <- FNN::get.knnx(data = locs_t,
                                  query = locs_new_line, k = 50)$nn.index
      pred_line <- predict(d3, newdata_line, nn_id_line)
      line_coords <- pred_line$newdata_swarped
      rbind(line_coords, c(NA, NA))
    }))
  )
  names(df_horiz_warped) <- c("s1", "s2")
  
  # -----------------------------
  # 3) Warped locations of sites for year_plot
  # -----------------------------
  dataset_year <- dataset %>% dplyr::filter(year == year_plot)
  locs_year <- t(rbind(dataset_year$s1,
                       dataset_year$s2,
                       dataset_year$year))
  nn_id_year <- FNN::get.knnx(data = locs_t,
                              query = locs_year, k = 50)$nn.index
  pred_year <- predict(d3, dataset_year, nn_id_year)
  S_warped_year <- as.data.frame(pred_year$newdata_swarped)
  names(S_warped_year) <- c("f1", "f2")
  
  # -----------------------------
  # 4) Correlation vectors for the two reference sites
  # -----------------------------
  newdata_year <- dataset_year
  K_year <- cov_fn_compute(d3, newdata_year, newdata_year)
  corr_ref1 <- as.vector(K_year[, ref.pts[1]])
  corr_ref2 <- as.vector(K_year[, ref.pts[2]])
  
  # -----------------------------
  # 5) Save everything needed by the plotting script
  # -----------------------------
  save(
    year_plot, ref.pts,
    df_contour_plot,
    df_verti_warped, df_horiz_warped,
    S_warped_year,
    corr_ref1, corr_ref2,
    file = "Examples/Nepal_GP_d3_plot_data.rda"
  )
}
