examples_path <- NULL
# Specify the path to the deepspat examples directory.
deepspat_path <- NULL
# Set `deepspat_path` to a local package path, or NULL to use library(deepspat).
if (!is.null(examples_path)) {
  setwd(examples_path)
}

if (is.null(deepspat_path)) {
  library(deepspat)
} else {
  pkgload::load_all(
    deepspat_path,
    quiet = TRUE
  )
}
library(GpGp)
library(tensorflow)
library(keras)
library(tfprobability)
library(dplyr)
library(fields)
library(ggplot2)
library(cocons)
data('holes')

message("Currently running: Fitting the models for simulation study with data from cocons")

# RMSPE
RMSPE <- function(true, pred){
  sqrt(mean((true - pred)^2))
}

# CRPS
library(verification)
CRPS <- function(true, pred, pred_var){
  crps(true, cbind(pred, sqrt(pred_var)))$CRPS
}


### Sample data for sims from cocons package

holes_all <- data.frame(rbind(holes$training, holes$test))

RNGkind(sample.kind = "Rounding")
set.seed(1)
holes_train <- holes_all[sample(1:nrow(holes_all), 1500),]
holes_test <- setdiff(holes_all, holes_train)

### Fit models

## cocons model
model.list <- list ("mean" = formula(~ 1),
                    "std.dev" = formula(~ 1 + cov_x + cov_y ),
                    "scale" = formula(~ 1 + cov_x + cov_y ),
                    "aniso" = 0,
                    "tilt" = 0,
                    "smooth" = 1/2,
                    "nugget" = -Inf)

coco_object <- coco ( type = "dense",
                      model.list = model.list,
                      locs = as.matrix ( holes_train[,1:2]),
                      z = holes_train$z,
                      data = holes_train)

coco_object <- cocoOptim (coco_object, ncores = "auto")

pred_cocons <- cocoPredict (coco_object,
                            newdataset = holes_test,
                            newlocs = as.matrix(holes_test[,1:2]),
                            type = "pred")

predall_cocons <- cocoPredict (coco_object,
                               newdataset = holes_all,
                               newlocs = as.matrix(holes_all[,1:2]),
                               type = "pred")

rmspe_cocons <- RMSPE(holes_test$z, pred_cocons$stochastic + pred_cocons$systematic)
crps_cocons <- CRPS(holes_test$z, 
                    pred_cocons$stochastic + pred_cocons$systematic, 
                    pred_cocons$sd.pred^2)



## gstat model
library(sp)
library(gstat)
sp_data <- data.frame(x = holes_train$x,
                      y = holes_train$y,
                      z = holes_train$z)
coordinates(sp_data) <- ~ x + y

# Fit variogram
vgm_exp <- variogram(z ~ 1, data = sp_data)
vgm_model <- vgm(model = "Exp", nugget = 0.1)
vgm_fit <- fit.variogram(vgm_exp, model = vgm_model)

# Pred
pred_loc <- data.frame(x = holes_test$x, y = holes_test$y)
coordinates(pred_loc) <- ~ x + y
krig_model <- gstat(formula = z ~ 1, data = sp_data, model = vgm_fit)
pred_gstat <- predict(krig_model, pred_loc)

rmspe_gstat <- RMSPE(holes_test$z, pred_gstat@data$var1.pred)
crps_gstat <- CRPS(holes_test$z, pred_gstat@data$var1.pred, pred_gstat@data$var1.var)

## deepspat models
# Set up warping layers
layers_gp <- c(AWU(r = 50L, dim = 1L, grad = 50, lims = c(-0.5, 0.5)),
               AWU(r = 50L, dim = 2L, grad = 50, lims = c(-0.5, 0.5)),
               RBF_block(),
               LFT())

## gp model
d_gp <- deepspat_GP(f = z ~ x + y - 1,
                    data = holes_train,
                    g = ~ 1,
                    layers = layers_gp,
                    method = "REML",
                    family = "exp_nonstat",
                    nsteps = 50L, # 150L,
                    par_init = initvars(l_top_layer = 0.5),
                    learn_rates = init_learn_rates(eta_mean = 0.02)
)

print(d_gp)
d_gp_summary <- summary(d_gp)
print(d_gp_summary)

pred_gp <- predict(d_gp, holes_test, type = "process")
predall_gp <- predict(d_gp, holes_all, type = "process")

rmspe_gp <- RMSPE(holes_test$z, pred_gp$df_pred$pred_mean)
crps_gp <- CRPS(holes_test$z, pred_gp$df_pred$pred_mean, pred_gp$df_pred$pred_var +
                  as.numeric(1/d_gp$precy_tf))

# Examples of the new S3 plot methods. These are quick checks only
# and are not saved.
plot_data_gp <- holes_test[seq_len(min(500L, nrow(holes_test))), ]
pred_cov_gp <- predict(d_gp, plot_data_gp, type = "covariance",
                       reference = 1L)
plot(d_gp, type = "space", pred = predall_gp)
plot(d_gp, type = "prediction", pred = pred_gp)
plot(d_gp, type = "covariance", pred = pred_cov_gp,
     value = "correlation")
plot(d_gp, type = "covariance", pred = pred_cov_gp,
     value = "covariance")

## nngp model
# Set up order and neighbor
locs <- as.matrix(holes_train)[, c("x", "y")]

# Order by max-min ordering
order_id <- order_maxmin(locs)
nn_id <- find_ordered_nn(order_id, m = 50) # increase number of neighbors from 50
m <- ncol(nn_id) - 1
n <- nrow(nn_id)
for (i in 1:m){
  nn_id[i, (i+1):(m+1)] <- (n+1):(n+1+m-i)
}

d_nngp <- deepspat_nn_GP(f = z ~ x + y - 1,
                         data = holes_train,
                         g = ~ 1,
                         layers = layers_gp,
                         m = 50L,
                         order_id = order_id,
                         nn_id = nn_id,
                         method = "REML",
                         family = "exp_nonstat",
                         nsteps = 50L,
                         par_init = initvars(l_top_layer = 0.5),
                         learn_rates = init_learn_rates(eta_mean = 0.02))

print(d_nngp)
d_nngp_summary <- summary(d_nngp)
print(d_nngp_summary)

nn_id_pred <- FNN::get.knnx(data = locs,
                            query = as.matrix(holes_test[,c("x", "y")]),
                            k = 50)$nn.index
pred_nngp <- predict(d_nngp, holes_test, nn_id = nn_id_pred,
                     type = "process")

nn_id_pred <- FNN::get.knnx(data = locs,
                            query = as.matrix(holes_all[,c("x", "y")]),
                            k = 50)$nn.index
predall_nngp <- predict(d_nngp, holes_all, nn_id = nn_id_pred,
                        type = "process")


rmspe_nngp <- RMSPE(holes_test$z, pred_nngp$df_pred$pred_mean)
crps_nngp <- CRPS(holes_test$z, pred_nngp$df_pred$pred_mean, pred_nngp$df_pred$pred_var + 
                    as.numeric(1/d_nngp$precy_tf))

pred_cov_nngp <- predict(d_nngp, plot_data_gp, type = "covariance",
                         reference = 1L)
plot(d_nngp, type = "space", pred = predall_nngp)
plot(d_nngp, type = "prediction", pred = pred_nngp)
plot(d_nngp, type = "covariance", pred = pred_cov_nngp,
     value = "correlation")
plot(d_nngp, type = "covariance", pred = pred_cov_nngp,
     value = "covariance")

## frk model
layers <- c(layers_gp,
            bisquares2D(r = 400L))

d_frk <- deepspat(f = z ~ x + y - 1, data = holes_train, layers = layers,
                  method = "ML", nsteps = 50L,
                  learn_rates = init_learn_rates(eta_mean = 0.02)) 

pred_frk <- predict(d_frk, holes_test)
predall_frk <- predict(d_frk, holes_all)

rmspe_frk <- RMSPE(holes_test$z, pred_frk$df_pred$pred_mean)
crps_frk <- CRPS(holes_test$z, pred_frk$df_pred$pred_mean, pred_frk$df_pred$pred_var +
                   as.numeric(1/d_frk$precy_tf))


### save results
save(pred_cocons, pred_gp, pred_nngp, pred_frk, pred_gstat,
     predall_cocons, predall_gp, predall_nngp, predall_frk,
     rmspe_cocons, rmspe_gp, rmspe_nngp, rmspe_frk, rmspe_gstat,
     crps_cocons, crps_gp, crps_nngp, crps_frk, crps_gstat,
     file = "Examples/Data/sim_results_from_cocons.RData")
