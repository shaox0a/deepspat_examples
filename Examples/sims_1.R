library(reticulate)
library(GpGp)
library(tensorflow)
library(keras)
library(tfprobability)
library(dplyr)
library(fields)
library(ggplot2)
library(devtools)
library(deepspat)
library(cocons)
#data('holes')

# RMSPE
RMSPE <- function(true, pred){
  sqrt(mean((true - pred)^2))
}

# CRPS
library(verification)
CRPS <- function(true, pred, pred_var){
  crps(true, cbind(pred, sqrt(pred_var)))$CRPS
}

### Sample data for sims from deepspat package

set.seed(1)
deepspat_data <- sim_data(type = "AWU_RBF_2D", ds = 0.01, n = 6000L, sigma2y = 0.01)

deepspat_data_all <- data.frame(x = deepspat_data$sobs[,1],
                                y = deepspat_data$sobs[,2],
                                z = deepspat_data$y)

RNGkind(sample.kind = "Rounding")
deepspat_data_train <- deepspat_data_all[sample(1:nrow(deepspat_data_all), 1500),]
deepspat_data_test <- setdiff(deepspat_data_all, deepspat_data_train)

save(deepspat_data_all, deepspat_data_train, deepspat_data_test,
     file = "sim_deepspat_datasets.rda")

### Fit models

## gstat model
library(sp)
library(gstat)
sp_data <- data.frame(x = deepspat_data_train$x,
                      y = deepspat_data_train$y,
                      z = deepspat_data_train$z)
coordinates(sp_data) <- ~ x + y

# Fit variogram
vgm_exp <- variogram(z ~ 1, data = sp_data)
vgm_model <- vgm(model = "Exp", nugget = 0.1)
vgm_fit <- fit.variogram(vgm_exp, model = vgm_model)

# Pred
pred_loc <- data.frame(x = deepspat_data_test$x, y = deepspat_data_test$y)
coordinates(pred_loc) <- ~ x + y
krig_model <- gstat(formula = z ~ 1, data = sp_data, model = vgm_fit)
pred_gstat <- predict(krig_model, pred_loc)

rmspe_gstat <- RMSPE(deepspat_data_test$z, pred_gstat@data$var1.pred)
crps_gstat <- CRPS(deepspat_data_test$z, pred_gstat@data$var1.pred, pred_gstat@data$var1.var)

## deepspat models
# Set up warping layers
layers_gp <- c(AWU(r = 50L, dim = 1L, grad = 50, lims = c(-0.5, 0.5)),
               AWU(r = 50L, dim = 2L, grad = 50, lims = c(-0.5, 0.5)),
               RBF_block(),
               LFT())

## gp model
d_gp <- deepspat_GP(f = z ~ x + y - 1,
                    data = deepspat_data_train,
                    g = ~ 1,
                    layers = layers_gp,
                    method = "REML",
                    family = "exp_nonstat",
                    nsteps = 50L, # 150L,
                    par_init = initvars(l_top_layer = 0.5),
                    learn_rates = init_learn_rates(eta_mean = 0.02)
)


pred_gp <- predict(d_gp, deepspat_data_test)
predall_gp <- predict(d_gp, deepspat_data_all)

rmspe_gp <- RMSPE(deepspat_data_test$z, pred_gp$df_pred$pred_mean)
crps_gp <- CRPS(deepspat_data_test$z, pred_gp$df_pred$pred_mean, pred_gp$df_pred$pred_var +
                  as.numeric(1/d_gp$precy_tf))

## nngp model
# Set up order and neighbor
locs <- as.matrix(deepspat_data_train)[, c("x", "y")]

# Order by max-min ordering
order_id <- order_maxmin(locs)
nn_id <- find_ordered_nn(order_id, m = 50) # increase number of neighbors from 50
m <- ncol(nn_id) - 1
n <- nrow(nn_id)
for (i in 1:m){
  nn_id[i, (i+1):(m+1)] <- (n+1):(n+1+m-i)
}

d_nngp <- deepspat_nn_GP(f = z ~ x + y - 1,
                         data = deepspat_data_train,
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


nn_id_pred <- FNN::get.knnx(data = locs,
                            query = as.matrix(deepspat_data_test[,c("x", "y")]),
                            k = 50)$nn.index
pred_nngp <- predict(d_nngp, deepspat_data_test, nn_id_pred)

nn_id_pred <- FNN::get.knnx(data = locs,
                            query = as.matrix(deepspat_data_all[,c("x", "y")]),
                            k = 50)$nn.index
predall_nngp <- predict(d_nngp, deepspat_data_all, nn_id_pred)


rmspe_nngp <- RMSPE(deepspat_data_test$z, pred_nngp$df_pred$pred_mean)
crps_nngp <- CRPS(deepspat_data_test$z, pred_nngp$df_pred$pred_mean, pred_nngp$df_pred$pred_var + 
                    as.numeric(1/d_nngp$precy_tf))

## frk model
layers <- c(layers_gp,
            bisquares2D(r = 400L))

d_frk <- deepspat(f = z ~ x + y - 1, data = deepspat_data_train, layers = layers,
                  method = "ML", nsteps = 50L,
                  learn_rates = init_learn_rates(eta_mean = 0.02)) 

pred_frk <- predict(d_frk, deepspat_data_test)
predall_frk <- predict(d_frk, deepspat_data_all)

rmspe_frk <- RMSPE(deepspat_data_test$z, pred_frk$df_pred$pred_mean)
crps_frk <- CRPS(deepspat_data_test$z, pred_frk$df_pred$pred_mean, pred_frk$df_pred$pred_var +
                   as.numeric(1/d_frk$precy_tf))

### save results
save(pred_gp, pred_nngp, pred_frk, pred_gstat,
     predall_gp, predall_nngp, predall_frk,
     rmspe_gp, rmspe_nngp, rmspe_frk, rmspe_gstat,
     crps_gp, crps_nngp, crps_frk, crps_gstat,
     file = "sim_results_from_deepspat.rda")



## cocons model
model.list <- list ("mean" = formula(~ 1),
                    "std.dev" = formula(~ 1 + x + y ),
                    "scale" = formula(~ 1 + x + y ),
                    "aniso" = 0,
                    "tilt" = 0,
                    "smooth" = 1/2,
                    "nugget" = -Inf)

coco_object <- coco ( type = "dense",
                      model.list = model.list,
                      locs = as.matrix ( deepspat_data_train[,1:2]),
                      z = deepspat_data_train$z,
                      data = deepspat_data_train)

coco_object <- cocoOptim (coco_object, ncores = "auto")

pred_cocons <- cocoPredict (coco_object,
                            newdataset = deepspat_data_test,
                            newlocs = as.matrix(deepspat_data_test[,1:2]),
                            type = "pred")

predall_cocons <- cocoPredict (coco_object,
                               newdataset = deepspat_data_all,
                               newlocs = as.matrix(deepspat_data_all[,1:2]),
                               type = "pred")

rmspe_cocons <- RMSPE(deepspat_data_test$z, pred_cocons$mean + pred_cocons$trend)
crps_cocons <- CRPS(deepspat_data_test$z, pred_cocons$mean + pred_cocons$trend, pred_cocons$sd.pred^2)


### save results
save(pred_cocons, pred_gp, pred_nngp, pred_frk, pred_gstat,
     predall_cocons, predall_gp, predall_nngp, predall_frk,
     rmspe_cocons, rmspe_gp, rmspe_nngp, rmspe_frk, rmspe_gstat,
     crps_cocons, crps_gp, crps_nngp, crps_frk, crps_gstat,
     file = "sim_results_from_deepspat.rda")


