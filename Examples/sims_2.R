library("reticulate")
library("GpGp")
library(tensorflow)
library(keras)
library(tfprobability)
library(dplyr)
library(fields)
library(ggplot2)
library("devtools")
load_all("deepspat")
library(cocons)
data('holes')

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

rmspe_cocons <- RMSPE(holes_test$z, pred_cocons$mean + pred_cocons$trend)
crps_cocons <- CRPS(holes_test$z, pred_cocons$mean + pred_cocons$trend, pred_cocons$sd.pred^2)


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


pred_gp <- predict.deepspat_GP(d_gp, holes_test)
predall_gp <- predict.deepspat_GP(d_gp, holes_all)

rmspe_gp <- RMSPE(holes_test$z, pred_gp$df_pred$pred_mean)
crps_gp <- CRPS(holes_test$z, pred_gp$df_pred$pred_mean, pred_gp$df_pred$pred_var +
                  as.numeric(1/d_gp$precy_tf))

## nngp model
# Set up order and neighbor
locs <- as.matrix(holes_train)[, c("x", "y")]

# Order by max-min ordering
order_id <- order_maxmin(locs)
nn_id <- find_ordered_nn(order_id, m = 50) # increase number of neighbors from 50

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


nn_id_pred <- FNN::get.knnx(data = locs,
                            query = as.matrix(holes_test[,c("x", "y")]),
                            k = 50)$nn.index
pred_nngp <- predict.deepspat_nn_GP(d_nngp, holes_test, nn_id_pred)

nn_id_pred <- FNN::get.knnx(data = locs,
                            query = as.matrix(holes_all[,c("x", "y")]),
                            k = 50)$nn.index
predall_nngp <- predict.deepspat_nn_GP(d_nngp, holes_all, nn_id_pred)


rmspe_nngp <- RMSPE(holes_test$z, pred_nngp$df_pred$pred_mean)
crps_nngp <- CRPS(holes_test$z, pred_nngp$df_pred$pred_mean, pred_nngp$df_pred$pred_var + 
                    as.numeric(1/d_nngp$precy_tf))

## frk model
layers <- c(layers_gp,
            bisquares2D(r = 400L))

d_frk <- deepspat(f = z ~ x + y - 1, data = holes_train, layers = layers,
                  method = "ML", nsteps = 50L,
                  learn_rates = init_learn_rates(eta_mean = 0.02)) 

pred_frk <- predict.deepspat(d_frk, holes_test)
predall_frk <- predict.deepspat(d_frk, holes_all)

rmspe_frk <- RMSPE(holes_test$z, pred_frk$df_pred$pred_mean)
crps_frk <- CRPS(holes_test$z, pred_frk$df_pred$pred_mean, pred_frk$df_pred$pred_var +
                   as.numeric(1/d_frk$precy_tf))


### save results
save(pred_cocons, pred_gp, pred_nngp, pred_frk, pred_gstat,
     predall_cocons, predall_gp, predall_nngp, predall_frk,
     rmspe_cocons, rmspe_gp, rmspe_nngp, rmspe_frk, rmspe_gstat,
     crps_cocons, crps_gp, crps_nngp, crps_frk, crps_gstat,
     file = "sim_results_from_cocons.rda")




