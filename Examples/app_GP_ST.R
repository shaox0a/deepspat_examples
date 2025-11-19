rm(list = ls())

setwd(this.path::here())

library(reticulate)
library(tensorflow)
library(tfprobability)
library(keras)
library(dplyr)
library(fields)
library(ggplot2)
library(GpGp)
library(devtools)
library(deepspat)

######
# RMSPE
RMSPE <- function(true, pred){
  sqrt(mean((true - pred)^2))
}

# CRPS
library(verification)
CRPS <- function(true, pred, pred_var){
  crps(true, cbind(pred, sqrt(pred_var)))$CRPS
}

# Cov_fn_compute
cov_fn_compute <- function(object, newdata1, newdata2, ...) {
  
  d <- object
  
  mmat1 <- model.matrix(update(d$f, NULL ~ .), newdata1)
  X1_new1 <- model.matrix(update(d$g, NULL ~ .), newdata1)
  X_new1 <- tf$constant(X1_new1, dtype="float32")
  
  mmat2 <- model.matrix(update(d$f, NULL ~ .), newdata2)
  X1_new2 <- model.matrix(update(d$g, NULL ~ .), newdata2)
  X_new2 <- tf$constant(X1_new2, dtype="float32")
  
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
  
  if (d$family %in% c("exp_stat_sep")){
    obs_swarped <- d$swarped_tf
    newdata_swarped1 <- s_tf1
    newdata_swarped2 <- s_tf2
    
    obs_twarped <- d$twarped_tf
    newdata_twarped1 <- t_tf1
    newdata_twarped2 <- t_tf2
    
  }
  
  if (d$family %in% c("exp_nonstat_sep")){
    
    s_in1 <- scale_0_5_tf(s_tf1, d$scalings[[1]]$min, d$scalings[[1]]$max)
    t_in1 <- scale_0_5_tf(t_tf1, d$scalings_t[[1]]$min, d$scalings_t[[1]]$max)
    
    s_in2 <- scale_0_5_tf(s_tf2, d$scalings[[1]]$min, d$scalings[[1]]$max)
    t_in2 <- scale_0_5_tf(t_tf2, d$scalings_t[[1]]$min, d$scalings_t[[1]]$max)
    
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
  
  # I <- tf$eye(m) %>% tf$reshape(c(1L, m, m)) %>% tf$tile(c(npred, 1L, 1L)) 
  
  K <- cov_exp_tf(x1 = newdata_swarped1, x2 = newdata_swarped2, sigma2f = 1L, alpha = 1/d$l_tf)
  
  return(K)
  
}


## Load dataset
names = load("NepalExtended_mean.rda")

# plot(dataset[1:1419,c("s1", "s2")]); points(dataset[c(348,363),c("s1", "s2")], col="red")

meanY <- mean(dataset$Y_mean)
sdY <- sd(dataset$Y_mean)

dataset$Y_mean <- (dataset$Y_mean - meanY)/sdY

## Subsample
# dataset <- dataset |> filter(year < 2014)
set.seed(1)
sam2 <- sample(1:nrow(dataset), 0.2 * nrow(dataset))  # 0.8
train_data <- dataset[sam2,]
test_data <- dplyr::setdiff(dataset, train_data)

obsdata = train_data
newdata = test_data
alldata = rbind(test_data, train_data)

## Set up warping layers
layers_spat <- c(AWU(r = 100L, dim = 1L, grad = 20),
                 AWU(r = 100L, dim = 2L, grad = 20),
                 RBF_block(res = 1L),
                 RBF_block(res = 2L),
                 LFT()
)

layers_temp <- c(AWU(r = 20L, dim = 1L, grad = 20))

locs <- t(rbind(obsdata$s1, obsdata$s2))
locs_t <- t(rbind(obsdata$s1, obsdata$s2, obsdata$year))
# Order by random
set.seed(10)
order_id <- sample(1:nrow(locs))
# Nearest neighbor in space and time
nn_id <- find_ordered_nn(locs_t[order_id,], m = 50)
m <- ncol(nn_id) - 1
n <- nrow(nn_id)
for (i in 1:m){
  nn_id[i, (i+1):(m+1)] <- (n+1):(n+1+m-i)
}

# Fit Models: Stationary
# ------------------------------------------------------------------------------
d1 <- deepspat_nn_ST_GP(f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ elev,
                        family = "exp_stat_sep",
                        layers_spat = layers_spat, layers_temp = layers_temp,
                        m = 50L,
                        order_id = order_id, nn_id = nn_id,
                        method = "REML", nsteps = 50L,
                        par_init = initvars(l_top_layer = 0.5),
                        learn_rates = init_learn_rates(eta_mean = 0.01))

# Predictions
locs_new <- t(rbind(alldata$s1, alldata$s2, alldata$year))
nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
pred_d1 <- predict(d1, alldata, nn_id_pred)
RMSPE_d1 <- RMSPE(test_data$Y_mean, pred_d1$df_pred$pred_mean[1:nrow(test_data)])
CRPS_d1 <- CRPS(test_data$Y_mean,
                        pred_d1$df_pred$pred_mean[1:nrow(test_data)],
                        pred_d1$df_pred$pred_var[1:nrow(test_data)] + 1/d1$precy_tf)


d2 <- deepspat_nn_ST_GP(f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ 1,
                        family = "exp_stat_sep",
                        layers_spat = layers_spat, layers_temp = layers_temp,
                        m = 50L,
                        order_id = order_id, nn_id = nn_id,
                        method = "REML", nsteps = 50L,
                        par_init = initvars(l_top_layer = 0.5),
                        learn_rates = init_learn_rates(eta_mean = 0.01))

# Predictions
locs_new <- t(rbind(alldata$s1, alldata$s2, alldata$year))
nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
pred_d2 <- predict(d2, alldata, nn_id_pred)
RMSPE_d2 <- RMSPE(test_data$Y_mean, pred_d2$df_pred$pred_mean[1:nrow(test_data)])
CRPS_d2 <- CRPS(test_data$Y_mean,
                     pred_d2$df_pred$pred_mean[1:nrow(test_data)],
                     pred_d2$df_pred$pred_var[1:nrow(test_data)] + 1/d2$precy_tf)


# Fit Models: Nonstationary
# ------------------------------------------------------------------------------
d3 <- deepspat_nn_ST_GP(f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ elev,
                        family = "exp_nonstat_sep",
                        layers_spat = layers_spat, layers_temp = layers_temp,
                        m = 50L,
                        order_id = order_id, nn_id = nn_id,
                        method = "REML", nsteps = 50L,
                        par_init = initvars(l_top_layer = 0.1),
                        learn_rates = init_learn_rates(eta_mean = 0.003, LFTpars = 0.001))

# Predictions
locs_new <- t(rbind(alldata$s1, alldata$s2, alldata$year))
nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
pred_d3 <- predict(d3, alldata, nn_id_pred)
RMSPE_d3 <- RMSPE(test_data$Y_mean, pred_d3$df_pred$pred_mean[1:nrow(test_data)])
CRPS_d3 <- CRPS(test_data$Y_mean,
                     pred_d3$df_pred$pred_mean[1:nrow(test_data)],
                     pmax(pred_d3$df_pred$pred_var[1:nrow(test_data)] + as.numeric(1/d3$precy_tf), rep(1e-3, nrow(test_data)))
)

d4 <- deepspat_nn_ST_GP(f = Y_mean ~ s1 + s2 + year - 1, data = obsdata, g = ~ 1,
                        family = "exp_nonstat_sep",
                        layers_spat = layers_spat, layers_temp = layers_temp,
                        m = 50L,
                        order_id = order_id, nn_id = nn_id,
                        method = "REML", nsteps = 50L,
                        par_init = initvars(l_top_layer = 0.1),
                        learn_rates = init_learn_rates(eta_mean = 0.003, LFTpars = 0.001))

# Predictions
locs_new <- t(rbind(alldata$s1, alldata$s2, alldata$year))
nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
pred_d4 <- predict(d4, alldata, nn_id_pred)
RMSPE_d4 <- RMSPE(test_data$Y_mean, pred_d4$df_pred$pred_mean[1:nrow(test_data)])
CRPS_d4 <- CRPS(test_data$Y_mean,
                pred_d4$df_pred$pred_mean[1:nrow(test_data)],
                pmax(pred_d4$df_pred$pred_var[1:nrow(test_data)] + as.numeric(1/d4$precy_tf), rep(1e-3, nrow(test_data)))
)




save(pred_d1, pred_d2, pred_d3, pred_d4,
     RMSPE_d1, RMSPE_d2, RMSPE_d3, RMSPE_d4,
     CRPS_d1, CRPS_d2, CRPS_d3, CRPS_d4,
     file = "Nepal_GP_pred_results.rda")



# ==============================================================================
library(ggpubr)
library(ggnewscale)
library(RColorBrewer)
library(viridis)
library(grid)
library(gridExtra)
pic_path = "Pic_nepal_GP_ST/"
if (!dir.exists(pic_path)) {dir.create(pic_path)}

width1 = 11.5
unit.w1 = unit(width1, "cm")
width2 = 8
unit.w2 = unit(width2, "cm")
height1 = 8
unit.h1 = unit(height1, "cm")
ref_shap = 21
ref_shap1 = 8
axis.title.size = 16
axis.text.size = 16
legend.text.size = 15
legend.title.size = 16
text.size = 5


ref.pts = c(348, 363)
# pred_dtmp = pred_d3
# d_plot = d3

year = 2004
# ==============================================================================
# extract elevation information
df_elev = data.frame(dataset) %>% distinct(s1, s2, elev)
df_contour0 = contoureR::getContourLines(df_elev, nlevels = 4)
df_contour_rep = do.call(rbind, replicate(16, 
                                          data.frame(s1 = df_contour0$x, 
                                                     s2 = df_contour0$y,
                                                     elev = df_contour0$z), 
                                          simplify = FALSE))
df_contour_rep$year = rep(2004:2019, each = nrow(df_contour0))
locs_contour <- t(rbind(df_contour_rep$s1, df_contour_rep$s2, df_contour_rep$year))
nn_id_contour <- FNN::get.knnx(data = locs_t, query = locs_contour, k = 50)$nn.index
df_contour.warped = predict(d3, df_contour_rep, nn_id_contour)$newdata_swarped


df_contour = df_contour0
df_contour$xw = df_contour.warped[1:nrow(df_contour0) + (year-2004)*nrow(df_contour0),1]
df_contour$yw = df_contour.warped[1:nrow(df_contour0) + (year-2004)*nrow(df_contour0),2]


locs_new <- t(rbind(dataset$s1, dataset$s2, dataset$year))
nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
pred_d3hat <- predict(d3, dataset, nn_id_pred)
# ==============================================================================
### Plot priginal space

uni.lon = unique(dataset$s1)
uni.lat = unique(dataset$s2)

verti = lapply(1:length(uni.lon), function(i) {
  data_tmp = data.frame(dataset[which(dataset$s1 == uni.lon[i]), ])
  data_tmp_uni = data_tmp %>% distinct(s1, s2)
  data_tmp_ord = data_tmp_uni[order(data_tmp_uni$s2),]
})
horiz = lapply(1:length(uni.lat), function(i) {
  data_tmp = data.frame(dataset[which(dataset$s2 == uni.lat[i]), ])
  data_tmp_uni = data_tmp %>% distinct(s1, s2)
  data_tmp_ord = data_tmp_uni[order(data_tmp_uni$s1),]
})
df_verti = data.frame(do.call("rbind", lapply(1:length(verti), function(i) rbind(verti[[i]], c(NA, NA))) ))
df_horiz = data.frame(do.call("rbind", lapply(1:length(horiz), function(i) rbind(horiz[[i]], c(NA, NA))) ))

S.plot = data.frame(dataset[dataset$year==year,c("s1", "s2")])
grid1 = ggplot(df_verti, aes(x=s1,y=s2)) + geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(data = df_horiz, mapping = aes(x=s1,y=s2), colour = "gray80",
            inherit.aes = FALSE, linewidth = 0.4) +
  geom_path(data = df_contour, aes(x,y,group=Group, colour=z), linewidth = 0.8, inherit.aes = FALSE) +
  scale_color_viridis("Elevation (m)", discrete = F,
                      breaks = c(1000, 3000, 5000), labels = c(1000, 3000, 5000)) +
  geom_point(data = data.frame(x = S.plot[ref.pts,1], y = S.plot[ref.pts,2]),
             aes(x, y), size = 2, shape = ref_shap, fill="red", color="black") +
  xlab("Longitude") + ylab("Latitude") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size),
        legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))
  # scale_x_continuous(breaks = NULL) + scale_y_continuous(breaks = NULL)
grid1

legend_grob <- get_legend(grid1)
grid1hat = grid.arrange(grid1+theme(legend.position = "none"), legend_grob,
                        ncol = 2,
                        widths = unit.c(unit.w1, unit(width1/3, "cm")),
                        heights = unit.c(unit.h1)) #c(width1, width1/3)


ggsave(paste0(pic_path, "original_space.pdf"),
       plot = grid1hat, width = width1+width1/3, height = height1, units = "cm")


# ==============================================================================
### Plot warped space

verti = lapply(1:length(uni.lon), function(i) {
  data_tmp = data.frame(dataset[which(dataset$s1 == uni.lon[i]), ])
  data_tmp_uni = data_tmp %>% distinct(s1, s2, elev)
  data_tmp_ord = data_tmp_uni[order(data_tmp_uni$s2),]
  data_tmp_rep = do.call(rbind, replicate(16, data_tmp_ord, simplify = FALSE))
  data_tmp_rep$year = rep(2004:2019, each = nrow(data_tmp_uni))
  data_tmp_rep
})
horiz = lapply(1:length(uni.lat), function(i) {
  data_tmp = data.frame(dataset[which(dataset$s2 == uni.lat[i]), ])
  data_tmp_uni = data_tmp %>% distinct(s1, s2, elev)
  data_tmp_ord = data_tmp_uni[order(data_tmp_uni$s1),]
  data_tmp_rep = do.call(rbind, replicate(16, data_tmp_ord, simplify = FALSE))
  data_tmp_rep$year = rep(2004:2019, each = nrow(data_tmp_uni))
  data_tmp_rep
})
# you may change the gap in seq to decide the dense
df_verti = data.frame(do.call("rbind", lapply(seq(1, length(verti), 1), function(i) {
  newdata = verti[[i]]
  locs_new <- t(rbind(newdata$s1, newdata$s2, newdata$year))
  nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
  swarped_line = predict(d3, newdata, nn_id_pred)$newdata_swarped
  swarped_line = swarped_line[newdata$year == year,]
  rbind(swarped_line, c(NA, NA))
}) ))
df_horiz = data.frame(do.call("rbind", lapply(seq(1, length(horiz), 1), function(i) {
  newdata = horiz[[i]]
  locs_new <- t(rbind(newdata$s1, newdata$s2, newdata$year))
  nn_id_pred <- FNN::get.knnx(data = locs_t, query = locs_new, k = 50)$nn.index
  swarped_line = predict(d3, newdata, nn_id_pred)$newdata_swarped
  swarped_line = swarped_line[newdata$year == year,]
  rbind(swarped_line, c(NA, NA))
}) ))
names(df_verti) = names(df_horiz) = c("s1", "s2")



S.plot = pred_d3hat$newdata_swarped[dataset$year==year,]
grid2 = ggplot(df_verti, aes(x=s1,y=s2)) + geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(data = df_horiz, mapping = aes(x=s1,y=s2), colour = "gray80",
            inherit.aes = FALSE, linewidth = 0.4) +
  geom_path(data = df_contour, aes(xw,yw,group=Group, colour=z), size = 0.8, inherit.aes = FALSE) +
  scale_color_viridis("Elevation (m)", discrete = F,
                      breaks = c(1000, 3000, 5000), labels = c(1000, 3000, 5000)) +
  geom_point(data = data.frame(x = S.plot[ref.pts,1], y = S.plot[ref.pts,2]),
             aes(x, y), size = 2, shape = ref_shap, fill="red", color="black") +
  xlab(expression(f[n1])) + ylab(expression(f[n2])) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size),
        legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))
  # scale_x_continuous(breaks = NULL) + scale_y_continuous(breaks = NULL)
grid2

legend_grob <- get_legend(grid2)
grid2hat = grid.arrange(grid2+theme(legend.position = "none"), legend_grob,
                        ncol = 2,
                        widths = unit.c(unit.w1, unit(width1/3, "cm")),
                        heights = unit.c(unit.h1)) #c(width1, width1/3)


ggsave(paste0(pic_path, "nepal_warped_space.pdf"),
       plot = grid2hat, width = width1+width1/3, height = height1, units = "cm")



legend_grob <- get_legend(grid1)
grid3hat = grid.arrange(grid1+theme(legend.position = "none"),
                        grid2+theme(legend.position = "none"), legend_grob,
                        ncol = 3,
                        widths = unit.c(unit.w1, unit.w1, unit(width1/3, "cm")),
                        heights = unit.c(unit.h1)) #c(width1, width1/3)
ggsave(paste0(pic_path, "nepal_spaces.pdf"),
       plot = grid3hat, width = width1+width1+width1/3, height = height1, units = "cm")

# ==============================================================================
### Plot covariance heat map

my_colors = RColorBrewer::brewer.pal(n=5, name="RdYlBu")[5:1]

newdata1 <- dataset %>% filter(year == year)
newdata2 <- dataset %>% filter(year == year)
K_1 <- cov_fn_compute(d3, newdata1, newdata1)


ref.point1 <- 348
## choose ref points 348, 363
## change palette to RdBu or BrBG for clearer difference


plot_corr1 <- ggplot(data = newdata2[1:1419+(year-2004)*1419,]) +
  geom_point(aes(s1, s2, color = as.vector(K_1[1:1419+(year-2004)*1419,ref.point1])),
             alpha = 0.9, size = 2, shape = 15) +
  scale_color_gradientn(colors = my_colors,
                        name = expression(Corr(Y(bold(s)[0]), Y(bold(s)))), limits = c(0,1),
                        breaks = seq(0,1,0.25),
                        labels = c("0.00", "0.25", "0.50", "0.75", "1.00")) +
  geom_point(aes(x = s1[ref.point1], y = s2[ref.point1]),
             size = 2, shape = ref_shap, fill="red", color="black") +
  theme_bw() + coord_fixed() +
  xlab("Longitude") + ylab("Latitude") +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))
plot_corr1

legend_grob <- get_legend(plot_corr1)
plot_corr11 = grid.arrange(plot_corr1+theme(legend.position = "none"), legend_grob,
                      ncol = 2,
                      widths = unit.c(unit.w1, unit(width1/2.5, "cm")),
                      heights = unit.c(unit.h1)) #c(width1, width1/3)


ggsave(paste0(pic_path, "nepal_corr_", ref.point1, ".pdf"),
       plot = plot_corr11, width = width1+width1/3, height = height1, units = "cm")



ref.point2 <- 363
## choose ref points 348, 363
## change palette to RdBu or BrBG for clearer difference


plot_corr2 <- ggplot(data = newdata2[1:1419+(year-2004)*1419,]) +
  geom_point(aes(s1, s2, color = as.vector(K_1[1:1419+(year-2004)*1419,ref.point2])),
             alpha = 0.9, size = 2, shape = 15) +
  scale_color_gradientn(colors = my_colors,
                        name = expression(Corr(Y(bold(s)[0]), Y(bold(s)))), limits = c(0,1),
                        breaks = seq(0,1,0.25),
                        labels = c("0.00", "0.25", "0.50", "0.75", "1.00")) +
  geom_point(aes(x = s1[ref.point2], y = s2[ref.point2]),
             size = 2, shape = ref_shap, fill="red", color="black") +
  theme_bw() + coord_fixed() +
  xlab("Longitude") + ylab("Latitude") +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))
plot_corr2

legend_grob <- get_legend(plot_corr2)
plot_corr21 = grid.arrange(plot_corr2+theme(legend.position = "none"), legend_grob,
                           ncol = 2,
                           widths = unit.c(unit.w1, unit(width1/2.5, "cm")),
                           heights = unit.c(unit.h1)) #c(width1, width1/3)


ggsave(paste0(pic_path, "nepal_corr_", ref.point2, ".pdf"),
       plot = plot_corr21, width = width1+width1/3, height = height1, units = "cm")




legend_grob <- get_legend(plot_corr1)
plot_corr = grid.arrange(plot_corr1+theme(legend.position = "none"),
                         plot_corr2+theme(legend.position = "none"), legend_grob,
                        ncol = 3,
                        widths = unit.c(unit.w1, unit.w1, unit(width1/2.5, "cm")),
                        heights = unit.c(unit.h1)) #c(width1, width1/3)
ggsave(paste0(pic_path, "nepal_corr", ".pdf"),
       plot = plot_corr, width = width1+width1+width1/2.5, height = height1, units = "cm")

# plot_temp <- ggplot(data = newdata2) +
#   geom_tile(aes(s1, s2, fill = Y_mean )) +
#   scale_fill_distiller(palette = "BrBG", name = "Temp", limits = c(-2.5,2.5), direction = -1) +
#   geom_point(aes(x = s1[ref.point], y = s2[ref.point]),
#              colour = "red", size = 1, shape = 0) +
#   #scale_fill_viridis(option = "G", name = "Corr", limits = c(0,1), direction = -1) +
#   theme_bw() + coord_fixed()
# 
# ggsave(filename="nepal_gaussian_plot.png", plot=plot_corr,
#        device="png", width=25, height=15, scale=1, units="cm", dpi=300)


# plot_temp <- ggplot(data = newdata2[1:1419+(year-2004)*1419,]) +
#   geom_point(aes(s1, s2, color = Y_mean),
#              alpha = 0.9, size = 2, shape = 15) +
#   scale_color_gradientn(colors = my_colors,
#                         name = "Temp", limits = c(-2.5,2.5)) +
#   geom_point(aes(x = s1[ref.point], y = s2[ref.point]),
#              colour = "black", size = 1, shape = 0) +
#   theme_bw() + coord_fixed() +
#   xlab("Longitude") + ylab("Latitude") +
#   theme(plot.title = element_text(hjust = 0.5, size=25),
#         axis.title=element_text(size=axis.title.size),
#         axis.text = element_text(size=axis.text.size),
#         axis.line = element_blank(),
#         axis.ticks = element_blank(),
#         legend.key.size = unit(0.25, "in"),
#         legend.text = element_text(size=legend.text.size),
#         legend.title = element_text(size=legend.title.size),
#         legend.position = "right", #c(0.18, 0.8),
#         legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
#         legend.direction = "vertical",
#         legend.box = "vertical",
#         legend.spacing.y = unit(0.4, "lines"),
#         legend.margin = margin(5, 5, 5, 5))
# plot_temp
# 
# legend_grob <- get_legend(plot_temp)
# plot_temp1 = grid.arrange(plot_temp+theme(legend.position = "none"), legend_grob,
#                           ncol = 2,
#                           widths = unit.c(unit.w1, unit(width1/3, "cm")),
#                           heights = unit.c(unit.h1)) #c(width1, width1/3)
# 
# 
# ggsave(paste0(pic_path, "nepal_temp_", ref.point, ".pdf"),

#        plot = plot_temp1, width = width1+width1/3, height = height1, units = "cm")

