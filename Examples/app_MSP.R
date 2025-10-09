rm(list = ls())

setwd(this.path::here())


# library("deepspat")
library(tensorflow)
library(keras)
library(tfprobability)
library(dplyr)
library(fields)
library(maps)

# adapt to tf2
devtools::load_all("D:/Research/DCSMExt/deepspat")
source("utils_ext.R")


#################################################
# load data
model <- "MSP-BR"

# --------------------------------
app_data <- "NepalExtended" # "UKpr"
simnames <- load(file="NepalExtended.rds")
simnames

data <- Z.max

#################################################
df <- cbind(S, data) %>% as.data.frame() #
names(df) <- c("s1", "s2", paste0("z", 1:(ncol(df)-2)))
df_loc <- dplyr::select(df, s1, s2)
df_data <- df[,3:ncol(df)]


# ------------------------------------------------
# empirical extremal dependence measure estimates for all locarions
edm_est_filename <- paste0(app_data, "_", model, "_empextdep.rds")
if (!file.exists(edm_est_filename)) {
  all_edm_est <- edm_est(df_data, as.matrix(df_loc), model)
  saveRDS(all_edm_est, edm_est_filename)
}
# ------------------------------------------------


# RNGkind(sample.kind <- "Rounding")
seedn1 <- 1
set.seed(seedn1)
D_obs <- 500

sam1 <- sample(1:nrow(df), D_obs)

df.obs <- df[sam1,]
obs_loc <- df.obs[,c("s1", "s2")]
obs_data <- df.obs[,3:ncol(df)]

plot(obs_loc)

obs_all <- cbind(obs_loc, obs_data) %>% as.data.frame()


method <- "MRPL"
family <- "power_nonstat"
dtype <- "float64"


obs_edm_est <- edm_est(obs_data, as.matrix(obs_loc), model)$edm
saveRDS(obs_edm_est, file = paste0(app_data, "_", model, "_empextdep_train.rds"))

obs_edm_emp <- obs_edm_est[,1]


## Set up warping layers
r1 <- 50L
layers <- c(AWU(r = r1, dim = 1L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
            AWU(r = r1, dim = 2L, grad = 200, lims = c(-0.5, 0.5), dtype = dtype),
            RBF_block(1L, dtype = dtype),
            # RBF_block(2L, dtype = dtype),
            LFT(dtype = dtype))



d1 <- deepspat_MSP(f = as.formula(paste(paste(paste0("z", 1:(ncol(obs_all)-2)), collapse= "+"), "~ s1 + s2 -1")),
                   data = obs_all,
                   layers = layers,
                   method = method,
                   family = family,
                   dtype = dtype,
                   nsteps = 50L,
                   nsteps_pre = 50L,
                   par_init = initvars(),
                   learn_rates = init_learn_rates(eta_mean = 0.01, vario = 0.01),
                   edm_emp = obs_edm_emp,
                   p = 0.01)


# plot(as.matrix(d1$swarped))
# fitted.phi <- as.numeric(exp(d1$logphi_tf))
# fitted.kappa <- as.numeric(2*tf$sigmoid(d1$logitkappa_tf))
# cat(round(fitted.phi, 3), ",", round(fitted.kappa, 3))
# 
# pred <- summary(d1, df_loc, 
#                 uncAss = T, 
#                 edm_emp = obs_edm_emp)
# 
# swarped  <- pred$swarped
# plot(swarped)
# 
# cat(round(pred$fitted.kappa, 3), ",", round(pred$fitted.phi, 3))
# sqrt(diag(pred$Sigma_psi))

################################################################################
################################################################################
################################################################################

library(ggplot2)
library(verification)
library(ggmap)
library(ggpubr)
library(ggnewscale)
library(RColorBrewer)
library(viridis)
library(grid)
library(gridExtra)

# Loads objects such as S (locations), df, df_loc, nepal basemap, etc.
load(paste0(this.path::here(), "/NepalMap.Rdata"))

# Output directory for figures
pic_path = "Pic_nepal_MSP/"
if (!dir.exists(pic_path)) {dir.create(pic_path)}

# --- Quick checks of site ordering and reference points (visual sanity checks) ---
plot(S); points(S[c(348,363),], col="red")

Shat = S[order(S[,1]),]
plot(Shat); points(Shat[c(348,363),], col="red")
plot(S); points(S[order(S[,1])[c(348,363)],], col="red")

# --- Model summary & derived quantities from deepspat MSP fit ---
# pred aggregates key outputs at observation sites (and later at queried coordinates)
pred = summary(d1, df_loc)
S.rescaled = pred$srescaled               # Rescaled original coordinates (if applicable)
S.warped   = pred$swarped                 # Warped coordinates from fitted deformation f
range_fitted = pred$fitted.phi            # Fitted range (phi) parameter(s)
dof_fitted   = pred$fitted.kappa          # Fitted degrees-of-freedom (kappa) parameter(s)

# Pairwise distances in warped space (used for fitted EC curves)
D.warped = rdist(S.warped)

# Inspect stored warped grid (if available in d1) and the warped sites
plot(as.matrix(d1$swarped))
plot(S.warped)

# Number of replicates (e.g., months x years maxima); assumed available from environment
nrepli <- dim(Z.max)[2]

# --- Elevation contours (extracted at S and transformed to warped space for overlays) ---
lon_range=range(df_loc[,1])
lat_range=range(df_loc[,2])

# Elevation extraction (EPSG:4326) at original locations S
elev_extract = elevatr::get_elev_point(data.frame(x=S[,1], y=S[,2]),
                                       prj = 4326, src = "aws")
elev = elev_extract$elevation
elev[is.na(elev)] = 0

# Build contour lines from pointwise elevation field
df_elev = data.frame(s1 = S[,1], s2 = S[,2], elev = elev)
df_contour = contoureR::getContourLines(df_elev, nlevels = 4)

# Map each contour vertex into warped space for overlay on f(s)
df_contour.warped = summary(d1, data.frame(s1 = df_contour$x, s2 = df_contour$y), F)$swarped
df_contour$xw = df_contour.warped[,1]; df_contour$yw = df_contour.warped[,2]

# --- Figure layout constants ---
width1 = 11.5; unit.w1 = unit(width1, "cm")
width2 = 8;    unit.w2 = unit(width2, "cm")
height1 = 8;   unit.h1 = unit(height1, "cm")

ref_shap = 21           # Marker shape for reference sites
ref_shap1 = 8           # (Unused) alternative shape
axis.title.size = 16
axis.text.size  = 16
legend.text.size  = 15
legend.title.size = 16
text.size = 5           # (Unused text size placeholder)

# Two reference site indices used across maps
ref.pts = c(549, 1317)

# --- Basemap: elevation raster over Nepal with reference sites highlighted ---
p.elev <- ggmap(nepal) +
  geom_tile(df, mapping = aes(x = s1, y = s2, fill = elev), width = 0.25, height = 0.25) +
  geom_point(data = data.frame(x = S[ref.pts,1], y = S[ref.pts,2]),
             aes(x, y), size = 1, shape = ref_shap, fill="red", color="red") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size),
        legend.position = "right",
        legend.background = element_rect(fill='transparent'),
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5)) +
  xlab("Longitude") + ylab("Latitude") +
  scale_fill_gradientn("Elevation (m)", colours = terrain.colors(10))
print(p.elev)

ggsave(paste0(pic_path, "nepal_elev.pdf"),
       plot = p.elev, width = width1+width1/3, height = height1, units = "cm")

# --- Original space grid + contours + reference sites ---
S.plot = S
uni.lon = unique(S[,1])
uni.lat = unique(S[,2])

# Build vertical/horizontal grid lines by grouping sites with same lon/lat, ordered
verti = lapply(1:length(uni.lon), function(i) {
  data_tmp = data.frame(S[which(S[,1] == uni.lon[i]), ])
  data_tmp_ord = data_tmp[order(data_tmp[,2]),]
  names(data_tmp_ord) = c("s1", "s2")
  data_tmp_ord
})
horiz = lapply(1:length(uni.lat), function(i) {
  data_tmp = data.frame(S[which(S[,2] == uni.lat[i]), ])
  data_tmp_ord = data_tmp[order(data_tmp[,1]),]
  names(data_tmp_ord) = c("s1", "s2")
  data_tmp_ord
})

# Insert NA separators so geom_path breaks between polylines
df_verti = data.frame(do.call("rbind", lapply(1:length(verti), function(i) rbind(verti[[i]], c(NA, NA))) ))
df_horiz = data.frame(do.call("rbind", lapply(1:length(horiz), function(i) rbind(horiz[[i]], c(NA, NA))) ))
names(df_verti) = names(df_horiz) = c("s1", "s2")

grid1 = ggplot(df_verti, aes(x=s1,y=s2)) +
  geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(data = df_horiz, mapping = aes(x=s1,y=s2), colour = "gray80",
            inherit.aes = FALSE, linewidth = 0.4) +
  geom_path(data = df_contour, aes(x,y,group=Group, colour=z), linewidth = 0.8, inherit.aes = FALSE) +
  scale_color_viridis("Elevation (m)", discrete = FALSE,
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
        legend.position = "right",
        legend.background = element_rect(fill='transparent'),
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))
grid1

legend_grob <- get_legend(grid1)
grid1hat = grid.arrange(grid1+theme(legend.position = "none"), legend_grob,
                        ncol = 2,
                        widths = unit.c(unit.w1, unit(width1/3, "cm")),
                        heights = unit.c(unit.h1))

ggsave(paste0(pic_path, "original_space.pdf"),
       plot = grid1hat, width = width1+width1/3, height = height1, units = "cm")

# --- Warped space grid + warped elevation contours + reference sites ---
S.plot = S.warped

# Map the original grid polylines through the fitted deformation f
df_verti = data.frame(do.call("rbind", lapply(1:length(verti), function(i) rbind(summary(d1, verti[[i]], F)$swarped, c(NA, NA))) ))
df_horiz = data.frame(do.call("rbind", lapply(1:length(horiz), function(i) rbind(summary(d1, horiz[[i]], F)$swarped, c(NA, NA))) ))
names(df_verti) = names(df_horiz) = c("s1", "s2")

grid2 = ggplot(df_verti, aes(x=s1,y=s2)) +
  geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(data = df_horiz, mapping = aes(x=s1,y=s2), colour = "gray80",
            inherit.aes = FALSE, linewidth = 0.4) +
  geom_path(data = df_contour, aes(xw,yw,group=Group, colour=z), linewidth = 0.8, inherit.aes = FALSE) +
  scale_color_viridis("Elevation (m)", discrete = FALSE,
                      breaks = c(1000, 3000, 5000), labels = c(1000, 3000, 5000)) +
  geom_point(data = data.frame(x = S.plot[ref.pts,1], y = S.plot[ref.pts,2]),
             aes(x, y), size = 2, shape = ref_shap, fill="red", color="black") +
  xlab(expression(f[n1])) + ylab(expression(f[n2])) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size),
        legend.position = "right",
        legend.background = element_rect(fill='transparent'),
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))
grid2

legend_grob <- get_legend(grid2)
grid2hat = grid.arrange(grid2+theme(legend.position = "none"), legend_grob,
                        ncol = 2,
                        widths = unit.c(unit.w1, unit(width1/3, "cm")),
                        heights = unit.c(unit.h1))

ggsave(paste0(pic_path, "nepal_MSP_warped_space.pdf"),
       plot = grid2hat, width = width1+width1/3, height = height1, units = "cm")

legend_grob <- get_legend(grid1)
grid3hat = grid.arrange(grid1+theme(legend.position = "none"),
                        grid2+theme(legend.position = "none"), legend_grob,
                        ncol = 3,
                        widths = unit.c(unit.w1, unit.w1, unit(width1/3, "cm")),
                        heights = unit.c(unit.h1))
ggsave(paste0(pic_path, "nepal_MSP_spaces.pdf"),
       plot = grid3hat, width = width1+width1+width1/3, height = height1, units = "cm")



# --- Empirical extremal coefficient (EC) maps (two reference sites) ---
# Reads precomputed empirical EC pairs; expects first column concatenating the upper-triangular ECs
emp_extdep_filename = paste0(app_data, "_", model, "_empextdep.rds")
ec.mat.all = readRDS(file = emp_extdep_filename)
ec.emp.all = ec.mat.all[,1]

# Rebuild full symmetric EC matrix with diagonal 1
ec.uppermat = matrix(0, nrow(S), nrow(S))
ec.uppermat[lower.tri(ec.uppermat, diag=FALSE)] <- ec.emp.all
ec.uppermat <- t(ec.uppermat)
ec.wholemat = ec.uppermat + t(ec.uppermat)
diag(ec.wholemat) = 1
rm(ec.uppermat)

my_colors = RColorBrewer::brewer.pal(n=5, name="RdYlBu")[1:5]

# --- EC map for reference site 1 ---
ref_id = 1
data.plot = data.frame(s1 = S[,1], s2 = S[,2], ec = ec.wholemat[, ref.pts[ref_id]])
p.emp10 = eval(substitute(
  ggplot(data = data.plot) +
    geom_point(aes(s1, s2, color = ec),
               alpha = 0.9, size = 2, shape = 15) +
    scale_color_gradientn(colors = my_colors,
                          name = expression(EC(bold(s)[0], bold(s))),
                          limits = c(1,2),
                          breaks = c(1.00, 1.5, 2.00),
                          labels = c("1.00", "1.50", "2.00")) +
    geom_point(aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
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
          legend.position = "right",
          legend.background = element_rect(fill='transparent'),
          legend.direction = "vertical",
          legend.box = "vertical",
          legend.spacing.y = unit(0.4, "lines"),
          legend.margin = margin(5, 5, 5, 5)),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.emp10)
p.emp1 = grid.arrange(p.emp10+theme(legend.position = "none"), legend_grob,
                      ncol = 2,
                      widths = unit.c(unit.w1, unit(width1/3, "cm")),
                      heights = unit.c(unit.h1))
ggsave(paste0(pic_path, "nepal_empec_", ref.pts[ref_id], ".pdf"),
       plot = p.emp1, width = width1+width1/3, height = height1, units = "cm")

# --- Fitted EC map for reference site 1 based on warped distances ---
D2 = rdist(S.warped)
ec.fit = sapply(1:nrow(D2), function(i) extcoef(c(range_fitted, dof_fitted), D2[ref.pts[ref_id], i]))
data.plot = data.frame(s1 = S[,1], s2 = S[,2], ec = ec.fit)
p.fit10 = eval(substitute(
  ggplot(data = data.plot) +
    geom_point(aes(s1, s2, color = ec),
               alpha = 0.9, size = 2, shape = 15) +
    scale_color_gradientn(colors = my_colors,
                          name = expression(EC(bold(s)[0], bold(s))),
                          limits = c(1,2),
                          breaks = c(1.00, 1.5, 2.00),
                          labels = c("1.00", "1.50", "2.00")) +
    geom_point(aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
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
          legend.position = "right",
          legend.background = element_rect(fill='transparent'),
          legend.direction = "vertical",
          legend.box = "vertical",
          legend.spacing.y = unit(0.4, "lines"),
          legend.margin = margin(5, 5, 5, 5)),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.fit10)
p.fit1 = grid.arrange(p.fit10+theme(legend.position = "none"), legend_grob,
                      ncol = 2,
                      widths = unit.c(unit.w1, unit(width1/3, "cm")),
                      heights = unit.c(unit.h1))
ggsave(paste0(pic_path, "nepal_fitec_", ref.pts[ref_id], ".pdf"),
       plot = p.fit1, width = width1+width1/3, height = height1, units = "cm")

# --- EC & fitted EC maps for reference site 2 ---
ref_id = 2
data.plot = data.frame(s1 = S[,1], s2 = S[,2], ec = ec.wholemat[, ref.pts[ref_id]])
p.emp20 = eval(substitute(
  ggplot(data = data.plot) +
    geom_point(aes(s1, s2, color = ec),
               alpha = 0.9, size = 2, shape = 15) +
    scale_color_gradientn(colors = my_colors,
                          name = expression(EC(bold(s)[0], bold(s))),
                          limits = c(1,2),
                          breaks = c(1.00, 1.5, 2.00),
                          labels = c("1.00", "1.50", "2.00")) +
    geom_point(aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
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
          legend.position = "right",
          legend.background = element_rect(fill='transparent'),
          legend.direction = "vertical",
          legend.box = "vertical",
          legend.spacing.y = unit(0.4, "lines"),
          legend.margin = margin(5, 5, 5, 5)),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.emp20)
p.emp2 = grid.arrange(p.emp20+theme(legend.position = "none"), legend_grob,
                      ncol = 2,
                      widths = unit.c(unit.w1, unit(width1/3, "cm")),
                      heights = unit.c(unit.h1))
ggsave(paste0(pic_path, "nepal_empec_", ref.pts[ref_id], ".pdf"),
       plot = p.emp2, width = width1+width1/3, height = height1, units = "cm")

D2 = rdist(S.warped)
ec.fit = sapply(1:nrow(D2), function(i) extcoef(c(range_fitted, dof_fitted), D2[ref.pts[ref_id], i]))
data.plot = data.frame(s1 = S[,1], s2 = S[,2], ec = ec.fit)
p.fit20 = eval(substitute(
  ggplot(data = data.plot) +
    geom_point(aes(s1, s2, color = ec),
               alpha = 0.9, size = 2, shape = 15) +
    scale_color_gradientn(colors = my_colors,
                          name = expression(EC(bold(s)[0], bold(s))),
                          limits = c(1,2),
                          breaks = c(1.00, 1.5, 2.00),
                          labels = c("1.00", "1.50", "2.00")) +
    geom_point(aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
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
          legend.position = "right",
          legend.background = element_rect(fill='transparent'),
          legend.direction = "vertical",
          legend.box = "vertical",
          legend.spacing.y = unit(0.4, "lines"),
          legend.margin = margin(5, 5, 5, 5)),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.fit20)
p.fit2 = grid.arrange(p.fit20+theme(legend.position = "none"), legend_grob,
                      ncol = 2,
                      widths = unit.c(unit.w1, unit(width1/3, "cm")),
                      heights = unit.c(unit.h1))
ggsave(paste0(pic_path, "nepal_fitec_", ref.pts[ref_id], ".pdf"),
       plot = p.fit2, width = width1+width1/3, height = height1, units = "cm")

# --- Pair-clouds: empirical EC vs. distance in original and warped spaces ---
str(ec.mat.all)

# Empirical FMADogram in original and warped spaces; build EC–distance scatter
fmad   <- fmadogram(data = t(Z.max), coord = as.matrix(S.rescaled))
fmad.w <- fmadogram(data = t(Z.max), coord = as.matrix(S.warped))

# Original space cloud
distances  = fmad[,1]
extcoeffs  <- pmin(fmad[,3], 2)
ec.emp.mat = rbind(extcoeffs, distances)
ec.vec  = ec.emp.mat[1,]
distmat = ec.emp.mat[2,]
plot_samp = sample(1:length(ec.vec), 4000)
df_cloud1 = data.frame(EC=ec.vec[plot_samp], distance=distmat[plot_samp])
pcloud1 = ggplot(df_cloud1, aes(x=distance, y=EC)) + geom_point(shape=1) + theme_bw() +
  xlab("Distance") + ylab("Extremal Coefficient") +
  theme(plot.title = element_text(hjust = 0.5, size=25),,
        legend.key.size = unit(0.1, "in"), axis.title=element_text(size=16),
        axis.text=element_text(size=14), legend.text = element_text(size=12),
        legend.title = element_text(size=12), legend.position = "right")
pcloud1

# Warped space cloud
distances.w  = fmad.w[,1]
extcoeffs.w  <- pmin(fmad.w[,3], 2)
ec.emp.mat.w = rbind(extcoeffs.w, distances.w)
ec.vec  = ec.emp.mat.w[1,]
distmat = ec.emp.mat.w[2,]
plot_samp = sample(1:length(ec.vec), 4000)
df_cloud2 = data.frame(EC=ec.vec[plot_samp], distance=distmat[plot_samp])
pcloud2 = ggplot(df_cloud2, aes(x=distance, y=EC)) + geom_point(shape=1) + theme_bw() +
  xlab("Distance") + ylab("Extremal Coefficient") +
  theme(plot.title = element_text(hjust = 0.5, size=25),,
        legend.key.size = unit(0.1, "in"), axis.title=element_text(size=16),
        axis.text=element_text(size=14), legend.text = element_text(size=12),
        legend.title = element_text(size=12), legend.position = "right")
pcloud2

# Fitted EC curve in warped space for comparison
df.line.warped = data.frame(x = seq(0,1.4,0.01),
                            y = sapply(seq(0,1.4,0.01), function(i) extcoef(c(range_fitted, dof_fitted), i)))

# Overlay: original-cloud (circles), warped-cloud (triangles), fitted curve (line)
pcloud = ggplot(df_cloud1, aes(x=distance, y=EC)) + 
  geom_point(shape=1) + 
  geom_point(aes(x=df_cloud2$distance, y=df_cloud2$EC), shape=2, 
             color = "#2C7BB6", alpha = 0.4) +
  geom_line(df.line.warped, mapping = aes(x=x, y=y), 
            color = "#D7191C", linewidth=1) +
  theme_bw() +
  xlab("Distance") + ylab("Extremal Coefficient") +
  theme(plot.title = element_text(hjust = 0.5, size=25),,
        legend.key.size = unit(0.1, "in"), axis.title=element_text(size=16),
        axis.text=element_text(size=14), legend.text = element_text(size=12),
        legend.title = element_text(size=12), legend.position = "right")
pcloud

ggsave(paste0(pic_path, "nepal_cloud.pdf"),
       plot = pcloud, width = 1.2*width1, 
       height = height1, units = "cm")

# --- Uncertainty quantification for fitted EC via delta method (two refs) ---
my_colors = RColorBrewer::brewer.pal(n=5, name="BrBG")[5:1]

# Reference site 1
ref_id = 1
D.warped = rdist(S.warped)
grads_EC = sapply(1:nrow(D.warped), function(d) {
  grad_extcoef(c(range_fitted, dof_fitted), D.warped[ref.pts[ref_id], d])
})
var_extcoef = sapply(1:nrow(D.warped), function(d) {
  t(grads_EC[,d])%*%pred$Sigma_psi%*%grads_EC[,d]
})
data.plot = data.frame(s1 = S[, 1], s2 = S[, 2], sd = sqrt(var_extcoef))
p.sd10 = eval(substitute(
  ggplot(data = data.plot) +
    geom_point(aes(s1, s2, color = sd),
               alpha = 0.9, size = 2, shape = 15) +
    scale_color_gradientn(colors = my_colors,
                          name = expression(SD),
                          limits = range(data.plot$sd, na.rm = TRUE),
                          breaks = c(0.0125, 0.0075, 0.0025),
                          labels = c("0.0125", "0.0075", "0.0025")) +
    geom_point(aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]), 
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
          legend.position = "right",
          legend.background = element_rect(fill='transparent'),
          legend.direction = "vertical",
          legend.box = "vertical",
          legend.spacing.y = unit(0.4, "lines"),
          legend.margin = margin(5, 5, 5, 5)),
  list(ref_id = ref_id)
))
p.sd10
legend_grob <- get_legend(p.sd10)
p.sd1 = grid.arrange(p.sd10+theme(legend.position = "none"), legend_grob,
                     ncol = 2,
                     widths = unit.c(unit.w1, unit(width1/3, "cm")),
                     heights = unit.c(unit.h1))
ggsave(paste0(pic_path, "nepal_fitec_unc_", ref.pts[ref_id], ".pdf"),
       plot = p.sd1, width = width1+width1/3, height = height1, units = "cm")

# Reference site 2
ref_id = 2
D.warped = rdist(S.warped)
grads_EC = sapply(1:nrow(D.warped), function(d) {
  grad_extcoef(c(range_fitted, dof_fitted), D.warped[ref.pts[ref_id], d])
})
var_extcoef = sapply(1:nrow(D.warped), function(d) {
  t(grads_EC[,d])%*%pred$Sigma_psi%*%grads_EC[,d]
})
data.plot = data.frame(s1 = S[, 1], s2 = S[, 2], sd = sqrt(var_extcoef))
p.sd20 = eval(substitute(
  ggplot(data = data.plot) +
    geom_point(aes(s1, s2, color = sd),
               alpha = 0.9, size = 2, shape = 15) +
    scale_color_gradientn(colors = my_colors,
                          name = expression(SD),
                          limits = range(data.plot$sd, na.rm = TRUE),
                          breaks = c(0.0125, 0.0075, 0.0025),
                          labels = c("0.0125", "0.0075", "0.0025")) +
    geom_point(aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]), 
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
          legend.position = "right",
          legend.background = element_rect(fill='transparent'),
          legend.direction = "vertical",
          legend.box = "vertical",
          legend.spacing.y = unit(0.4, "lines"),
          legend.margin = margin(5, 5, 5, 5)),
  list(ref_id = ref_id)
))
p.sd20
legend_grob <- get_legend(p.sd20)
p.sd2 = grid.arrange(p.sd20+theme(legend.position = "none"), legend_grob,
                     ncol = 2,
                     widths = unit.c(unit.w1, unit(width1/3, "cm")),
                     heights = unit.c(unit.h1))
ggsave(paste0(pic_path, "nepal_fitec_unc_", ref.pts[ref_id], ".pdf"),
       plot = p.sd2, width = width1+width1/3, height = height1, units = "cm")

# --- 3x3 panel: empirical EC, fitted EC, and SD maps for both refs with shared legends ---
legend_grob1 <- get_legend(p.emp10)
legend_grob2 <- get_legend(p.sd10)
p.ext = grid.arrange(p.emp10+theme(legend.position = "none"),
                     p.emp20+theme(legend.position = "none"),
                     legend_grob1,
                     p.fit10+theme(legend.position = "none"), 
                     p.fit20+theme(legend.position = "none"), 
                     legend_grob1,
                     p.sd10+theme(legend.position = "none"), 
                     p.sd20+theme(legend.position = "none"), 
                     legend_grob2,
                     ncol = 3,
                     nrow = 3,
                     widths = unit.c(unit.w1, unit.w1, unit(width1/3, "cm")),
                     heights = unit.c(unit.h1, unit.h1, unit.h1))
ggsave(paste0(pic_path, "nepal_ext.pdf"),
       plot = p.ext, width = width1+width1+width1/3, height = 3*height1, units = "cm")


