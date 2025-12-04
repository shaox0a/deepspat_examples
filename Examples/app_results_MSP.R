###############################################
# 02_nepal_MSP_results.R
# - Load precomputed model summaries from *_fitresults.rds
# - Load map / grid data
# - Produce all figures (original vs warped space, EC maps, clouds, SD maps)
#   without re-fitting the model and without needing the object `d1`
###############################################

rm(list = ls())
# Set working directory to the repo root
# setwd(...)

###############################################
# Load libraries for plotting and diagnostics
###############################################
library(dplyr)
library(fields)

library(ggplot2)
library(verification)
library(ggmap)
library(ggpubr)
library(ggnewscale)
library(RColorBrewer)
library(viridis)
library(grid)
library(gridExtra)

# Custom helper functions: extcoef, grad_extcoef, fmadogram, edm_est, etc.
source("Examples/utils_ext.R")


###############################################
# Application & model labels
###############################################
model    <- "MSP-BR"
app_data <- "NepalExtended"


###############################################
# Load map-related objects (NepalMap.Rdata)
# Typically contains:
#  - S: locations
#  - df: grid df with "s1","s2","elev" for the background tile
#  - df_loc: locations in data frame form
#  - nepal: ggmap base map object
###############################################
load("Examples/NepalMap.Rdata")


###############################################
# Load numerical model summaries (no `d1` required)
###############################################
fit_results <- readRDS(paste0("Examples/", app_data, "_", model, "_fitresults.rds"))

# Overwrite S & df_loc with those used in the model fit (for safety)
S       <- fit_results$S
df_loc  <- fit_results$df_loc

# Coordinates and parameter estimates
S.rescaled   <- fit_results$S_rescaled
S.warped     <- fit_results$S_warped
range_fitted <- fit_results$range_fitted
dof_fitted   <- fit_results$dof_fitted
Sigma.psi    <- fit_results$Sigma_psi

# Elevation / contour information
df_elev         <- fit_results$df_elev
df_contour      <- fit_results$df_contour     # includes x,y,z,Group,level,xw,yw
df_verti_warped <- fit_results$df_verti_warped
df_horiz_warped <- fit_results$df_horiz_warped
ref.pts         <- fit_results$ref_pts


###############################################
# Load full data again for EC / FMADogram diagnostics
# (Z.max is needed for fmadogram)
###############################################
simnames <- load(file = "Examples/NepalExtended.rds")
# Now Z.max, S, etc. are available.
nrepli <- dim(Z.max)[2]


###############################################
# Output directory for figures
###############################################
pic_path <- "Examples/Pic_nepal_MSP/"
if (!dir.exists(pic_path)) {
  dir.create(pic_path)
}


###############################################
# Quick checks of site ordering and reference points
###############################################
plot(S)                           # All grid locations
points(S[c(348, 363), ], col = "red")  # Just a sanity check; not used later


###############################################
# Model summary & derived quantities
# (now loaded from fit_results instead of recomputing)
###############################################
# S.rescaled, S.warped, range_fitted, dof_fitted already loaded above.

# Pairwise distances in warped space (used for EC curves etc.)
D.warped <- rdist(S.warped)

# Optional visual check of warped coordinates
plot(S.warped)


###############################################
# Elevation contours (original and warped space)
# We do NOT recompute elevation or contours here.
# Instead, we use df_contour from the model script, which already
# contains:
#  - x, y, z, Group, level    (original-space contour vertices)
#  - xw, yw                   (warped-space contour vertices)
###############################################
# Example structure check:
# str(df_contour)


###############################################
# Figure layout constants and reference site indices
###############################################
width1  <- 11.5
unit.w1 <- unit(width1, "cm")
width2  <- 8
unit.w2 <- unit(width2, "cm")
height1 <- 8
unit.h1 <- unit(height1, "cm")

ref_shap  <- 21  # Shape for highlighted reference sites
ref_shap1 <- 8   # Unused, alternative shape

axis.title.size   <- 16
axis.text.size    <- 16
legend.text.size  <- 15
legend.title.size <- 16
text.size         <- 5   # Unused placeholder

# Reference sites used in EC maps, SD maps, etc.
ref.pts <- ref.pts


###############################################
# Basemap: elevation raster over Nepal with reference sites
###############################################
p.elev <- ggmap(nepal) +  # 'nepal' comes from NepalMap.Rdata
  geom_tile(
    df_elev,
    mapping = aes(x = s1, y = s2, fill = elev),
    width = 0.25, height = 0.25
  ) +
  geom_point(data = data.frame(x = S[ref.pts,1], y = S[ref.pts,2]),
             aes(x, y), 
             size = 2, shape = ref_shap, fill="red", color="black") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 25),
    axis.title = element_text(size = axis.title.size),
    axis.text  = element_text(size = axis.text.size),
    axis.line  = element_blank(),
    axis.ticks = element_blank(),
    legend.key.size = unit(0.25, "in"),
    legend.text  = element_text(size = legend.text.size),
    legend.title = element_text(size = legend.title.size),
    legend.position = "right",
    legend.background = element_rect(fill = 'transparent'),
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.4, "lines"),
    legend.margin     = margin(5, 5, 5, 5)
  ) +
  xlab("Longitude") + ylab("Latitude") +
  scale_fill_gradientn("Elevation (m)", colours = terrain.colors(10))
print(p.elev)

ggsave(
  paste0(pic_path, "nepal_elev.pdf"),
  plot   = p.elev,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


###############################################
# Original space grid + contours + reference sites
###############################################
S.plot <- S

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

# Add NA separator rows between polylines for ggplot
df_verti <- data.frame(
  do.call(
    "rbind",
    lapply(seq_along(verti), function(i) rbind(verti[[i]], c(NA, NA)))
  )
)
df_horiz <- data.frame(
  do.call(
    "rbind",
    lapply(seq_along(horiz), function(i) rbind(horiz[[i]], c(NA, NA)))
  )
)
names(df_verti) <- names(df_horiz) <- c("s1", "s2")

# Grid + contours plot in original coordinate space
grid1 <- ggplot(df_verti, aes(x = s1, y = s2)) +
  geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(
    data = df_horiz,
    mapping = aes(x = s1, y = s2),
    colour = "gray80",
    inherit.aes = FALSE,
    linewidth = 0.4
  ) +
  geom_path(
    data = df_contour,
    aes(x, y, group = Group, colour = z),
    linewidth = 0.8,
    inherit.aes = FALSE
  ) +
  scale_color_viridis(
    "Elevation (m)",
    discrete = FALSE,
    breaks  = c(1000, 3000, 5000),
    labels  = c(1000, 3000, 5000)
  ) +
  geom_point(
    data = data.frame(x = S.plot[ref.pts, 1], y = S.plot[ref.pts, 2]),
    aes(x, y),
    size = 2, shape = ref_shap, fill = "red", color = "black"
  ) +
  xlab("Longitude") + ylab("Latitude") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 25),
    axis.title = element_text(size = axis.title.size),
    axis.text  = element_text(size = axis.text.size),
    axis.line  = element_blank(),
    axis.ticks = element_blank(),
    legend.key.size = unit(0.25, "in"),
    legend.text  = element_text(size = legend.text.size),
    legend.title = element_text(size = legend.title.size),
    legend.position = "right",
    legend.background = element_rect(fill = 'transparent'),
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.4, "lines"),
    legend.margin     = margin(5, 5, 5, 5)
  )
grid1

legend_grob <- get_legend(grid1)
grid1hat <- grid.arrange(
  grid1 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "original_space.pdf"),
  plot   = grid1hat,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


###############################################
# Warped space grid + warped elevation contours + reference sites
###############################################
S.plot <- S.warped

# Use precomputed warped grid lines from fit_results
df_verti <- df_verti_warped
df_horiz <- df_horiz_warped

grid2 <- ggplot(df_verti, aes(x = s1, y = s2)) +
  geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(
    data = df_horiz,
    mapping = aes(x = s1, y = s2),
    colour = "gray80",
    inherit.aes = FALSE,
    linewidth = 0.4
  ) +
  geom_path(
    data = df_contour,
    aes(xw, yw, group = Group, colour = z),
    linewidth = 0.8,
    inherit.aes = FALSE
  ) +
  scale_color_viridis(
    "Elevation (m)",
    discrete = FALSE,
    breaks  = c(1000, 3000, 5000),
    labels  = c(1000, 3000, 5000)
  ) +
  geom_point(
    data = data.frame(x = S.plot[ref.pts, 1], y = S.plot[ref.pts, 2]),
    aes(x, y),
    size = 2, shape = ref_shap, fill = "red", color = "black"
  ) +
  xlab(expression(f[1])) + ylab(expression(f[2])) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 25),
    axis.title = element_text(size = axis.title.size),
    axis.text  = element_text(size = axis.text.size),
    axis.line  = element_blank(),
    legend.key.size = unit(0.25, "in"),
    legend.text  = element_text(size = legend.text.size),
    legend.title = element_text(size = legend.title.size),
    legend.position = "right",
    legend.background = element_rect(fill = 'transparent'),
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.4, "lines"),
    legend.margin     = margin(5, 5, 5, 5)
  )
grid2

legend_grob <- get_legend(grid2)
grid2hat <- grid.arrange(
  grid2 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "nepal_MSP_warped_space.pdf"),
  plot   = grid2hat,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)

# Side-by-side comparison: original vs warped space
legend_grob <- get_legend(grid1)
grid3hat <- grid.arrange(
  grid1 + theme(legend.position = "none"),
  grid2 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 3,
  widths = unit.c(unit.w1, unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_MSP_spaces.pdf"),
  plot   = grid3hat,
  width  = width1 + width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


###############################################
# Empirical extremal coefficient (EC) maps
# for two reference sites (using EDM from file)
###############################################
emp_extdep_filename <- paste0("Examples/", app_data, "_", model, "_empextdep.rds")
all_edm_est <- readRDS(file = emp_extdep_filename)
ec.emp.all <- all_edm_est$edm[, 1]

# Reconstruct full symmetric EC matrix with diagonal = 1
ec.uppermat <- matrix(0, nrow(S), nrow(S))
ec.uppermat[lower.tri(ec.uppermat, diag = FALSE)] <- ec.emp.all
ec.uppermat <- t(ec.uppermat)
ec.wholemat <- ec.uppermat + t(ec.uppermat)
diag(ec.wholemat) <- 1
rm(ec.uppermat)

my_colors <- RColorBrewer::brewer.pal(n = 5, name = "RdYlBu")[1:5]


#######################
# EC map: reference site 1
#######################
ref_id <- 1
data.plot <- data.frame(
  s1 = S[, 1],
  s2 = S[, 2],
  ec = ec.wholemat[, ref.pts[ref_id]]
)
p.emp10 <- eval(substitute(
  ggplot(data = data.plot) +
    geom_point(
      aes(s1, s2, color = ec),
      alpha = 0.9, size = 2, shape = 15
    ) +
    scale_color_gradientn(
      colors = my_colors,
      name   = expression(EC(bold(s)[0], bold(s))),
      limits = c(1, 2),
      breaks = c(1.00, 1.5, 2.00),
      labels = c("1.00", "1.50", "2.00")
    ) +
    geom_point(
      aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
      size = 2, shape = ref_shap, fill = "red", color = "black"
    ) +
    theme_bw() + coord_fixed() +
    xlab("Longitude") + ylab("Latitude") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 25),
      axis.title = element_text(size = axis.title.size),
      axis.text  = element_text(size = axis.text.size),
      axis.line  = element_blank(),
      axis.ticks = element_blank(),
      legend.key.size = unit(0.25, "in"),
      legend.text  = element_text(size = legend.text.size),
      legend.title = element_text(size = legend.title.size,
                                  margin = margin(b = 10)),
      legend.position = "right",
      legend.background = element_rect(fill = 'transparent'),
      legend.direction  = "vertical",
      legend.box        = "vertical",
      legend.spacing.y  = unit(0.4, "lines"),
      legend.margin     = margin(5, 5, 5, 5)
    ),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.emp10)
p.emp1 <- grid.arrange(
  p.emp10 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_empec_", ref.pts[ref_id], ".pdf"),
  plot   = p.emp1,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


#######################
# Fitted EC map: reference site 1 (warped distances)
#######################
D2 <- rdist(S.warped)
ec.fit <- sapply(
  1:nrow(D2),
  function(i) extcoef(c(range_fitted, dof_fitted), D2[ref.pts[ref_id], i])
)
data.plot <- data.frame(s1 = S[, 1], s2 = S[, 2], ec = ec.fit)
p.fit10 <- eval(substitute(
  ggplot(data = data.plot) +
    geom_point(
      aes(s1, s2, color = ec),
      alpha = 0.9, size = 2, shape = 15
    ) +
    scale_color_gradientn(
      colors = my_colors,
      name   = expression(EC(bold(s)[0], bold(s))),
      limits = c(1, 2),
      breaks = c(1.00, 1.5, 2.00),
      labels = c("1.00", "1.50", "2.00")
    ) +
    geom_point(
      aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
      size = 2, shape = ref_shap, fill = "red", color = "black"
    ) +
    theme_bw() + coord_fixed() +
    xlab("Longitude") + ylab("Latitude") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 25),
      axis.title = element_text(size = axis.title.size),
      axis.text  = element_text(size = axis.text.size),
      axis.line  = element_blank(),
      axis.ticks = element_blank(),
      legend.key.size = unit(0.25, "in"),
      legend.text  = element_text(size = legend.text.size),
      legend.title = element_text(size = legend.title.size,
                                  margin = margin(b = 10)),
      legend.position = "right",
      legend.background = element_rect(fill = 'transparent'),
      legend.direction  = "vertical",
      legend.box        = "vertical",
      legend.spacing.y  = unit(0.4, "lines"),
      legend.margin     = margin(5, 5, 5, 5)
    ),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.fit10)
p.fit1 <- grid.arrange(
  p.fit10 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_fitec_", ref.pts[ref_id], ".pdf"),
  plot   = p.fit1,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


#######################
# EC & fitted EC maps: reference site 2
#######################
ref_id <- 2
data.plot <- data.frame(
  s1 = S[, 1],
  s2 = S[, 2],
  ec = ec.wholemat[, ref.pts[ref_id]]
)
p.emp20 <- eval(substitute(
  ggplot(data = data.plot) +
    geom_point(
      aes(s1, s2, color = ec),
      alpha = 0.9, size = 2, shape = 15
    ) +
    scale_color_gradientn(
      colors = my_colors,
      name   = expression(EC(bold(s)[0], bold(s))),
      limits = c(1, 2),
      breaks = c(1.00, 1.5, 2.00),
      labels = c("1.00", "1.50", "2.00")
    ) +
    geom_point(
      aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
      size = 2, shape = ref_shap, fill = "red", color = "black"
    ) +
    theme_bw() + coord_fixed() +
    xlab("Longitude") + ylab("Latitude") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 25),
      axis.title = element_text(size = axis.title.size),
      axis.text  = element_text(size = axis.text.size),
      axis.line  = element_blank(),
      axis.ticks = element_blank(),
      legend.key.size = unit(0.25, "in"),
      legend.text  = element_text(size = legend.text.size),
      legend.title = element_text(size = legend.title.size,
                                  margin = margin(b = 10)),
      legend.position = "right",
      legend.background = element_rect(fill = 'transparent'),
      legend.direction  = "vertical",
      legend.box        = "vertical",
      legend.spacing.y  = unit(0.4, "lines"),
      legend.margin     = margin(5, 5, 5, 5)
    ),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.emp20)
p.emp2 <- grid.arrange(
  p.emp20 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_empec_", ref.pts[ref_id], ".pdf"),
  plot   = p.emp2,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)

D2 <- rdist(S.warped)
ec.fit <- sapply(
  1:nrow(D2),
  function(i) extcoef(c(range_fitted, dof_fitted), D2[ref.pts[ref_id], i])
)
data.plot <- data.frame(s1 = S[, 1], s2 = S[, 2], ec = ec.fit)
p.fit20 <- eval(substitute(
  ggplot(data = data.plot) +
    geom_point(
      aes(s1, s2, color = ec),
      alpha = 0.9, size = 2, shape = 15
    ) +
    scale_color_gradientn(
      colors = my_colors,
      name   = expression(EC(bold(s)[0], bold(s))),
      limits = c(1, 2),
      breaks = c(1.00, 1.5, 2.00),
      labels = c("1.00", "1.50", "2.00")
    ) +
    geom_point(
      aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
      size = 2, shape = ref_shap, fill = "red", color = "black"
    ) +
    theme_bw() + coord_fixed() +
    xlab("Longitude") + ylab("Latitude") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 25),
      axis.title = element_text(size = axis.title.size),
      axis.text  = element_text(size = axis.text.size),
      axis.line  = element_blank(),
      axis.ticks = element_blank(),
      legend.key.size = unit(0.25, "in"),
      legend.text  = element_text(size = legend.text.size),
      legend.title = element_text(size = legend.title.size,
                                  margin = margin(b = 10)),
      legend.position = "right",
      legend.background = element_rect(fill = 'transparent'),
      legend.direction  = "vertical",
      legend.box        = "vertical",
      legend.spacing.y  = unit(0.4, "lines"),
      legend.margin     = margin(5, 5, 5, 5)
    ),
  list(ref_id = ref_id)
))
legend_grob <- get_legend(p.fit20)
p.fit2 <- grid.arrange(
  p.fit20 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_fitec_", ref.pts[ref_id], ".pdf"),
  plot   = p.fit2,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


###############################################
# Pair-clouds: empirical EC vs distance
# (original vs warped space, with fitted curve)
###############################################
# FMADogram in original and warped spaces
fmad   <- fmadogram(data = t(Z.max), coord = as.matrix(S.rescaled))
fmad.w <- fmadogram(data = t(Z.max), coord = as.matrix(S.warped))

# Original space cloud
distances <- fmad[, 1]
extcoeffs <- pmin(fmad[, 3], 2)
ec.emp.mat <- rbind(extcoeffs, distances)
ec.vec  <- ec.emp.mat[1, ]
distmat <- ec.emp.mat[2, ]
plot_samp <- sample(1:length(ec.vec), 4000)
df_cloud1 <- data.frame(
  EC       = ec.vec[plot_samp],
  distance = distmat[plot_samp]
)
pcloud1 <- ggplot(df_cloud1, aes(x = distance, y = EC)) +
  geom_point(shape = 1) + theme_bw() +
  xlab("Distance") + ylab("Extremal Coefficient") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 25),
    legend.key.size = unit(0.1, "in"),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 14),
    legend.text  = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.position = "right"
  )
pcloud1

# Warped space cloud
distances.w   <- fmad.w[, 1]
extcoeffs.w   <- pmin(fmad.w[, 3], 2)
ec.emp.mat.w  <- rbind(extcoeffs.w, distances.w)
ec.vec        <- ec.emp.mat.w[1, ]
distmat       <- ec.emp.mat.w[2, ]
plot_samp     <- sample(1:length(ec.vec), 4000)
df_cloud2 <- data.frame(
  EC       = ec.vec[plot_samp],
  distance = distmat[plot_samp]
)
pcloud2 <- ggplot(df_cloud2, aes(x = distance, y = EC)) +
  geom_point(shape = 1) + theme_bw() +
  xlab("Distance") + ylab("Extremal Coefficient") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 25),
    legend.key.size = unit(0.1, "in"),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 14),
    legend.text  = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.position = "right"
  )
pcloud2

# Fitted EC curve in warped space
df.line.warped <- data.frame(
  x = seq(0, 1.4, 0.01),
  y = sapply(
    seq(0, 1.4, 0.01),
    function(i) extcoef(c(range_fitted, dof_fitted), i)
  )
)

# Overlay: original cloud (circles), warped cloud (triangles), fitted curve (line)
pcloud <- ggplot(df_cloud1, aes(x = distance, y = EC)) +
  geom_point(shape = 1) +
  geom_point(
    aes(x = df_cloud2$distance, y = df_cloud2$EC),
    shape = 2, color = "#2C7BB6", alpha = 0.4
  ) +
  geom_line(
    df.line.warped,
    mapping = aes(x = x, y = y),
    color = "#D7191C", linewidth = 1
  ) +
  theme_bw() +
  xlab("Distance") + ylab("Extremal Coefficient") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 25),
    legend.key.size = unit(0.1, "in"),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 14),
    legend.text  = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.position = "right"
  )
pcloud

ggsave(
  paste0(pic_path, "nepal_cloud.pdf"),
  plot   = pcloud,
  width  = 1.2 * width1,
  height = height1,
  units  = "cm"
)


###############################################
# Uncertainty of fitted EC via delta method
# for two reference sites
###############################################
my_colors <- RColorBrewer::brewer.pal(n = 5, name = "BrBG")[5:1]


#######################
# Uncertainty: reference site 1
#######################
ref_id <- 1
D.warped <- rdist(S.warped)

# Gradient of EC wrt parameters for each location (warped distance)
grads_EC <- sapply(
  1:nrow(D.warped),
  function(d) grad_extcoef(c(range_fitted, dof_fitted),
                           D.warped[ref.pts[ref_id], d])
)

# Delta-method variance: g'(psi)^T Sigma_psi g'(psi)
var_extcoef <- sapply(
  1:nrow(D.warped),
  function(d) {
    t(grads_EC[, d]) %*% Sigma.psi %*% grads_EC[, d]
  }
)
data.plot <- data.frame(
  s1 = S[, 1],
  s2 = S[, 2],
  sd = sqrt(var_extcoef)
)
p.sd10 <- eval(substitute(
  ggplot(data = data.plot) +
    geom_point(
      aes(s1, s2, color = sd),
      alpha = 0.9, size = 2, shape = 15
    ) +
    scale_color_gradientn(
      colors = my_colors,
      name   = expression(SD),
      limits = range(data.plot$sd, na.rm = TRUE),
      breaks = c(0.0125, 0.0075, 0.0025),
      labels = c("0.0125", "0.0075", "0.0025")
    ) +
    geom_point(
      aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
      size = 2, shape = ref_shap, fill = "red", color = "black"
    ) +
    theme_bw() + coord_fixed() +
    xlab("Longitude") + ylab("Latitude") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 25),
      axis.title = element_text(size = axis.title.size),
      axis.text  = element_text(size = axis.text.size),
      axis.line  = element_blank(),
      axis.ticks = element_blank(),
      legend.key.size = unit(0.25, "in"),
      legend.text  = element_text(size = legend.text.size),
      legend.title = element_text(size = legend.title.size,
                                  margin = margin(b = 10)),
      legend.position = "right",
      legend.background = element_rect(fill = 'transparent'),
      legend.direction  = "vertical",
      legend.box        = "vertical",
      legend.spacing.y  = unit(0.4, "lines"),
      legend.margin     = margin(5, 5, 5, 5)
    ),
  list(ref_id = ref_id)
))
p.sd10
legend_grob <- get_legend(p.sd10)
p.sd1 <- grid.arrange(
  p.sd10 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_fitec_unc_", ref.pts[ref_id], ".pdf"),
  plot   = p.sd1,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


#######################
# Uncertainty: reference site 2
#######################
ref_id <- 2
D.warped <- rdist(S.warped)
grads_EC <- sapply(
  1:nrow(D.warped),
  function(d) grad_extcoef(c(range_fitted, dof_fitted),
                           D.warped[ref.pts[ref_id], d])
)
var_extcoef <- sapply(
  1:nrow(D.warped),
  function(d) {
    t(grads_EC[, d]) %*% Sigma.psi %*% grads_EC[, d]
  }
)
data.plot <- data.frame(
  s1 = S[, 1],
  s2 = S[, 2],
  sd = sqrt(var_extcoef)
)
p.sd20 <- eval(substitute(
  ggplot(data = data.plot) +
    geom_point(
      aes(s1, s2, color = sd),
      alpha = 0.9, size = 2, shape = 15
    ) +
    scale_color_gradientn(
      colors = my_colors,
      name   = expression(SD),
      limits = range(data.plot$sd, na.rm = TRUE),
      breaks = c(0.0125, 0.0075, 0.0025),
      labels = c("0.0125", "0.0075", "0.0025")
    ) +
    geom_point(
      aes(x = s1[ref.pts[ref_id]], y = s2[ref.pts[ref_id]]),
      size = 2, shape = ref_shap, fill = "red", color = "black"
    ) +
    theme_bw() + coord_fixed() +
    xlab("Longitude") + ylab("Latitude") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 25),
      axis.title = element_text(size = axis.title.size),
      axis.text  = element_text(size = axis.text.size),
      axis.line  = element_blank(),
      axis.ticks = element_blank(),
      legend.key.size = unit(0.25, "in"),
      legend.text  = element_text(size = legend.text.size),
      legend.title = element_text(size = legend.title.size,
                                  margin = margin(b = 10)),
      legend.position = "right",
      legend.background = element_rect(fill = 'transparent'),
      legend.direction  = "vertical",
      legend.box        = "vertical",
      legend.spacing.y  = unit(0.4, "lines"),
      legend.margin     = margin(5, 5, 5, 5)
    ),
  list(ref_id = ref_id)
))
p.sd20
legend_grob <- get_legend(p.sd20)
p.sd2 <- grid.arrange(
  p.sd20 + theme(legend.position = "none"),
  legend_grob,
  ncol   = 2,
  widths = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_fitec_unc_", ref.pts[ref_id], ".pdf"),
  plot   = p.sd2,
  width  = width1 + width1 / 3,
  height = height1,
  units  = "cm"
)


###############################################
# 3x3 panel: empirical EC, fitted EC, SD maps
# for both reference sites
###############################################
legend_grob1 <- get_legend(p.emp10)  # EC legend
legend_grob2 <- get_legend(p.sd10)   # SD legend

p.ext <- grid.arrange(
  p.emp10 + theme(legend.position = "none"),
  p.emp20 + theme(legend.position = "none"),
  legend_grob1,
  p.fit10 + theme(legend.position = "none"),
  p.fit20 + theme(legend.position = "none"),
  legend_grob1,
  p.sd10 + theme(legend.position = "none"),
  p.sd20 + theme(legend.position = "none"),
  legend_grob2,
  ncol   = 3,
  nrow   = 3,
  widths = unit.c(unit.w1, unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1, unit.h1, unit.h1)
)
ggsave(
  paste0(pic_path, "nepal_ext.pdf"),
  plot   = p.ext,
  width  = width1 + width1 + width1 / 3,
  height = 3 * height1,
  units  = "cm"
)

cat("All figures saved under:", pic_path, "\n")
