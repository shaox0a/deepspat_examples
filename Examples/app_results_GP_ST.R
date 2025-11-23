############################################################
## File: nepal_results_ST_GP.R
## Role: Load saved numeric results and produce all plots
##       (original space grid + warped space + covariance maps)
############################################################

rm(list = ls())
setwd(this.path::here())

# -------------------------------------------------------------------
# Packages (only plotting / data manipulation)
# -------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(ggpubr)
library(ggnewscale)
library(RColorBrewer)
library(viridis)
library(grid)
library(gridExtra)

# -------------------------------------------------------------------
# Load data and precomputed plot objects
# -------------------------------------------------------------------

# Original dataset (as in the .rda provided)
load("NepalExtended_mean.rda")          # provides 'dataset'

# Objects precomputed by nepal_model_ST_GP.R for d3
load("Nepal_GP_d3_plot_data.rda")       # year_plot, ref.pts, df_contour_plot,
# df_verti_warped, df_horiz_warped,
# S_warped_year, corr_ref1, corr_ref2

year <- year_plot   # just a shorter name

# -------------------------------------------------------------------
# Plotting layout parameters
# -------------------------------------------------------------------

pic_path <- "Pic_nepal_GP_ST/"
if (!dir.exists(pic_path)) { dir.create(pic_path) }

width1  <- 11.5
unit.w1 <- unit(width1, "cm")
width2  <- 8
unit.w2 <- unit(width2, "cm")
height1 <- 8
unit.h1 <- unit(height1, "cm")

ref_shap  <- 21
ref_shap1 <- 8
axis.title.size  <- 16
axis.text.size   <- 16
legend.text.size <- 15
legend.title.size <- 16
text.size <- 5

# -------------------------------------------------------------------
# Common objects for plots
# -------------------------------------------------------------------

# Elevation contours in original and warped space
df_contour <- df_contour_plot   # has x, y, z, Group, xw, yw

# Unique longitudes/latitudes for grid construction
uni.lon <- unique(dataset$s1)
uni.lat <- unique(dataset$s2)

# Reference sites (indices within year == year_plot)
ref.pts <- ref.pts

# -------------------------------------------------------------------
# 1) Original space: grid lines + elevation contours + reference sites
# -------------------------------------------------------------------

# Vertical grid lines (original space)
verti_orig <- lapply(seq_along(uni.lon), function(i) {
  data_tmp     <- data.frame(dataset[dataset$s1 == uni.lon[i], ])
  data_tmp_uni <- data_tmp %>% distinct(s1, s2)
  data_tmp_ord <- data_tmp_uni[order(data_tmp_uni$s2), ]
})

# Horizontal grid lines (original space)
horiz_orig <- lapply(seq_along(uni.lat), function(i) {
  data_tmp     <- data.frame(dataset[dataset$s2 == uni.lat[i], ])
  data_tmp_uni <- data_tmp %>% distinct(s1, s2)
  data_tmp_ord <- data_tmp_uni[order(data_tmp_uni$s1), ]
})

# Add NA rows between path segments for ggplot
df_verti_orig <- data.frame(
  do.call("rbind", lapply(seq_along(verti_orig), function(i) {
    rbind(verti_orig[[i]], c(NA, NA))
  }))
)
df_horiz_orig <- data.frame(
  do.call("rbind", lapply(seq_along(horiz_orig), function(i) {
    rbind(horiz_orig[[i]], c(NA, NA))
  }))
)

# Sites in the chosen year (original space)
S.plot.orig <- data.frame(dataset[dataset$year == year, c("s1", "s2")])

# Plot in original space
grid1 <- ggplot(df_verti_orig, aes(x = s1, y = s2)) +
  geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(data = df_horiz_orig, mapping = aes(x = s1, y = s2),
            colour = "gray80", inherit.aes = FALSE, linewidth = 0.4) +
  geom_path(data = df_contour, aes(x, y, group = Group, colour = z),
            linewidth = 0.8, inherit.aes = FALSE) +
  scale_color_viridis("Elevation (m)", discrete = FALSE,
                      breaks = c(1000, 3000, 5000),
                      labels = c(1000, 3000, 5000)) +
  geom_point(
    data = data.frame(x = S.plot.orig[ref.pts, 1],
                      y = S.plot.orig[ref.pts, 2]),
    aes(x, y),
    size = 2, shape = ref_shap, fill = "red", color = "black"
  ) +
  xlab("Longitude") + ylab("Latitude") +
  theme_bw() +
  theme(
    plot.title  = element_text(hjust = 0.5, size = 25),
    axis.title  = element_text(size = axis.title.size),
    axis.text   = element_text(size = axis.text.size),
    axis.line   = element_blank(),
    axis.ticks  = element_blank(),
    legend.key.size = unit(0.25, "in"),
    legend.text = element_text(size = legend.text.size),
    legend.title = element_text(size = legend.title.size),
    legend.position = "right",
    legend.background = element_rect(fill = "transparent"),
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.4, "lines"),
    legend.margin     = margin(5, 5, 5, 5)
  )

legend_grob <- get_legend(grid1)
grid1hat <- grid.arrange(
  grid1 + theme(legend.position = "none"),
  legend_grob,
  ncol = 2,
  widths  = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "original_space.pdf"),
  plot = grid1hat,
  width = width1 + width1 / 3, height = height1, units = "cm"
)

# -------------------------------------------------------------------
# 2) Warped space: warped grid + warped contours + reference sites
# -------------------------------------------------------------------

df_verti <- df_verti_warped
df_horiz <- df_horiz_warped
S.plot   <- S_warped_year   # has columns f1, f2; we index by position

grid2 <- ggplot(df_verti, aes(x = s1, y = s2)) +
  geom_path(colour = "gray80", linewidth = 0.4) +
  geom_path(data = df_horiz, mapping = aes(x = s1, y = s2),
            colour = "gray80", inherit.aes = FALSE, linewidth = 0.4) +
  geom_path(data = df_contour, aes(xw, yw, group = Group, colour = z),
            linewidth = 0.8, inherit.aes = FALSE) +
  scale_color_viridis("Elevation (m)", discrete = FALSE,
                      breaks = c(1000, 3000, 5000),
                      labels = c(1000, 3000, 5000)) +
  geom_point(
    data = data.frame(x = S.plot[ref.pts, 1],
                      y = S.plot[ref.pts, 2]),
    aes(x, y),
    size = 2, shape = ref_shap, fill = "red", color = "black"
  ) +
  xlab(expression(f[1])) + ylab(expression(f[2])) +
  theme_bw() +
  theme(
    plot.title  = element_text(hjust = 0.5, size = 25),
    axis.title  = element_text(size = axis.title.size),
    axis.text   = element_text(size = axis.text.size),
    axis.line   = element_blank(),
    axis.ticks  = element_blank(),
    legend.key.size = unit(0.25, "in"),
    legend.text = element_text(size = legend.text.size),
    legend.title = element_text(size = legend.title.size),
    legend.position = "right",
    legend.background = element_rect(fill = "transparent"),
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.4, "lines"),
    legend.margin     = margin(5, 5, 5, 5)
  )

legend_grob <- get_legend(grid2)
grid2hat <- grid.arrange(
  grid2 + theme(legend.position = "none"),
  legend_grob,
  ncol = 2,
  widths  = unit.c(unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "nepal_warped_space.pdf"),
  plot = grid2hat,
  width = width1 + width1 / 3, height = height1, units = "cm"
)

# Combined original vs warped space
legend_grob <- get_legend(grid1)
grid3hat <- grid.arrange(
  grid1 + theme(legend.position = "none"),
  grid2 + theme(legend.position = "none"),
  legend_grob,
  ncol = 3,
  widths  = unit.c(unit.w1, unit.w1, unit(width1 / 3, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "nepal_spaces.pdf"),
  plot = grid3hat,
  width = width1 + width1 + width1 / 3,
  height = height1, units = "cm"
)

# -------------------------------------------------------------------
# 3) Covariance heat maps using precomputed correlation vectors
# -------------------------------------------------------------------

my_colors <- RColorBrewer::brewer.pal(n = 5, name = "RdYlBu")[5:1]

newdata_year <- dataset %>% dplyr::filter(year == year)

# --- First reference point ---

ref.point1 <- ref.pts[1]

plot_corr1 <- ggplot(data = newdata_year[1:1419+(year-2004)*1419,]) +
  geom_point(
    aes(s1, s2, color = corr_ref1),
    alpha = 0.9, size = 2, shape = 15
  ) +
  scale_color_gradientn(
    colors = my_colors,
    name   = expression(Corr(Y(bold(s)[0], t), Y(bold(s), t))),
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    labels = c("0.00", "0.25", "0.50", "0.75", "1.00")
  ) +
  geom_point(
    aes(x = newdata_year$s1[ref.point1],
        y = newdata_year$s2[ref.point1]),
    size = 2, shape = ref_shap, fill = "red", color = "black"
  ) +
  theme_bw() + coord_fixed() +
  xlab("Longitude") + ylab("Latitude") +
  theme(
    plot.title  = element_text(hjust = 0.5, size = 25),
    axis.title  = element_text(size = axis.title.size),
    axis.text   = element_text(size = axis.text.size),
    axis.line   = element_blank(),
    axis.ticks  = element_blank(),
    legend.key.size = unit(0.25, "in"),
    legend.text = element_text(size = legend.text.size),
    legend.title = element_text(size = legend.title.size,
                                margin = margin(b = 10)),
    legend.position = "right",
    legend.background = element_rect(fill = "transparent"),
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.4, "lines"),
    legend.margin     = margin(5, 5, 5, 5)
  )

legend_grob <- get_legend(plot_corr1)
plot_corr11 <- grid.arrange(
  plot_corr1 + theme(legend.position = "none"),
  legend_grob,
  ncol = 2,
  widths  = unit.c(unit.w1, unit(width1 / 2.5, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "nepal_corr_", ref.point1, ".pdf"),
  plot = plot_corr11,
  width = width1 + width1 / 3, height = height1, units = "cm"
)

# --- Second reference point ---

ref.point2 <- ref.pts[2]

plot_corr2 <- ggplot(data = newdata_year[1:1419+(year-2004)*1419,]) +
  geom_point(
    aes(s1, s2, color = corr_ref2),
    alpha = 0.9, size = 2, shape = 15
  ) +
  scale_color_gradientn(
    colors = my_colors,
    name   = expression(Corr(Y(bold(s)[0]), Y(bold(s)))),
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    labels = c("0.00", "0.25", "0.50", "0.75", "1.00")
  ) +
  geom_point(
    aes(x = newdata_year$s1[ref.point2],
        y = newdata_year$s2[ref.point2]),
    size = 2, shape = ref_shap, fill = "red", color = "black"
  ) +
  theme_bw() + coord_fixed() +
  xlab("Longitude") + ylab("Latitude") +
  theme(
    plot.title  = element_text(hjust = 0.5, size = 25),
    axis.title  = element_text(size = axis.title.size),
    axis.text   = element_text(size = axis.text.size),
    axis.line   = element_blank(),
    axis.ticks  = element_blank(),
    legend.key.size = unit(0.25, "in"),
    legend.text = element_text(size = legend.text.size),
    legend.title = element_text(size = legend.title.size,
                                margin = margin(b = 10)),
    legend.position = "right",
    legend.background = element_rect(fill = "transparent"),
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.spacing.y  = unit(0.4, "lines"),
    legend.margin     = margin(5, 5, 5, 5)
  )

legend_grob <- get_legend(plot_corr2)
plot_corr21 <- grid.arrange(
  plot_corr2 + theme(legend.position = "none"),
  legend_grob,
  ncol = 2,
  widths  = unit.c(unit.w1, unit(width1 / 2.5, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "nepal_corr_", ref.point2, ".pdf"),
  plot = plot_corr21,
  width = width1 + width1 / 3, height = height1, units = "cm"
)

# Combined covariance plots
legend_grob <- get_legend(plot_corr1)
plot_corr <- grid.arrange(
  plot_corr1 + theme(legend.position = "none"),
  plot_corr2 + theme(legend.position = "none"),
  legend_grob,
  ncol = 3,
  widths  = unit.c(unit.w1, unit.w1, unit(width1 / 2, "cm")),
  heights = unit.c(unit.h1)
)

ggsave(
  paste0(pic_path, "nepal_corr", ".pdf"),
  plot = plot_corr,
  width = width1 + width1 + width1 / 2,
  height = height1, units = "cm"
)


